"""Builtin evaluate plugin for iterative loop."""

from __future__ import annotations

from typing import Any, Callable

from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec


class EvaluatePlugin(CompressionPlugin):
    """Score relative to baseline.

    Default: synthetic score from sparsity/bits bookkeeping (no GPU).
    Inject ``score_fn(state) -> float`` via step or constructor for tests / real eval.
    """

    name = "evaluate_stub"
    kind = "evaluate"

    def __init__(self, score_fn: Callable[[ModelState], float] | None = None) -> None:
        self._score_fn = score_fn

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        baseline = float(step.get("baseline_score") or state.meta.get("baseline_score") or 1.0)
        if self._score_fn is not None:
            score = float(self._score_fn(state))
        elif step.get("force_score") is not None:
            score = float(step["force_score"])
        else:
            score = self._heuristic_score(state, baseline)

        drop = 0.0 if baseline <= 0 else max(0.0, (baseline - score) / baseline)
        max_drop = float(step.get("max_relative_drop") or state.meta.get("max_relative_drop") or 0.05)
        ok = drop <= max_drop + 1e-12

        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "baseline_score": baseline,
            "score": score,
            "relative_drop": drop,
            "near_lossless_ok": ok,
            "within_threshold": ok,
            "max_relative_drop": max_drop,
            "message": "[OK] evaluate_stub",
        }
        state.meta["last_score"] = score
        state.meta["last_relative_drop"] = drop
        return state, metrics

    @staticmethod
    def _heuristic_score(state: ModelState, baseline: float) -> float:
        """Mild degradation from accumulated sparsity / lower bits (for dry iterate)."""
        if not state.layer_sparsity and not state.layer_bits:
            return baseline
        spar_pen = sum(state.layer_sparsity.values()) * 0.002
        bit_pen = 0.0
        for b in state.layer_bits.values():
            if b < 16:
                bit_pen += (16 - b) * 0.0003
        return max(0.0, baseline - spar_pen - bit_pen)
