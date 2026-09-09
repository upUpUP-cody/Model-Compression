"""Real hellaswag@64 evaluate plugin."""

from __future__ import annotations

from typing import Any

from compression_harness.model_io import get_model, get_tokenizer
from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec


class RealEvaluatePlugin(CompressionPlugin):
    """Score with Phase B Evaluator (hellaswag@64) using in-memory model."""

    name = "evaluate_real"
    kind = "evaluate"

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        from compression_harness.evaluator import Evaluator

        baseline = float(step.get("baseline_score") or state.meta.get("baseline_score") or 0.0)
        max_drop = float(step.get("max_relative_drop") or state.meta.get("max_relative_drop") or 0.05)
        goal_doc = step.get("goal_doc") or state.meta.get("goal_doc") or {
            "goal": {
                "quality": {"max_relative_drop": max_drop},
                "evaluation": {"limit": 64, "primary_metric": "acc_norm"},
                "model": {"path": state.model_ref, "dtype": "bfloat16"},
            }
        }

        model = get_model(state)
        tokenizer = get_tokenizer(state)
        evaluator = Evaluator(use_real=True)
        recipe = {"layers": {"default": {"weight_bits": 8}}}
        result = evaluator.evaluate(
            state.model_ref,
            recipe,
            goal_doc,
            baseline_score=baseline if baseline > 0 else None,
            model=model,
            tokenizer=tokenizer,
        )
        score = float(result["score"])
        base = float(result["baseline_score"])
        drop = float(result["relative_drop"])
        ok = drop <= max_drop + 1e-12
        state.meta["baseline_score"] = base
        state.meta["last_score"] = score
        state.meta["last_relative_drop"] = drop
        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "baseline_score": base,
            "score": score,
            "relative_drop": drop,
            "near_lossless_ok": ok,
            "within_threshold": ok,
            "max_relative_drop": max_drop,
            "task": result.get("task", "hellaswag"),
            "limit": result.get("limit", 64),
            "message": "[OK] evaluate_real hellaswag",
        }
        return state, metrics
