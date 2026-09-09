"""Builtin prune plugin (short-step stub / bookkeeping)."""

from __future__ import annotations

from typing import Any

from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec


class PrunePlugin(CompressionPlugin):
    """Records per-layer sparsity increments without touching real weights in v0.

    Phase II will call Wanda. Framework still advances state for iterative loop tests.
    """

    name = "prune_stub"
    kind = "prune"

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        lr = step.get("layer_range") or [0, 0]
        lo, hi = int(lr[0]), int(lr[1])
        delta = float(step.get("sparsity_delta") or step.get("sparsity") or 0.05)
        touched: list[int] = []
        for i in range(lo, hi + 1):
            prev = float(state.layer_sparsity.get(i, 0.0))
            state.layer_sparsity[i] = min(0.95, prev + delta)
            touched.append(i)
        state.applied_steps.append(dict(step))
        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "layers": touched,
            "sparsity_delta": delta,
            "layer_sparsity": {str(i): state.layer_sparsity[i] for i in touched},
            "message": "[OK] prune_stub applied (bookkeeping only; no weight write)",
        }
        return state, metrics
