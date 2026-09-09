"""Builtin quantize plugin (short-step; bookkeeping + optional torchao hook)."""

from __future__ import annotations

from typing import Any

from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec


class QuantizePlugin(CompressionPlugin):
    """Per-layer bit bookkeeping; real torchao apply is opt-in via step.real=true.

    Framework default keeps stub so iterate loops are fast and GPU-free in tests.
    """

    name = "quantize_stub"
    kind = "quantize"

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        lr = step.get("layer_range") or [0, 0]
        lo, hi = int(lr[0]), int(lr[1])
        bits = int(step.get("weight_bits") or 8)
        touched: list[int] = []
        for i in range(lo, hi + 1):
            state.layer_bits[i] = bits
            touched.append(i)
        state.applied_steps.append(dict(step))

        real = bool(step.get("real"))
        message = "[OK] quantize_stub bookkeeping (no weight write)"
        if real:
            message = (
                "[WARNING] real torchao per-layer path not wired in Phase I; "
                "bookkeeping only. Use Phase B global INT8 adapter for full quantize."
            )

        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "layers": touched,
            "weight_bits": bits,
            "layer_bits": {str(i): state.layer_bits[i] for i in touched},
            "message": message,
        }
        return state, metrics
