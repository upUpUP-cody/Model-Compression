"""Real torchao INT8 short-step quantize plugin (layer window)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from compression_harness.model_io import get_model
from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec
from compression_harness.plugins.layer_utils import fqn_in_layer_range, layer_indices


def _layer_filter(
    lo: int,
    hi: int,
    *,
    exclude: Sequence[str] = ("lm_head",),
    skip_already_quantized: set[int] | None = None,
) -> Callable[[nn.Module, str], bool]:
    exclude_set = {str(x) for x in exclude}
    skip = skip_already_quantized or set()

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        parts = fqn.split(".")
        for ex in exclude_set:
            if fqn == ex or fqn.endswith("." + ex) or ex in parts:
                return False
        if not fqn_in_layer_range(fqn, lo, hi):
            return False
        # Skip layers already marked INT8 in bookkeeping
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                if int(parts[i + 1]) in skip:
                    return False
        return True

    return filter_fn


def apply_int8_to_layer_range(
    model: nn.Module,
    lo: int,
    hi: int,
    *,
    exclude: Sequence[str] = ("lm_head",),
    skip_layers: set[int] | None = None,
    group_size: int | None = None,
) -> None:
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    gs = group_size if group_size and group_size > 0 else None
    cfg = Int8WeightOnlyConfig(group_size=gs)
    print(f"[INFO] torchao INT8 layers=[{lo},{hi}] exclude={list(exclude)}")
    quantize_(
        model,
        cfg,
        filter_fn=_layer_filter(lo, hi, exclude=exclude, skip_already_quantized=skip_layers),
    )


class TorchAOLayerQuantizePlugin(CompressionPlugin):
    name = "quantize_torchao_layers"
    kind = "quantize"

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        model = get_model(state)
        indices = layer_indices(step.get("layer_range"))
        lo, hi = min(indices), max(indices)
        bits = int(step.get("weight_bits") or 8)
        if bits != 8:
            raise ValueError(f"Phase II only supports weight_bits=8; got {bits}")

        already = {i for i in indices if state.layer_bits.get(i, 16) <= 8}
        to_do = [i for i in indices if i not in already]
        if not to_do:
            return state, {
                "status": "ok",
                "plugin": self.name,
                "kind": self.kind,
                "layers": indices,
                "message": "[OK] quantize_torchao_layers no-op (already INT8)",
            }

        apply_int8_to_layer_range(
            model,
            min(to_do),
            max(to_do),
            exclude=tuple(step.get("exclude") or ["lm_head"]),
            skip_layers=set(state.layer_bits.keys()),
        )
        for i in to_do:
            state.layer_bits[i] = bits
        state.applied_steps.append(dict(step))
        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "layers": to_do,
            "weight_bits": bits,
            "layer_bits": {str(i): state.layer_bits[i] for i in to_do},
            "message": "[OK] quantize_torchao_layers applied",
        }
        return state, metrics
