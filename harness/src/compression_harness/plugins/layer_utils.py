"""Layer helpers shared by real prune/quantize plugins."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def layer_indices(layer_range: Sequence[int] | None) -> List[int]:
    if not layer_range or len(layer_range) != 2:
        return [0]
    lo, hi = int(layer_range[0]), int(layer_range[1])
    if hi < lo:
        lo, hi = hi, lo
    return list(range(lo, hi + 1))


def mlp_intermediate_names(indices: Iterable[int]) -> List[str]:
    """Qwen Stage-A prunable MLP names: layers.{i}.mlp.intermediate."""
    return [f"layers.{int(i)}.mlp.intermediate" for i in indices]


def fqn_in_layer_range(fqn: str, lo: int, hi: int) -> bool:
    """True if module FQN belongs to model.layers.[lo..hi] (HF Qwen layout)."""
    # Common patterns: model.layers.3.mlp.gate_proj, layers.3.self_attn.q_proj
    parts = fqn.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = int(parts[i + 1])
            return lo <= idx <= hi
    return False
