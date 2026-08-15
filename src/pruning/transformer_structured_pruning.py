"""Physical structured pruning for Qwen2-style Transformer blocks.

Prunable units:
- attention KV-groups (each group = num_key_value_groups query heads + 1 KV head)
- MLP intermediate dimensions (gate/up/down)

Shapes are rebuilt (not mask-sparsity). Shared HF config is not mutated per-layer;
each module's local attributes (num_key_value_groups / intermediate_size) are updated.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _is_qwen2_like(model: nn.Module) -> bool:
    name = type(model).__name__.lower()
    if "qwen2" in name or "qwen" in name:
        return True
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    return model_type in {"qwen2", "qwen3"}


def _base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise ValueError("model does not expose .model.layers or .layers")


def _get_module_by_name(root: nn.Module, layer_name: str) -> nn.Module:
    module: nn.Module = root
    try:
        for part in layer_name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(f"Unknown layer: {layer_name}") from error
    return module


def _replace_linear(
    old: nn.Linear,
    new_weight: torch.Tensor,
    new_bias: Optional[torch.Tensor],
) -> nn.Linear:
    out_features, in_features = new_weight.shape
    layer = nn.Linear(in_features, out_features, bias=new_bias is not None)
    layer = layer.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(new_weight)
        if new_bias is not None and layer.bias is not None:
            layer.bias.copy_(new_bias)
    return layer


class TransformerStructuredPruning:
    """Physical head-group and FFN-intermediate pruning for Qwen2-like models."""

    HEAD_SUFFIX = ".self_attn.heads"
    MLP_SUFFIX = ".mlp.intermediate"

    def __init__(self, model: nn.Module) -> None:
        if not _is_qwen2_like(model):
            raise ValueError("TransformerStructuredPruning requires a Qwen2-like model")
        self.model = model
        self.base = _base_model(model)
        self.pruning_masks: Dict[str, dict] = {}
        self._validate_structure()

    def _validate_structure(self) -> None:
        if not hasattr(self.base, "layers") or len(self.base.layers) < 1:
            raise ValueError("expected non-empty transformer layers")
        for index, layer in enumerate(self.base.layers):
            if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
                raise ValueError(f"layer {index} missing self_attn/mlp")
            attn = layer.self_attn
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim"):
                if not hasattr(attn, attr):
                    raise ValueError(f"self_attn missing {attr} at layer {index}")
            mlp = layer.mlp
            for attr in ("gate_proj", "up_proj", "down_proj"):
                if not hasattr(mlp, attr):
                    raise ValueError(f"mlp missing {attr} at layer {index}")

    def prunable_layer_names(self) -> List[str]:
        names: List[str] = []
        for index in range(len(self.base.layers)):
            names.append(f"layers.{index}{self.HEAD_SUFFIX}")
            names.append(f"layers.{index}{self.MLP_SUFFIX}")
        return names

    def _parse_name(self, layer_name: str) -> Tuple[str, int]:
        if layer_name.endswith(self.HEAD_SUFFIX):
            prefix = layer_name[: -len(self.HEAD_SUFFIX)]
            kind = "heads"
        elif layer_name.endswith(self.MLP_SUFFIX):
            prefix = layer_name[: -len(self.MLP_SUFFIX)]
            kind = "mlp"
        else:
            raise ValueError(f"unsupported prunable name: {layer_name}")
        parts = prefix.split(".")
        if len(parts) != 2 or parts[0] != "layers" or not parts[1].isdigit():
            raise ValueError(f"unsupported prunable name: {layer_name}")
        return kind, int(parts[1])

    def output_size(self, layer_name: str) -> int:
        kind, index = self._parse_name(layer_name)
        layer = self.base.layers[index]
        if kind == "heads":
            return self._num_query_heads(layer.self_attn)
        return int(layer.mlp.gate_proj.out_features)

    def _num_query_heads(self, attn: nn.Module) -> int:
        head_dim = int(attn.head_dim)
        out = int(attn.q_proj.out_features)
        if out % head_dim != 0:
            raise ValueError("q_proj.out_features is not divisible by head_dim")
        return out // head_dim

    def _num_kv_heads(self, attn: nn.Module) -> int:
        head_dim = int(attn.head_dim)
        out = int(attn.k_proj.out_features)
        if out % head_dim != 0:
            raise ValueError("k_proj.out_features is not divisible by head_dim")
        return out // head_dim

    def _group_size(self, attn: nn.Module) -> int:
        n_q = self._num_query_heads(attn)
        n_kv = self._num_kv_heads(attn)
        if n_kv < 1 or n_q % n_kv != 0:
            raise ValueError(f"invalid GQA layout n_q={n_q} n_kv={n_kv}")
        return n_q // n_kv

    @staticmethod
    def _validate_prune_ratio(prune_ratio: float) -> None:
        if isinstance(prune_ratio, bool) or not isinstance(prune_ratio, Real):
            raise TypeError("prune_ratio must be a real number")
        if not 0.0 <= float(prune_ratio) < 1.0:
            raise ValueError("prune_ratio must satisfy 0 <= ratio < 1")

    @staticmethod
    def _validate_keep_indices(size: int, keep_indices: Sequence[int]) -> List[int]:
        if len(keep_indices) < 1:
            raise ValueError("keep_indices must be non-empty")
        normalized = sorted({int(index) for index in keep_indices})
        if any(index < 0 or index >= size for index in normalized):
            raise ValueError(f"keep_indices out of range for size={size}")
        if len(normalized) != len(keep_indices):
            # allow duplicates by uniquifying, but reject if empty after
            pass
        return normalized

    def prune_heads(self, layer_name: str, keep_head_indices: Sequence[int]) -> None:
        kind, index = self._parse_name(layer_name)
        if kind != "heads":
            raise ValueError(f"{layer_name} is not a head pruning target")
        attn = self.base.layers[index].self_attn
        n_q = self._num_query_heads(attn)
        n_kv = self._num_kv_heads(attn)
        group = self._group_size(attn)
        keep_q = self._validate_keep_indices(n_q, keep_head_indices)

        # Require whole KV-groups so num_key_value_groups stays uniform.
        kept_groups: List[int] = []
        for kv_idx in range(n_kv):
            group_heads = list(range(kv_idx * group, (kv_idx + 1) * group))
            present = [h for h in group_heads if h in keep_q]
            if not present:
                continue
            if present != group_heads:
                raise ValueError(
                    "GQA head pruning requires keeping whole KV-groups "
                    f"(group {kv_idx} partial keep={present})"
                )
            kept_groups.append(kv_idx)
        if not kept_groups:
            raise ValueError("must keep at least one KV-group")

        keep_q_final = [h for kv_idx in kept_groups for h in range(kv_idx * group, (kv_idx + 1) * group)]
        head_dim = int(attn.head_dim)
        q_index = torch.tensor(
            [h * head_dim + d for h in keep_q_final for d in range(head_dim)],
            device=attn.q_proj.weight.device,
            dtype=torch.long,
        )
        kv_index = torch.tensor(
            [h * head_dim + d for h in kept_groups for d in range(head_dim)],
            device=attn.k_proj.weight.device,
            dtype=torch.long,
        )

        q_w = attn.q_proj.weight.data.index_select(0, q_index).contiguous()
        q_b = None if attn.q_proj.bias is None else attn.q_proj.bias.data.index_select(0, q_index).contiguous()
        k_w = attn.k_proj.weight.data.index_select(0, kv_index).contiguous()
        k_b = None if attn.k_proj.bias is None else attn.k_proj.bias.data.index_select(0, kv_index).contiguous()
        v_w = attn.v_proj.weight.data.index_select(0, kv_index).contiguous()
        v_b = None if attn.v_proj.bias is None else attn.v_proj.bias.data.index_select(0, kv_index).contiguous()
        o_w = attn.o_proj.weight.data.index_select(1, q_index).contiguous()
        o_b = None if attn.o_proj.bias is None else attn.o_proj.bias.data.detach().clone()

        attn.q_proj = _replace_linear(attn.q_proj, q_w, q_b)
        attn.k_proj = _replace_linear(attn.k_proj, k_w, k_b)
        attn.v_proj = _replace_linear(attn.v_proj, v_w, v_b)
        attn.o_proj = _replace_linear(attn.o_proj, o_w, o_b)
        attn.num_key_value_groups = group

        self.pruning_masks[layer_name] = {
            "type": "attn_kv_group",
            "indices": keep_q_final,
            "kept_kv_groups": kept_groups,
            "original_size": n_q,
            "retained_size": len(keep_q_final),
        }

    def prune_mlp_intermediate(self, layer_name: str, keep_dim_indices: Sequence[int]) -> None:
        kind, index = self._parse_name(layer_name)
        if kind != "mlp":
            raise ValueError(f"{layer_name} is not an MLP intermediate target")
        mlp = self.base.layers[index].mlp
        width = int(mlp.gate_proj.out_features)
        keep = self._validate_keep_indices(width, keep_dim_indices)
        index_tensor = torch.tensor(keep, device=mlp.gate_proj.weight.device, dtype=torch.long)

        gate_w = mlp.gate_proj.weight.data.index_select(0, index_tensor).contiguous()
        up_w = mlp.up_proj.weight.data.index_select(0, index_tensor).contiguous()
        down_w = mlp.down_proj.weight.data.index_select(1, index_tensor).contiguous()

        mlp.gate_proj = _replace_linear(mlp.gate_proj, gate_w, None)
        mlp.up_proj = _replace_linear(mlp.up_proj, up_w, None)
        mlp.down_proj = _replace_linear(mlp.down_proj, down_w, None)
        mlp.intermediate_size = len(keep)

        self.pruning_masks[layer_name] = {
            "type": "mlp_intermediate",
            "indices": keep,
            "original_size": width,
            "retained_size": len(keep),
        }

    def prune_block(self, layer_name: str, keep_indices: Sequence[int]) -> None:
        kind, _ = self._parse_name(layer_name)
        if kind == "heads":
            self.prune_heads(layer_name, keep_indices)
        else:
            self.prune_mlp_intermediate(layer_name, keep_indices)

    def keep_indices_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: torch.Tensor,
    ) -> List[int]:
        self._validate_prune_ratio(prune_ratio)
        scores = importance_scores.detach().reshape(-1)
        size = self.output_size(layer_name)
        if scores.numel() != size:
            raise ValueError(f"importance_scores must have {size} elements")
        kind, index = self._parse_name(layer_name)
        keep_count = max(1, int(round(size * (1.0 - float(prune_ratio)))))
        if kind == "heads":
            attn = self.base.layers[index].self_attn
            group = self._group_size(attn)
            # Round keep_count up to whole groups (never prune to zero groups).
            n_groups = size // group
            keep_groups = max(1, int(math.ceil(keep_count / group)))
            keep_groups = min(keep_groups, n_groups)
            # Rank groups by mean query-head importance inside the group.
            group_scores = []
            for g in range(n_groups):
                heads = range(g * group, (g + 1) * group)
                group_scores.append((float(scores[list(heads)].mean()), g))
            group_scores.sort(key=lambda item: (-item[0], item[1]))
            kept_groups = sorted(g for _, g in group_scores[:keep_groups])
            keep_indices = [h for g in kept_groups for h in range(g * group, (g + 1) * group)]
        else:
            ranked = sorted(range(size), key=lambda i: (-float(scores[i]), i))
            keep_indices = sorted(ranked[:keep_count])
        self.prune_block(layer_name, keep_indices)
        return keep_indices

    def prune_by_ratio(self, layer_name: str, prune_ratio: float) -> List[int]:
        kind, index = self._parse_name(layer_name)
        if kind == "heads":
            attn = self.base.layers[index].self_attn
            head_dim = int(attn.head_dim)
            weight = attn.q_proj.weight.detach()
            n_q = self._num_query_heads(attn)
            scores = torch.stack(
                [
                    torch.norm(weight[h * head_dim : (h + 1) * head_dim, :], p=2)
                    for h in range(n_q)
                ]
            )
        else:
            mlp = self.base.layers[index].mlp
            scores = torch.norm(mlp.gate_proj.weight.detach(), p=2, dim=1)
        return self.keep_indices_by_ratio(layer_name, prune_ratio, scores)

    def prune_by_layer(self, layer_ratios: Dict[str, float]) -> int:
        for name, ratio in layer_ratios.items():
            self._parse_name(name)
            self._validate_prune_ratio(ratio)
        ordered = [name for name in self.prunable_layer_names() if name in layer_ratios]
        for name in reversed(ordered):
            self.prune_by_ratio(name, layer_ratios[name])
        return self.get_structural_parameter_count()

    def prune_by_layer_indices(self, layer_keep_indices: Dict[str, Sequence[int]]) -> int:
        ordered = [name for name in self.prunable_layer_names() if name in layer_keep_indices]
        for name in reversed(ordered):
            self.prune_block(name, layer_keep_indices[name])
        return self.get_structural_parameter_count()

    def create_pruned_model_by_indices(
        self, layer_keep_indices: Dict[str, Sequence[int]]
    ) -> nn.Module:
        pruned_model = copy.deepcopy(self.model)
        TransformerStructuredPruning(pruned_model).prune_by_layer_indices(layer_keep_indices)
        return pruned_model

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module:
        pruned_model = copy.deepcopy(self.model)
        TransformerStructuredPruning(pruned_model).prune_by_layer(pruning_config)
        return pruned_model

    def get_structural_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
