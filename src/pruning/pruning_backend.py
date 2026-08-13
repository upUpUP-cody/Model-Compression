"""Thin pruning backend abstraction for MLP and CNN models."""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from src.models.resnet_cifar import ResNetCIFAR
from src.pruning.cnn_structured_pruning import CnnStructuredPruning
from src.pruning.structured_pruning import StructuredPruning


@runtime_checkable
class PruningBackend(Protocol):
    def prunable_layer_names(self) -> List[str]: ...

    def output_size(self, layer_name: str) -> int: ...

    def create_pruned_model_by_indices(self, layer_keep_indices: Dict[str, Sequence[int]]) -> nn.Module: ...

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module: ...

    def keep_indices_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: torch.Tensor,
    ) -> List[int]: ...


class MlpBackend:
    def __init__(self, model: nn.Module) -> None:
        self._pruner = StructuredPruning(model)
        self.model = model

    def prunable_layer_names(self) -> List[str]:
        return self._pruner._hidden_linear_layer_names()

    def output_size(self, layer_name: str) -> int:
        layer = self._pruner._validate_hidden_linear(layer_name)
        return int(layer.out_features)

    def create_pruned_model_by_indices(self, layer_keep_indices: Dict[str, Sequence[int]]) -> nn.Module:
        return self._pruner.create_pruned_model_by_indices(layer_keep_indices)

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module:
        return self._pruner.create_pruned_model(pruning_config)

    def keep_indices_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: torch.Tensor,
    ) -> List[int]:
        return self._pruner.prune_mlp_by_ratio(layer_name, prune_ratio, importance_scores)


class CnnBackend:
    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, ResNetCIFAR):
            raise ValueError("CnnBackend requires ResNetCIFAR")
        self._pruner = CnnStructuredPruning(model)
        self.model = model

    def prunable_layer_names(self) -> List[str]:
        return self._pruner.prunable_layer_names()

    def output_size(self, layer_name: str) -> int:
        layer = self._pruner._get_layer(layer_name)
        if not isinstance(layer, nn.Conv2d):
            raise ValueError(f"{layer_name} is not Conv2d")
        return int(layer.out_channels)

    def create_pruned_model_by_indices(self, layer_keep_indices: Dict[str, Sequence[int]]) -> nn.Module:
        return self._pruner.create_pruned_model_by_indices(layer_keep_indices)

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module:
        return self._pruner.create_pruned_model(pruning_config)

    def keep_indices_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: torch.Tensor,
    ) -> List[int]:
        scores = importance_scores.detach().reshape(-1)
        layer = self._pruner._get_layer(layer_name)
        if not isinstance(layer, nn.Conv2d):
            raise ValueError(f"{layer_name} is not Conv2d")
        if scores.numel() != layer.out_channels:
            raise ValueError("importance_scores must match out_channels")
        keep_count = max(1, int(round(layer.out_channels * (1.0 - float(prune_ratio)))))
        ranked = sorted(
            range(layer.out_channels),
            key=lambda index: (-float(scores[index]), index),
        )
        keep_indices = sorted(ranked[:keep_count])
        self._pruner.prune_conv_block(layer_name, keep_indices)
        return keep_indices


def resolve_pruning_backend(model: nn.Module, model_type: str | None = None) -> PruningBackend:
    if model_type is None:
        model_type = "resnet_cifar" if isinstance(model, ResNetCIFAR) else "mlp"
    if model_type == "mlp":
        return MlpBackend(model)
    if model_type in {"resnet_cifar", "cnn"}:
        return CnnBackend(model)
    raise ValueError(f"unsupported model.type for pruning backend: {model_type}")
