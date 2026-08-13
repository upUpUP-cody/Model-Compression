"""Structured channel pruning for CNN / ResNet models."""
from __future__ import annotations

import copy
import math
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from src.pruning.cnn_dependency import (
    build_resnet_cifar_dependency_map,
    resnet_cifar_prunable_conv_names,
    validate_resnet_cifar_structure,
)


class CnnStructuredPruning:
    """Prune intermediate conv1 channels in ResNetCIFAR BasicBlocks."""

    def __init__(self, model: nn.Module):
        self.model = model
        validate_resnet_cifar_structure(model)
        self.dependency_map = build_resnet_cifar_dependency_map(model)
        self.pruning_masks: Dict[str, dict] = {}

    def prunable_layer_names(self) -> List[str]:
        return resnet_cifar_prunable_conv_names(self.model)

    def prune_conv_block(self, conv_name: str, keep_indices: Sequence[int]) -> None:
        if conv_name not in self.dependency_map:
            raise ValueError(f"Unknown prunable conv layer: {conv_name}")
        conv = self._get_layer(conv_name)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"{conv_name} is not Conv2d")
        normalized = self._validate_keep_indices(conv, keep_indices)
        index_tensor = torch.tensor(normalized, device=conv.weight.device, dtype=torch.long)

        consumer = self.dependency_map[conv_name]
        pruned_conv = self._make_pruned_conv_output(conv, index_tensor)
        self._replace_layer(conv_name, pruned_conv)

        if consumer.batch_norm_name:
            bn = self._get_layer(consumer.batch_norm_name)
            if isinstance(bn, nn.BatchNorm2d):
                pruned_bn = self._make_pruned_batch_norm2d(bn, index_tensor)
                self._replace_layer(consumer.batch_norm_name, pruned_bn)

        for downstream_name in consumer.input_conv_names:
            downstream = self._get_layer(downstream_name)
            if not isinstance(downstream, nn.Conv2d):
                raise ValueError(f"{downstream_name} is not Conv2d")
            pruned_downstream = self._make_pruned_conv_input(downstream, index_tensor)
            self._replace_layer(downstream_name, pruned_downstream)

        self.pruning_masks[conv_name] = {
            "type": "channel",
            "indices": normalized,
            "original_size": conv.out_channels,
            "retained_size": len(normalized),
        }

    def prune_by_ratio(self, conv_name: str, prune_ratio: float) -> List[int]:
        conv = self._get_layer(conv_name)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"{conv_name} is not Conv2d")
        self._validate_prune_ratio(prune_ratio)
        importance = torch.norm(conv.weight.detach(), p=2, dim=(1, 2, 3))
        keep_count = max(1, int(round(conv.out_channels * (1.0 - float(prune_ratio)))))
        keep_indices = torch.topk(importance, keep_count).indices.tolist()
        self.prune_conv_block(conv_name, keep_indices)
        return sorted(int(index) for index in keep_indices)

    def prune_uniform(self, prune_ratio: float) -> int:
        self._validate_prune_ratio(prune_ratio)
        for conv_name in reversed(self.prunable_layer_names()):
            self.prune_by_ratio(conv_name, prune_ratio)
        return self.get_structural_parameter_count()

    def prune_by_layer(self, layer_ratios: Dict[str, float]) -> int:
        for conv_name, ratio in layer_ratios.items():
            if conv_name not in self.dependency_map:
                raise ValueError(f"Unknown layer: {conv_name}")
            self._validate_prune_ratio(ratio)
        ordered = [name for name in self.prunable_layer_names() if name in layer_ratios]
        for conv_name in reversed(ordered):
            self.prune_by_ratio(conv_name, layer_ratios[conv_name])
        return self.get_structural_parameter_count()

    def prune_by_layer_indices(self, layer_keep_indices: Dict[str, Sequence[int]]) -> int:
        ordered = [name for name in self.prunable_layer_names() if name in layer_keep_indices]
        for conv_name in reversed(ordered):
            self.prune_conv_block(conv_name, layer_keep_indices[conv_name])
        return self.get_structural_parameter_count()

    def create_pruned_model_by_indices(
        self, layer_keep_indices: Dict[str, Sequence[int]]
    ) -> nn.Module:
        pruned_model = copy.deepcopy(self.model)
        CnnStructuredPruning(pruned_model).prune_by_layer_indices(layer_keep_indices)
        return pruned_model

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module:
        pruned_model = copy.deepcopy(self.model)
        CnnStructuredPruning(pruned_model).prune_by_layer(pruning_config)
        return pruned_model

    def get_structural_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def _get_layer(self, layer_name: str) -> nn.Module:
        module = self.model
        try:
            for part in layer_name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError(f"Unknown layer: {layer_name}") from error
        return module

    def _replace_layer(self, layer_name: str, replacement: nn.Module) -> None:
        parts = layer_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = replacement
        else:
            setattr(parent, last, replacement)

    @staticmethod
    def _validate_prune_ratio(prune_ratio: float) -> None:
        if (
            isinstance(prune_ratio, bool)
            or not isinstance(prune_ratio, Real)
            or not math.isfinite(float(prune_ratio))
            or not 0.0 <= float(prune_ratio) < 1.0
        ):
            raise ValueError("prune_ratio must be finite and in [0.0, 1.0)")

    @staticmethod
    def _validate_keep_indices(layer: nn.Conv2d, keep_indices: Sequence[int]) -> List[int]:
        if isinstance(keep_indices, (str, bytes)) or not isinstance(keep_indices, Sequence):
            raise ValueError("keep_indices must be a non-empty sequence of integers")
        if not keep_indices:
            raise ValueError("keep_indices must not be empty")
        normalized = sorted(int(index) for index in keep_indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError("keep_indices must be unique")
        if normalized[0] < 0 or normalized[-1] >= layer.out_channels:
            raise ValueError("keep_indices contains an out-of-range index")
        return normalized

    @staticmethod
    def _make_pruned_conv_output(layer: nn.Conv2d, index_tensor: torch.Tensor) -> nn.Conv2d:
        replacement = nn.Conv2d(
            layer.in_channels,
            index_tensor.numel(),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        replacement.train(layer.training)
        with torch.no_grad():
            replacement.weight.copy_(layer.weight.index_select(0, index_tensor))
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias.index_select(0, index_tensor))
        return replacement

    @staticmethod
    def _make_pruned_conv_input(layer: nn.Conv2d, index_tensor: torch.Tensor) -> nn.Conv2d:
        replacement = nn.Conv2d(
            index_tensor.numel(),
            layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        replacement.train(layer.training)
        with torch.no_grad():
            replacement.weight.copy_(layer.weight.index_select(1, index_tensor))
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias)
        return replacement

    @staticmethod
    def _make_pruned_batch_norm2d(bn: nn.BatchNorm2d, index_tensor: torch.Tensor) -> nn.BatchNorm2d:
        replacement = nn.BatchNorm2d(
            index_tensor.numel(),
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
            device=bn.weight.device if bn.affine else index_tensor.device,
            dtype=bn.weight.dtype if bn.affine else None,
        )
        replacement.train(bn.training)
        with torch.no_grad():
            if bn.affine:
                replacement.weight.copy_(bn.weight.index_select(0, index_tensor))
                replacement.bias.copy_(bn.bias.index_select(0, index_tensor))
            if bn.track_running_stats:
                replacement.running_mean.copy_(bn.running_mean.index_select(0, index_tensor))
                replacement.running_var.copy_(bn.running_var.index_select(0, index_tensor))
                replacement.num_batches_tracked.copy_(bn.num_batches_tracked)
        return replacement
