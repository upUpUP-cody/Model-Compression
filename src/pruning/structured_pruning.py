"""
Structured pruning utilities for MLP-style models.
"""
import copy
import math
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class StructuredPruning:
    """Physically prune hidden Linear layers and their connected modules.

    Shape-changing pruning replaces affected modules. Create an optimizer only
    after pruning, because optimizers retain references to the old parameters.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.pruning_masks: Dict[str, Dict] = {}

    def prune_mlp_neurons(self, layer_name: str, neuron_indices: List[int]) -> None:
        """Prune a hidden Linear layer while preserving connected dimensions."""
        self.prune_linear_block(layer_name, neuron_indices)

    def prune_mlp_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Return deterministic keep indices for a valid hidden Linear layer."""
        layer = self._validate_hidden_linear(layer_name)
        self._validate_prune_ratio(prune_ratio)

        if importance_scores is None:
            importance_scores = torch.norm(layer.weight.detach(), p=2, dim=1)
        scores = self._validate_importance_scores(layer, importance_scores)

        num_keep = int(layer.out_features * (1 - prune_ratio))
        if num_keep == layer.out_features:
            return list(range(layer.out_features))

        indexed_scores = [(-float(scores[index]), index) for index in range(layer.out_features)]
        indexed_scores.sort()
        return sorted(index for _, index in indexed_scores[:num_keep])

    def prune_linear_block(
        self,
        linear_layer_name: str,
        keep_indices: List[int],
        next_linear_layer_name: Optional[str] = None,
    ) -> None:
        """Prune a hidden Linear layer, its following BatchNorm, and next Linear.

        If ``next_linear_layer_name`` is supplied for compatibility, it must
        match the actual downstream Linear layer.
        """
        linear_layer = self._validate_hidden_linear(linear_layer_name)
        normalized_indices = self._validate_keep_indices(linear_layer, keep_indices)
        discovered_next_name = self._find_next_layer(linear_layer_name)
        if discovered_next_name is None:
            raise ValueError(f"Layer {linear_layer_name} has no downstream Linear layer")
        if (
            next_linear_layer_name is not None
            and next_linear_layer_name != discovered_next_name
        ):
            raise ValueError(
                f"Expected downstream layer {discovered_next_name}, got {next_linear_layer_name}"
            )

        next_linear = self._get_layer_by_name(discovered_next_name)
        if not isinstance(next_linear, nn.Linear):
            raise ValueError(f"Layer {discovered_next_name} is not a Linear layer")

        batch_norm_name = self._find_following_batch_norm(linear_layer_name)
        batch_norm = (
            self._get_layer_by_name(batch_norm_name)
            if batch_norm_name is not None
            else None
        )
        if batch_norm is not None and not isinstance(batch_norm, nn.BatchNorm1d):
            raise ValueError(f"Layer {batch_norm_name} is not a BatchNorm1d layer")

        index_tensor = torch.tensor(
            normalized_indices, device=linear_layer.weight.device, dtype=torch.long
        )
        pruned_linear = self._make_pruned_linear_output(linear_layer, index_tensor)
        pruned_next = self._make_pruned_linear_input(next_linear, index_tensor)
        pruned_batch_norm = (
            self._make_pruned_batch_norm(batch_norm, index_tensor)
            if batch_norm is not None
            else None
        )

        self._replace_layer(linear_layer_name, pruned_linear)
        if pruned_batch_norm is not None and batch_norm_name is not None:
            self._replace_layer(batch_norm_name, pruned_batch_norm)
        self._replace_layer(discovered_next_name, pruned_next)

        self.pruning_masks[linear_layer_name] = {
            "type": "neuron",
            "indices": normalized_indices,
            "original_size": linear_layer.out_features,
            "retained_size": len(normalized_indices),
        }

    def prune_connected_layers(
        self, current_layer: str, next_layer: str, keep_indices: List[int]
    ) -> None:
        """Backward-compatible alias for connected hidden-layer pruning."""
        self.prune_linear_block(current_layer, keep_indices, next_layer)

    def prune_uniform(self, prune_ratio: float) -> int:
        """Apply one pruning ratio to every hidden Linear layer in reverse order."""
        self._validate_prune_ratio(prune_ratio)
        hidden_layers = self._hidden_linear_layer_names()
        for layer_name in reversed(hidden_layers):
            keep_indices = self.prune_mlp_by_ratio(layer_name, prune_ratio)
            self.prune_linear_block(layer_name, keep_indices)
        return self.get_structural_parameter_count()

    def prune_by_layer(self, layer_ratios: Dict[str, float]) -> int:
        """Apply configured ratios to hidden Linear layers without partial validation."""
        if not isinstance(layer_ratios, dict):
            raise ValueError("layer_ratios must be a dictionary")

        for layer_name, prune_ratio in layer_ratios.items():
            self._validate_hidden_linear(layer_name)
            self._validate_prune_ratio(prune_ratio)

        ordered_layers = [
            layer_name
            for layer_name in self._hidden_linear_layer_names()
            if layer_name in layer_ratios
        ]
        for layer_name in reversed(ordered_layers):
            keep_indices = self.prune_mlp_by_ratio(layer_name, layer_ratios[layer_name])
            self.prune_linear_block(layer_name, keep_indices)
        return self.get_structural_parameter_count()

    def create_pruned_model(self, pruning_config: Dict[str, float]) -> nn.Module:
        """Create an independently pruned copy of the source model."""
        pruned_model = copy.deepcopy(self.model)
        StructuredPruning(pruned_model).prune_by_layer(pruning_config)
        return pruned_model

    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("layer_name must be a non-empty string")

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
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            last_part = parts[-1]
            if last_part.isdigit():
                parent[int(last_part)] = replacement
            else:
                setattr(parent, last_part, replacement)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError(f"Unable to replace layer: {layer_name}") from error

    def _linear_layer_names(self) -> List[str]:
        return [
            name for name, module in self.model.named_modules() if isinstance(module, nn.Linear)
        ]

    def _hidden_linear_layer_names(self) -> List[str]:
        linear_layers = self._linear_layer_names()
        return linear_layers[:-1]

    def _find_next_layer(self, layer_name: str) -> Optional[str]:
        linear_layers = self._linear_layer_names()
        try:
            index = linear_layers.index(layer_name)
        except ValueError:
            return None
        return linear_layers[index + 1] if index + 1 < len(linear_layers) else None

    def _find_following_batch_norm(self, layer_name: str) -> Optional[str]:
        named_modules = list(self.model.named_modules())
        names = [name for name, _ in named_modules]
        try:
            start_index = names.index(layer_name)
        except ValueError:
            return None

        for name, module in named_modules[start_index + 1 :]:
            if isinstance(module, nn.Linear):
                return None
            if isinstance(module, nn.BatchNorm1d):
                return name
        return None

    def _validate_hidden_linear(self, layer_name: str) -> nn.Linear:
        layer = self._get_layer_by_name(layer_name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer")
        if self._find_next_layer(layer_name) is None:
            raise ValueError(f"Layer {layer_name} is the final classifier and cannot be pruned")
        return layer

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
    def _validate_keep_indices(layer: nn.Linear, keep_indices: Sequence[int]) -> List[int]:
        if isinstance(keep_indices, (str, bytes)) or not isinstance(keep_indices, Sequence):
            raise ValueError("keep_indices must be a non-empty sequence of integers")
        if not keep_indices:
            raise ValueError("keep_indices must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in keep_indices):
            raise ValueError("keep_indices must contain only integers")

        normalized = sorted(int(index) for index in keep_indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError("keep_indices must be unique")
        if normalized[0] < 0 or normalized[-1] >= layer.out_features:
            raise ValueError("keep_indices contains an out-of-range index")
        return normalized

    @staticmethod
    def _validate_importance_scores(
        layer: nn.Linear, importance_scores: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(importance_scores, torch.Tensor):
            raise ValueError("importance_scores must be a torch.Tensor")
        scores = importance_scores.detach().reshape(-1)
        if importance_scores.ndim != 1 or scores.numel() != layer.out_features:
            raise ValueError("importance_scores must be one-dimensional and match out_features")
        if not torch.isfinite(scores).all().item():
            raise ValueError("importance_scores must contain only finite values")
        return scores.cpu()

    @staticmethod
    def _make_pruned_linear_output(layer: nn.Linear, index_tensor: torch.Tensor) -> nn.Linear:
        replacement = nn.Linear(
            layer.in_features,
            index_tensor.numel(),
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        replacement.train(layer.training)
        with torch.no_grad():
            replacement.weight.copy_(layer.weight.index_select(0, index_tensor))
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias.index_select(0, index_tensor))
        replacement.weight.requires_grad_(layer.weight.requires_grad)
        if layer.bias is not None:
            replacement.bias.requires_grad_(layer.bias.requires_grad)
        return replacement

    @staticmethod
    def _make_pruned_linear_input(layer: nn.Linear, index_tensor: torch.Tensor) -> nn.Linear:
        replacement = nn.Linear(
            index_tensor.numel(),
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        replacement.train(layer.training)
        with torch.no_grad():
            replacement.weight.copy_(layer.weight.index_select(1, index_tensor))
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias)
        replacement.weight.requires_grad_(layer.weight.requires_grad)
        if layer.bias is not None:
            replacement.bias.requires_grad_(layer.bias.requires_grad)
        return replacement

    @staticmethod
    def _make_pruned_batch_norm(
        batch_norm: nn.BatchNorm1d, index_tensor: torch.Tensor
    ) -> nn.BatchNorm1d:
        replacement = nn.BatchNorm1d(
            index_tensor.numel(),
            eps=batch_norm.eps,
            momentum=batch_norm.momentum,
            affine=batch_norm.affine,
            track_running_stats=batch_norm.track_running_stats,
            device=batch_norm.weight.device if batch_norm.affine else index_tensor.device,
            dtype=batch_norm.weight.dtype if batch_norm.affine else None,
        )
        replacement.train(batch_norm.training)
        with torch.no_grad():
            if batch_norm.affine:
                replacement.weight.copy_(batch_norm.weight.index_select(0, index_tensor))
                replacement.bias.copy_(batch_norm.bias.index_select(0, index_tensor))
                replacement.weight.requires_grad_(batch_norm.weight.requires_grad)
                replacement.bias.requires_grad_(batch_norm.bias.requires_grad)
            if batch_norm.track_running_stats:
                replacement.running_mean.copy_(
                    batch_norm.running_mean.index_select(0, index_tensor)
                )
                replacement.running_var.copy_(
                    batch_norm.running_var.index_select(0, index_tensor)
                )
                replacement.num_batches_tracked.copy_(batch_norm.num_batches_tracked)
        return replacement

    def get_sparsity(self) -> float:
        """Return numerical zero density, not physical structural compression."""
        total_params = self.get_structural_parameter_count()
        if total_params == 0:
            return 0.0
        zero_params = sum((parameter.detach() == 0).sum().item() for parameter in self.model.parameters())
        return zero_params / total_params

    def get_structural_parameter_count(self) -> int:
        """Return the number of physically retained parameters."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    def get_model_info(self) -> Dict:
        total_params = self.get_structural_parameter_count()
        trainable_params = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            "total_params": total_params,
            "structural_parameter_count": total_params,
            "trainable_params": trainable_params,
            "sparsity": self.get_sparsity(),
            "model_size_mb": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ) / (1024 ** 2),
        }


def apply_pruning_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Apply unstructured pruning masks to named parameters."""
    for name, parameter in model.named_parameters():
        if name in masks:
            with torch.no_grad():
                parameter.mul_(masks[name])


def compute_layer_importance(layer: nn.Linear, method: str = "l2") -> torch.Tensor:
    """Compute per-output-neuron importance for a Linear layer."""
    if method == "l2":
        return torch.norm(layer.weight.detach(), p=2, dim=1)
    if method == "l1":
        return torch.norm(layer.weight.detach(), p=1, dim=1)
    if method == "variance":
        return torch.var(layer.weight.detach(), dim=1)
    raise ValueError(f"Unknown method: {method}")


StructuredPruner = StructuredPruning
