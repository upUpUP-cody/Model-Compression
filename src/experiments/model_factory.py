"""Build baseline models from YAML model configuration."""
from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from src.models.dense_baseline import MLP
from src.models.resnet_cifar import resnet18_cifar, resnet18_cifar_small


def model_type_from_config(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model", {})
    return str(model_cfg.get("type", "mlp"))


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = dict(config["model"])
    model_type = str(model_cfg.pop("type", "mlp"))
    if model_type == "mlp":
        return MLP(**model_cfg)
    if model_type == "resnet_cifar":
        base_width = int(model_cfg.pop("base_width", 64))
        num_classes = int(model_cfg.pop("num_classes", 10))
        if model_cfg:
            raise ValueError(f"unknown resnet_cifar model keys: {sorted(model_cfg)}")
        return resnet18_cifar(num_classes=num_classes, base_width=base_width)
    raise ValueError(f"unsupported model.type: {model_type}")


def build_dense_small_model(config: Mapping[str, Any], device: str) -> nn.Module:
    comparison = config.get("comparison", {})
    model_type = model_type_from_config(config)
    if model_type == "resnet_cifar":
        variant = str(comparison.get("dense_small_variant", "small"))
        if variant == "small":
            model = resnet18_cifar_small(num_classes=int(config["model"].get("num_classes", 10)))
        else:
            base_width = int(comparison.get("dense_small_base_width", 32))
            model = resnet18_cifar(
                num_classes=int(config["model"].get("num_classes", 10)),
                base_width=base_width,
            )
        return model.to(device)
    hidden_dims = comparison.get("dense_small_hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise ValueError("comparison.dense_small_hidden_dims must be a non-empty list for MLP")
    model_cfg = dict(config["model"])
    model_cfg.pop("type", None)
    model_cfg["hidden_dims"] = [int(width) for width in hidden_dims]
    return MLP(**model_cfg).to(device)
