"""CNN layer dependency graph for structured channel pruning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from src.models.resnet_cifar import BasicBlock, ResNetCIFAR


@dataclass(frozen=True)
class ConvConsumer:
    """Downstream modules that must follow conv output channel pruning."""

    conv_name: str
    batch_norm_name: Optional[str]
    input_conv_names: Sequence[str]


def _block_conv1_name(layer_idx: int, block_idx: int) -> str:
    return f"layer{layer_idx}.{block_idx}.conv1"


def _block_bn1_name(layer_idx: int, block_idx: int) -> str:
    return f"layer{layer_idx}.{block_idx}.bn1"


def _block_conv2_name(layer_idx: int, block_idx: int) -> str:
    return f"layer{layer_idx}.{block_idx}.conv2"


def resnet_cifar_prunable_conv_names(model: nn.Module) -> List[str]:
    """Return conv1 layer names inside each BasicBlock (intermediate width pruning)."""
    if not isinstance(model, ResNetCIFAR):
        raise ValueError("model must be ResNetCIFAR")
    names: List[str] = []
    for layer_idx in range(1, 5):
        layer = getattr(model, f"layer{layer_idx}")
        for block_idx in range(len(layer)):
            block = layer[block_idx]
            if not isinstance(block, BasicBlock):
                raise ValueError(f"Unexpected block type at layer{layer_idx}.{block_idx}")
            names.append(_block_conv1_name(layer_idx, block_idx))
    return names


def build_resnet_cifar_dependency_map(model: nn.Module) -> Dict[str, ConvConsumer]:
    """Map each prunable conv1 to its synchronized consumers."""
    prunable = resnet_cifar_prunable_conv_names(model)
    dependency_map: Dict[str, ConvConsumer] = {}
    for conv_name in prunable:
        parts = conv_name.split(".")
        layer_idx, block_idx = int(parts[0].replace("layer", "")), int(parts[1])
        dependency_map[conv_name] = ConvConsumer(
            conv_name=conv_name,
            batch_norm_name=_block_bn1_name(layer_idx, block_idx),
            input_conv_names=(_block_conv2_name(layer_idx, block_idx),),
        )
    return dependency_map


def validate_resnet_cifar_structure(model: nn.Module) -> None:
    """Ensure expected ResNet BasicBlock wiring before pruning."""
    dependency_map = build_resnet_cifar_dependency_map(model)
    for conv_name, consumer in dependency_map.items():
        conv = _get_module(model, conv_name)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"{conv_name} is not Conv2d")
        if consumer.batch_norm_name:
            bn = _get_module(model, consumer.batch_norm_name)
            if not isinstance(bn, nn.BatchNorm2d):
                raise ValueError(f"{consumer.batch_norm_name} is not BatchNorm2d")
            if bn.num_features != conv.out_channels:
                raise ValueError(f"BN width mismatch for {conv_name}")
        for downstream in consumer.input_conv_names:
            downstream_conv = _get_module(model, downstream)
            if not isinstance(downstream_conv, nn.Conv2d):
                raise ValueError(f"{downstream} is not Conv2d")
            if downstream_conv.in_channels != conv.out_channels:
                raise ValueError(f"Conv input mismatch: {conv_name} -> {downstream}")


def _get_module(model: nn.Module, layer_name: str) -> nn.Module:
    module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module
