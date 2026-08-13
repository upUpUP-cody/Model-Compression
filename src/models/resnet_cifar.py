"""ResNet-18 for CIFAR-10 (32x32 input)."""
from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """ResNet-18 variant for 32x32 inputs."""

    def __init__(
        self,
        block: Type[BasicBlock] = BasicBlock,
        layers: Optional[List[int]] = None,
        num_classes: int = 10,
        base_width: int = 64,
    ):
        super().__init__()
        layers = layers or [2, 2, 2, 2]
        self.in_planes = base_width
        self.num_classes = num_classes
        self.base_width = base_width
        self.layer_config = list(layers)
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        modules = []
        for block_stride in strides:
            modules.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def resnet18_cifar(num_classes: int = 10, base_width: int = 64) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=base_width)


def resnet18_cifar_small(num_classes: int = 10) -> ResNetCIFAR:
    """Smaller width for dense_small comparison arm."""
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=32)
