"""Level 2 LoRA recovery for Conv2d and Linear layers."""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.device import resolve_device
from src.utils.precision import autocast_context, grad_scaler, resolve_precision


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + delta


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_channels, 1, 1))
        self.lora_b = nn.Parameter(torch.zeros(base.out_channels, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.conv2d(F.conv2d(x, self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + delta


def _wrap_lora_modules(model: nn.Module, rank: int, alpha: float) -> List[str]:
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
        and not isinstance(module, (LoRALinear, LoRAConv2d))
    ]
    wrapped: List[str] = []
    for name in sorted(targets, key=len, reverse=True):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        child = getattr(parent, attr)
        if isinstance(child, nn.Linear):
            setattr(parent, attr, LoRALinear(child, rank, alpha))
            wrapped.append(name)
        elif isinstance(child, nn.Conv2d):
            setattr(parent, attr, LoRAConv2d(child, rank, alpha))
            wrapped.append(name)
    return wrapped


def lora_recovery(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = "cpu",
    precision: str = "fp32",
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict]:
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    recoverer = _LoRARecoverer(
        model=copy.deepcopy(model),
        device=device,
        learning_rate=learning_rate,
        precision=precision,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    history = recoverer.recover(train_loader, validation_loader, epochs=epochs, verbose=verbose)
    return recoverer.model, history


class _LoRARecoverer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float,
        precision: str,
        lora_rank: int,
        lora_alpha: float,
    ) -> None:
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.non_blocking = self.device.type == "cuda"
        self.model = model.to(self.device)
        self.wrapped_layers = _wrap_lora_modules(self.model, lora_rank, lora_alpha)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("LoRA recovery found no trainable parameters")
        self.optimizer = torch.optim.Adam(trainable, lr=learning_rate)
        self.scaler = grad_scaler(self.device, self.precision)
        self.criterion = nn.CrossEntropyLoss()

    def recover(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        verbose: bool,
    ) -> Dict:
        history = {
            "recovery_level": 2,
            "wrapped_layers": list(self.wrapped_layers),
            "validation_accuracy": [],
            "best_validation_accuracy": None,
            "best_epoch": None,
        }
        best_state = None
        for epoch in range(1, epochs + 1):
            self._train_epoch(train_loader)
            accuracy = self._evaluate(validation_loader)
            history["validation_accuracy"].append(float(accuracy))
            if verbose:
                print(f"[LoRA] epoch {epoch}/{epochs}: validation acc {accuracy:.2f}%")
            if history["best_validation_accuracy"] is None or accuracy > history["best_validation_accuracy"]:
                history["best_validation_accuracy"] = float(accuracy)
                history["best_epoch"] = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def _train_epoch(self, train_loader: DataLoader) -> None:
        self.model.train()
        for data, target in train_loader:
            data = data.to(self.device, non_blocking=self.non_blocking)
            target = target.to(self.device, non_blocking=self.non_blocking)
            self.optimizer.zero_grad()
            with autocast_context(self.device, self.precision):
                output = self.model(data)
                loss = self.criterion(output, target)
            if self.scaler is None:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    @torch.no_grad()
    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for data, target in validation_loader:
            data = data.to(self.device, non_blocking=self.non_blocking)
            target = target.to(self.device, non_blocking=self.non_blocking)
            with autocast_context(self.device, self.precision):
                output = self.model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        if total == 0:
            raise ValueError("validation dataloader produced no samples")
        return 100.0 * correct / total
