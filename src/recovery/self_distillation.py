"""Level 3 self-distillation recovery."""
from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.device import resolve_device
from src.utils.precision import autocast_context, grad_scaler, resolve_precision


def self_distillation_recovery(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = "cpu",
    precision: str = "fp32",
    temperature: float = 2.0,
    alpha: float = 0.5,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict]:
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    recoverer = _SelfDistillRecoverer(
        student=copy.deepcopy(student_model),
        teacher=copy.deepcopy(teacher_model),
        device=device,
        learning_rate=learning_rate,
        precision=precision,
        temperature=temperature,
        alpha=alpha,
    )
    history = recoverer.recover(train_loader, validation_loader, epochs=epochs, verbose=verbose)
    return recoverer.student, history


class _SelfDistillRecoverer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str,
        learning_rate: float,
        precision: str,
        temperature: float,
        alpha: float,
    ) -> None:
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.non_blocking = self.device.type == "cuda"
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        self.scaler = grad_scaler(self.device, self.precision)

    def recover(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        verbose: bool,
    ) -> Dict:
        history = {
            "recovery_level": 3,
            "temperature": self.temperature,
            "alpha": self.alpha,
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
                print(f"[SelfDistill] epoch {epoch}/{epochs}: validation acc {accuracy:.2f}%")
            if history["best_validation_accuracy"] is None or accuracy > history["best_validation_accuracy"]:
                history["best_validation_accuracy"] = float(accuracy)
                history["best_epoch"] = epoch
                best_state = {
                    name: tensor.detach().cpu().clone() for name, tensor in self.student.state_dict().items()
                }
        if best_state is not None:
            self.student.load_state_dict(best_state)
        return history

    def _distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = self.ce(student_logits, target)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)
        return self.alpha * soft + (1.0 - self.alpha) * hard

    def _train_epoch(self, train_loader: DataLoader) -> None:
        self.student.train()
        for data, target in train_loader:
            data = data.to(self.device, non_blocking=self.non_blocking)
            target = target.to(self.device, non_blocking=self.non_blocking)
            self.optimizer.zero_grad()
            with autocast_context(self.device, self.precision):
                student_logits = self.student(data)
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                loss = self._distill_loss(student_logits, teacher_logits, target)
            if self.scaler is None:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    @torch.no_grad()
    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.student.eval()
        correct = 0
        total = 0
        for data, target in validation_loader:
            data = data.to(self.device, non_blocking=self.non_blocking)
            target = target.to(self.device, non_blocking=self.non_blocking)
            with autocast_context(self.device, self.precision):
                output = self.student(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        if total == 0:
            raise ValueError("validation dataloader produced no samples")
        return 100.0 * correct / total
