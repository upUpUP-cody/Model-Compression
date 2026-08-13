"""Fast, side-effect-free candidate evaluation."""
import time
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.device import resolve_device


@dataclass(frozen=True)
class CheapCriticResult:
    """JSON-safe metrics from a bounded inference evaluation."""

    loss: float
    accuracy: float
    samples: int
    parameter_count: int
    nonzero_parameter_count: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples": self.samples,
            "parameter_count": self.parameter_count,
            "nonzero_parameter_count": self.nonzero_parameter_count,
            "elapsed_seconds": self.elapsed_seconds,
        }


class CheapCritic:
    """Evaluate a candidate on an exact, bounded number of samples."""

    def __init__(self, criterion: nn.Module | None = None) -> None:
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        max_samples: int = 500,
        device: str | torch.device = "cpu",
    ) -> CheapCriticResult:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        device = resolve_device(device)
        non_blocking = device.type == "cuda"
        original_training = model.training
        total_loss = 0.0
        correct = 0
        samples = 0
        started_at = time.perf_counter()
        try:
            model.eval()
            with torch.inference_mode():
                for data, target in dataloader:
                    remaining = max_samples - samples
                    if remaining <= 0:
                        break
                    batch_size = min(remaining, target.size(0))
                    data = data[:batch_size].to(device, non_blocking=non_blocking)
                    target = target[:batch_size].to(device, non_blocking=non_blocking)
                    output = model(data)
                    loss = self.criterion(output, target)
                    total_loss += loss.item() * batch_size
                    correct += output.argmax(dim=1).eq(target).sum().item()
                    samples += batch_size
        finally:
            model.train(original_training)

        if samples == 0:
            raise ValueError("dataloader produced no samples")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        nonzero_parameter_count = sum(
            (parameter.detach() != 0).sum().item() for parameter in model.parameters()
        )
        return CheapCriticResult(
            loss=total_loss / samples,
            accuracy=100.0 * correct / samples,
            samples=samples,
            parameter_count=parameter_count,
            nonzero_parameter_count=nonzero_parameter_count,
            elapsed_seconds=time.perf_counter() - started_at,
        )
