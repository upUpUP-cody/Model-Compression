"""Dispatch recovery by configured level."""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from src.recovery.lora_recovery import lora_recovery
from src.recovery.reconstruction import quick_recovery
from src.recovery.self_distillation import self_distillation_recovery


def run_recovery(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    *,
    teacher_model: Optional[nn.Module] = None,
    verbose: bool = False,
) -> tuple[nn.Module, Dict[str, Any]]:
    recovery = config.get("recovery", {})
    level = int(recovery.get("level", 1))
    epochs = int(recovery.get("epochs", 0))
    learning_rate = float(recovery.get("learning_rate", 0.001))
    device = str(config["hardware"]["device"])
    precision = str(config["hardware"].get("precision", "fp32"))

    if level <= 1:
        return quick_recovery(
            model,
            train_loader,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            precision=precision,
            verbose=verbose,
        )
    if level == 2:
        return lora_recovery(
            model,
            train_loader,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            precision=precision,
            lora_rank=int(recovery.get("lora_rank", 4)),
            lora_alpha=float(recovery.get("lora_alpha", 8.0)),
            verbose=verbose,
        )
    if level == 3:
        if teacher_model is None:
            raise ValueError("self-distillation recovery requires teacher_model")
        return self_distillation_recovery(
            student_model=model,
            teacher_model=copy.deepcopy(teacher_model),
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            precision=precision,
            temperature=float(recovery.get("distill_temperature", 2.0)),
            alpha=float(recovery.get("distill_alpha", 0.5)),
            verbose=verbose,
        )
    raise ValueError(f"unsupported recovery.level: {level}")
