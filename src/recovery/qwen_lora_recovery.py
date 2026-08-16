"""LoRA (PEFT) recovery for pruned Qwen models on SQuAD chat packs.

Aligns with LLM-Pruner-style post-prune recovery: AdamW on low-rank adapters,
then merge_and_unload so the returned model stays a plain causal LM.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.recovery.qwen_lm_recovery import _batch_loss, _move_batch, evaluate_lm_loss
from src.utils.device import resolve_device

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _resolve_target_modules(
    model: nn.Module,
    target_modules: Optional[Sequence[str]] = None,
) -> List[str]:
    requested = list(target_modules) if target_modules else list(DEFAULT_LORA_TARGETS)
    present = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            present.add(name.rsplit(".", 1)[-1])
    resolved = [name for name in requested if name in present]
    if not resolved:
        raise ValueError(
            f"no LoRA target modules found; requested={requested} present_linear={sorted(present)}"
        )
    return resolved


class QwenLoraRecovery:
    """Fine-tune pruned causal LM with LoRA; select by lowest carved-val CE."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "peft is required for LoRA recovery; install with: pip install peft"
            ) from exc

        self.device = resolve_device(device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = _resolve_target_modules(model, target_modules)

        base = model.to(self.device)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base, lora_config)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA produced no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(self.model, batch)
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                continue
            loss.backward()
            self.optimizer.step()
            total += value
            batches += 1
        if batches == 0:
            return 0.0
        return total / batches

    def recover(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int = 2,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a non-negative integer")
        history: Dict[str, Any] = {
            "recovery_level": 1,
            "recovery_kind": "qwen_lora_ce",
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": list(self.target_modules),
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "best_epoch": None,
            "best_validation_loss": None,
            "best_validation_accuracy": None,
            "merged": False,
        }
        best_state = None
        if epochs == 0:
            metrics = evaluate_lm_loss(self.model, validation_loader, device=str(self.device))
            history["best_epoch"] = 0
            history["best_validation_loss"] = metrics["loss"]
            history["best_validation_accuracy"] = metrics["accuracy"]
        else:
            if verbose:
                print(
                    f"[LoRA LM Recovery] {epochs} epochs r={self.lora_r} "
                    f"targets={self.target_modules}"
                )
            for epoch in range(1, epochs + 1):
                train_loss = self.train_epoch(train_loader)
                metrics = evaluate_lm_loss(self.model, validation_loader, device=str(self.device))
                history["train_loss"].append(train_loss)
                history["validation_loss"].append(metrics["loss"])
                history["validation_accuracy"].append(metrics["accuracy"])
                if verbose:
                    print(
                        f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                        f"val_loss={metrics['loss']:.4f} proxy={metrics['accuracy']:.2f}"
                    )
                better = (
                    history["best_validation_loss"] is None
                    or metrics["loss"] < float(history["best_validation_loss"])
                )
                if better:
                    history["best_epoch"] = epoch
                    history["best_validation_loss"] = float(metrics["loss"])
                    history["best_validation_accuracy"] = float(metrics["accuracy"])
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
            if best_state is not None:
                self.model.load_state_dict(best_state)

        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            history["merged"] = True
        return history


def quick_lora_recovery(
    pruned_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
    weight_decay: float = 0.0,
    copy_model: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Level-1 LoRA recovery entrypoint; returns a merged plain LM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = copy.deepcopy(pruned_model) if copy_model else pruned_model
    recoverer = QwenLoraRecovery(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    history = recoverer.recover(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
        verbose=verbose,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recoverer.model, history


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "QwenLoraRecovery",
    "quick_lora_recovery",
]
