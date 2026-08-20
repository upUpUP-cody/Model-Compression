"""Level-1 causal-LM recovery for pruned Qwen models (SQuAD packs)."""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.cheap_critic import CheapCriticResult
from src.utils.device import resolve_device


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def _batch_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    loss = outputs.loss
    if loss is None:
        raise ValueError("model did not return LM loss")
    # Aggregate in float32 for stable CE under fp16 weights.
    return loss.float()


@torch.no_grad()
def evaluate_lm_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Mean token CE on LM packs; maps to search-friendly accuracy proxy."""
    import math

    resolved = resolve_device(device)
    original_training = model.training
    model.eval()
    total = 0.0
    batches = 0
    try:
        for batch in dataloader:
            if max_batches is not None and batches >= int(max_batches):
                break
            batch = _move_batch(batch, resolved)
            loss = _batch_loss(model, batch)
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                continue
            total += value
            batches += 1
    finally:
        model.train(original_training)
    if batches == 0:
        # Fall back so search/gates still get a comparable number.
        return {"loss": 1e6, "accuracy": 0.0, "samples": 0.0}
    mean_loss = total / batches
    accuracy_proxy = 100.0 / (1.0 + mean_loss)
    return {"loss": mean_loss, "accuracy": accuracy_proxy, "samples": float(batches)}


class QwenLmCheapCritic:
    """CheapCritic-compatible ranking via LM token CE (no classification)."""

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        max_samples: int = 500,
        device: str | torch.device = "cpu",
    ) -> CheapCriticResult:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        # Treat max_samples as an upper bound on batches for LM packs.
        max_batches = max(1, int(max_samples))
        started = time.perf_counter()
        metrics = evaluate_lm_loss(model, dataloader, device=str(device), max_batches=max_batches)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        nonzero = sum((parameter.detach() != 0).sum().item() for parameter in model.parameters())
        return CheapCriticResult(
            loss=float(metrics["loss"]),
            accuracy=float(metrics["accuracy"]),
            samples=int(metrics["samples"]),
            parameter_count=int(parameter_count),
            nonzero_parameter_count=int(nonzero),
            elapsed_seconds=time.perf_counter() - started,
        )


class QwenLmRecovery:
    """Fine-tune a pruned causal LM; select by lowest carved-val CE."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        precision: str = "fp16",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.precision = str(precision)
        # SGD avoids Adam moment buffers (~2x params) which OOM 1.5B smoke recovery.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        import math

        self.model.train()
        total = 0.0
        batches = 0
        # Prefer stable CE over amp for 1.5B smoke recovery.
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
        epochs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a non-negative integer")
        history: Dict[str, Any] = {
            "recovery_level": 1,
            "recovery_kind": "qwen_lm_ce",
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "best_epoch": None,
            "best_validation_loss": None,
            "best_validation_accuracy": None,
        }
        best_state = None
        if epochs == 0:
            metrics = evaluate_lm_loss(self.model, validation_loader, device=str(self.device))
            history["best_epoch"] = 0
            history["best_validation_loss"] = metrics["loss"]
            history["best_validation_accuracy"] = metrics["accuracy"]
            return history
        if verbose:
            print(f"[Level 1 LM Recovery] {epochs} epochs")
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
        return history


def quick_lm_recovery(
    pruned_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    device: str = "cpu",
    precision: str = "fp16",
    verbose: bool = True,
    weight_decay: float = 0.0,
    copy_model: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """AutonomousSearch-compatible Level-1 recovery entrypoint."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = copy.deepcopy(pruned_model) if copy_model else pruned_model
    recoverer = QwenLmRecovery(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        precision=precision,
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


def run_configured_recovery(
    pruned_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    *,
    epochs: Optional[int] = None,
    copy_model: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Dispatch SGD vs LoRA recovery from config['recovery']."""
    recovery = dict(config.get("recovery") or {})
    backend = str(recovery.get("backend", "sgd")).lower().strip()
    device = str(config.get("hardware", {}).get("device", "cpu"))
    precision = str(config.get("hardware", {}).get("precision", "fp16"))
    resolved_epochs = int(epochs if epochs is not None else recovery.get("epochs", 1))
    weight_decay = float(recovery.get("weight_decay", 0.0))

    if backend in ("lora", "peft"):
        from src.recovery.qwen_lora_recovery import quick_lora_recovery

        learning_rate = float(recovery.get("learning_rate", 1e-4))
        targets = recovery.get("lora_target_modules")
        max_steps = recovery.get("max_steps")
        return quick_lora_recovery(
            pruned_model,
            train_loader,
            validation_loader,
            epochs=resolved_epochs,
            learning_rate=learning_rate,
            device=device,
            verbose=verbose,
            weight_decay=weight_decay,
            copy_model=copy_model,
            lora_r=int(recovery.get("lora_r", 8)),
            lora_alpha=int(recovery.get("lora_alpha", 16)),
            lora_dropout=float(recovery.get("lora_dropout", 0.05)),
            target_modules=list(targets) if targets else None,
            max_steps=int(max_steps) if max_steps is not None else None,
        )

    learning_rate = float(recovery.get("learning_rate", 2e-5))
    return quick_lm_recovery(
        pruned_model,
        train_loader,
        validation_loader,
        epochs=resolved_epochs,
        learning_rate=learning_rate,
        device=device,
        precision=precision,
        verbose=verbose,
        weight_decay=weight_decay,
        copy_model=copy_model,
    )
