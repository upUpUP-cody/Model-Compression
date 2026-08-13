"""Validation-based Level 1 reconstruction recovery."""
import copy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ReconstructionRecovery:
    """Fine-tune a pruned model and restore its best validation state."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0
        for data, target in train_loader:
            batches += 1
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        if batches == 0 or total == 0:
            raise ValueError("train dataloader produced no samples")
        return total_loss / batches, 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self, validation_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0
        for data, target in validation_loader:
            batches += 1
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += self.criterion(output, target).item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        if batches == 0 or total == 0:
            raise ValueError("validation dataloader produced no samples")
        return total_loss / batches, 100.0 * correct / total

    def recover(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """Recover using train data and select weights by validation accuracy."""
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a non-negative integer")
        history = {
            "train_loss": [],
            "train_acc": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "best_epoch": None,
            "best_validation_loss": None,
            "best_validation_accuracy": None,
        }
        best_state = None
        if verbose:
            print(f"[Level 1 Recovery] {epochs} epochs")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            validation_loss, validation_accuracy = self.evaluate(validation_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            if verbose:
                print(
                    f"  Epoch {epoch}/{epochs}: Train Acc: {train_acc:.2f}%, "
                    f"Validation Acc: {validation_accuracy:.2f}%"
                )
            better = (
                history["best_validation_accuracy"] is None
                or validation_accuracy > history["best_validation_accuracy"]
                or (
                    validation_accuracy == history["best_validation_accuracy"]
                    and validation_loss < history["best_validation_loss"]
                )
            )
            if better:
                history["best_epoch"] = epoch
                history["best_validation_loss"] = float(validation_loss)
                history["best_validation_accuracy"] = float(validation_accuracy)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose and history["best_epoch"] is not None:
            print(
                f"[Best Validation] epoch {history['best_epoch']}: "
                f"{history['best_validation_accuracy']:.2f}%"
            )
        return history


def quick_recovery(
    pruned_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[nn.Module, Dict]:
    """Recover an isolated copy of a pruned model using validation data."""
    recoverer = ReconstructionRecovery(
        model=copy.deepcopy(pruned_model), device=device, learning_rate=learning_rate
    )
    history = recoverer.recover(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
        verbose=verbose,
    )
    return recoverer.model, history
