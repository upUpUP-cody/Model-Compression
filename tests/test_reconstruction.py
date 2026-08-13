import copy
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.recovery.reconstruction import ReconstructionRecovery, quick_recovery


class ScriptedRecovery(ReconstructionRecovery):
    def __init__(self, model, validation_metrics):
        super().__init__(model, learning_rate=0.0)
        self.validation_metrics = iter(validation_metrics)
        self.epoch_weight = 0

    def train_epoch(self, train_loader):
        self.epoch_weight += 1
        with torch.no_grad():
            self.model.weight.fill_(float(self.epoch_weight))
        return float(self.epoch_weight), 50.0

    def evaluate(self, validation_loader):
        return next(self.validation_metrics)


def make_loader(samples=4):
    features = torch.zeros(samples, 1)
    targets = torch.zeros(samples, dtype=torch.long)
    return DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)


def make_model():
    return nn.Linear(1, 2, bias=False)


def test_recovery_restores_best_validation_epoch_not_last_epoch():
    recovery = ScriptedRecovery(
        make_model(), validation_metrics=[(0.3, 90.0), (0.4, 80.0)]
    )

    history = recovery.recover(make_loader(), make_loader(), epochs=2, verbose=False)

    assert history == {
        "train_loss": [1.0, 2.0],
        "train_acc": [50.0, 50.0],
        "validation_loss": [0.3, 0.4],
        "validation_accuracy": [90.0, 80.0],
        "best_epoch": 1,
        "best_validation_loss": 0.3,
        "best_validation_accuracy": 90.0,
    }
    assert torch.equal(recovery.model.weight, torch.ones_like(recovery.model.weight))
    json.dumps(history)


def test_recovery_uses_lower_validation_loss_as_accuracy_tiebreak():
    recovery = ScriptedRecovery(
        make_model(), validation_metrics=[(0.4, 90.0), (0.2, 90.0)]
    )

    history = recovery.recover(make_loader(), make_loader(), epochs=2, verbose=False)

    assert history["best_epoch"] == 2
    assert history["best_validation_loss"] == 0.2
    assert torch.equal(recovery.model.weight, torch.full_like(recovery.model.weight, 2.0))


def test_zero_epoch_recovery_leaves_model_and_history_unmodified():
    model = make_model()
    expected_state = copy.deepcopy(model.state_dict())
    recovery = ReconstructionRecovery(model, learning_rate=0.1)

    history = recovery.recover(make_loader(), make_loader(), epochs=0, verbose=False)

    assert history["train_loss"] == []
    assert history["validation_accuracy"] == []
    assert history["best_epoch"] is None
    assert history["best_validation_accuracy"] is None
    for name, expected in expected_state.items():
        assert torch.equal(recovery.model.state_dict()[name], expected)


def test_recovery_rejects_invalid_epoch_count_and_empty_loaders():
    recovery = ReconstructionRecovery(make_model())
    empty_loader = make_loader(samples=0)

    with pytest.raises(ValueError, match="non-negative integer"):
        recovery.recover(make_loader(), make_loader(), epochs=-1, verbose=False)
    with pytest.raises(ValueError, match="train dataloader produced no samples"):
        recovery.recover(empty_loader, make_loader(), epochs=1, verbose=False)
    with pytest.raises(ValueError, match="validation dataloader produced no samples"):
        recovery.recover(make_loader(), empty_loader, epochs=1, verbose=False)


def test_quick_recovery_does_not_mutate_its_input_model():
    torch.manual_seed(7)
    model = make_model()
    source_state = copy.deepcopy(model.state_dict())
    train_loader = DataLoader(
        TensorDataset(torch.tensor([[1.0], [-1.0]]), torch.tensor([0, 1])), batch_size=2
    )
    validation_loader = train_loader

    recovered, history = quick_recovery(
        model,
        train_loader,
        validation_loader,
        epochs=1,
        learning_rate=0.1,
        verbose=False,
    )

    assert recovered is not model
    assert history["best_epoch"] == 1
    assert any(
        not torch.equal(recovered.state_dict()[name], expected)
        for name, expected in source_state.items()
    )
    for name, expected in source_state.items():
        assert torch.equal(model.state_dict()[name], expected)
