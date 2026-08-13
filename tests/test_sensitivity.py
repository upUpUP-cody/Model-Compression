import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense_baseline import MLP
from src.pruning.sensitivity import SensitivityAnalyzer


def make_model():
    torch.manual_seed(17)
    return MLP(
        input_dim=4,
        hidden_dims=[4, 3],
        num_classes=2,
        dropout_rate=0.0,
        use_batch_norm=False,
    )


def make_loader():
    torch.manual_seed(19)
    return DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))),
        batch_size=2,
        shuffle=False,
    )


def test_wanda_importance_streams_all_hidden_layers_and_restores_mode():
    model = make_model()
    model.train()
    analyzer = SensitivityAnalyzer(model)

    scores = analyzer.compute_wanda_importance(make_loader(), num_batches=2)

    assert scores.keys() == {"features.0", "features.2"}
    assert scores["features.0"].shape == (4,)
    assert scores["features.2"].shape == (3,)
    assert model.training
    assert not model.features[0]._forward_hooks
    assert not model.features[2]._forward_hooks


def test_wanda_importance_restores_mode_and_hooks_after_forward_error():
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(4, 3)
            self.classifier = nn.Linear(3, 2)

        def forward(self, data):
            self.hidden(data)
            raise RuntimeError("expected forward failure")

    model = FailingModel()
    model.eval()

    with pytest.raises(RuntimeError, match="expected forward failure"):
        SensitivityAnalyzer(model).compute_wanda_importance(make_loader(), num_batches=1)

    assert not model.training
    assert not model.hidden._forward_hooks


@pytest.mark.parametrize("num_batches", [0, -1, True, 1.5])
def test_wanda_importance_rejects_invalid_batch_budget(num_batches):
    with pytest.raises(ValueError, match="positive integer"):
        SensitivityAnalyzer(make_model()).compute_wanda_importance(make_loader(), num_batches=num_batches)


def test_wanda_importance_rejects_empty_loader_and_models_without_hidden_linear():
    empty_loader = DataLoader(
        TensorDataset(torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="must yield"):
        SensitivityAnalyzer(make_model()).compute_wanda_importance(empty_loader, num_batches=1)

    classifier_only = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="hidden Linear"):
        SensitivityAnalyzer(classifier_only).compute_wanda_importance(make_loader(), num_batches=1)
