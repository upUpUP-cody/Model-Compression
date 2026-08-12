import copy
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.cheap_critic import CheapCritic


def make_loader():
    inputs = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [3.0], [-3.0], [4.0]])
    targets = torch.tensor([1, 0, 1, 0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)


def make_model():
    model = nn.Sequential(nn.BatchNorm1d(1), nn.Linear(1, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model[1].bias.zero_()
    return model


def test_critic_applies_exact_sample_limit_without_model_side_effects():
    model = make_model()
    model.train()
    state_before = copy.deepcopy(model.state_dict())

    result = CheapCritic().evaluate(model, make_loader(), max_samples=5)

    assert result.samples == 5
    assert result.accuracy == pytest.approx(100.0)
    assert result.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert result.nonzero_parameter_count == 3
    assert result.elapsed_seconds >= 0.0
    assert model.training
    for name, value in state_before.items():
        assert torch.equal(model.state_dict()[name], value), name
    assert json.loads(json.dumps(result.to_dict()))["samples"] == 5


def test_critic_restores_eval_mode_and_reports_weighted_loss():
    model = make_model()
    model.eval()
    result = CheapCritic(nn.CrossEntropyLoss()).evaluate(model, make_loader(), max_samples=4)

    expected_logits = model(torch.tensor([[1.0], [-1.0], [2.0], [-2.0]]))
    expected_loss = nn.CrossEntropyLoss()(expected_logits, torch.tensor([1, 0, 1, 0]))
    assert result.loss == pytest.approx(expected_loss.item())
    assert not model.training


@pytest.mark.parametrize("max_samples", [0, -1, True, 1.5])
def test_critic_rejects_invalid_sample_limit(max_samples):
    with pytest.raises(ValueError):
        CheapCritic().evaluate(make_model(), make_loader(), max_samples=max_samples)


def test_critic_rejects_empty_loader():
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 1), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError):
        CheapCritic().evaluate(make_model(), empty_loader, max_samples=1)
