"""Tests for CIFAR comparison recovery-budget override."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.cifar_p12_comparison import _iterative_structured
from src.models.resnet_cifar import ResNetCIFAR
from src.pruning.pruning_backend import resolve_pruning_backend


def _tiny_config() -> Dict[str, Any]:
    return {
        "hardware": {"device": "cpu", "precision": "fp32"},
        "recovery": {"level": 1, "epochs": 2, "learning_rate": 0.001},
        "comparison": {
            "iterative_stage_ratios": [{"layer1.0.conv1": 0.25}],
            "wanda_batches": 1,
            "iterative_recovery_epochs": 6,
        },
    }


def _stub_keep(monkeypatch, backend):
    names = backend.prunable_layer_names()

    def fake_keep_indices(current_backend, model, train_loader, ratios, importance_method, device, comparison, model_type):
        keep = {}
        for name in names:
            module = dict(model.named_modules())[name]
            width = int(module.weight.shape[0])
            keep[name] = list(range(max(1, width // 2)))
        return keep

    monkeypatch.setattr("src.experiments.cifar_p12_comparison._keep_indices", fake_keep_indices)
    monkeypatch.setattr(
        "src.experiments.cifar_p12_comparison._layer_ratios",
        lambda model, ratio_config, model_type: {name: 0.5 for name in names},
    )


def test_iterative_structured_honors_iterative_recovery_epochs(monkeypatch):
    torch.manual_seed(0)
    model = ResNetCIFAR(num_classes=10, base_width=16)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))),
        batch_size=4,
    )
    seen_epochs = []

    def fake_recovery(model, train_loader, validation_loader, config, teacher_model=None, verbose=False):
        seen_epochs.append(int(config["recovery"]["epochs"]))
        return model, {"epochs": config["recovery"]["epochs"]}

    monkeypatch.setattr("src.experiments.cifar_p12_comparison.run_recovery", fake_recovery)
    backend = resolve_pruning_backend(model, "resnet_cifar")
    _stub_keep(monkeypatch, backend)
    cfg = _tiny_config()
    out_model, details, _ = _iterative_structured(
        model, backend, loader, loader, cfg, "cpu", "resnet_cifar"
    )
    assert seen_epochs == [6]
    assert details["stages"][0]["recovery_epochs"] == 6
    assert int(cfg["recovery"]["epochs"]) == 2
    assert out_model is not None


def test_iterative_structured_defaults_to_recovery_epochs(monkeypatch):
    torch.manual_seed(0)
    model = ResNetCIFAR(num_classes=10, base_width=16)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))),
        batch_size=4,
    )
    seen_epochs = []

    def fake_recovery(model, train_loader, validation_loader, config, teacher_model=None, verbose=False):
        seen_epochs.append(int(config["recovery"]["epochs"]))
        return model, {}

    monkeypatch.setattr("src.experiments.cifar_p12_comparison.run_recovery", fake_recovery)
    backend = resolve_pruning_backend(model, "resnet_cifar")
    _stub_keep(monkeypatch, backend)
    cfg = _tiny_config()
    del cfg["comparison"]["iterative_recovery_epochs"]
    _iterative_structured(model, backend, loader, loader, cfg, "cpu", "resnet_cifar")
    assert seen_epochs == [2]
