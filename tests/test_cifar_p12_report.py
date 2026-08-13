import torch

from experiments.run_cifar_p12_comparison import rebuild_cifar_model
from src.experiments.model_factory import build_model_from_config
from src.pruning.pruning_backend import resolve_pruning_backend


def _config():
    return {
        "model": {"type": "resnet_cifar", "num_classes": 10, "base_width": 16},
        "comparison": {"dense_small_variant": "small"},
        "hardware": {"device": "cpu"},
    }


def test_rebuild_dense_baseline_matches_full_resnet():
    config = _config()
    model = rebuild_cifar_model(config, {"method": "dense_baseline"}, "cpu")
    baseline = build_model_from_config(config)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())


def test_rebuild_pruned_model_matches_keep_indices():
    config = _config()
    source = build_model_from_config(config)
    keep_indices = {"layer1.0.conv1": [0, 1]}
    expected = resolve_pruning_backend(source, "resnet_cifar").create_pruned_model_by_indices(keep_indices)
    rebuilt = rebuild_cifar_model(
        config,
        {"method": "oneshot_wanda", "layer_keep_indices": keep_indices},
        "cpu",
    )
    assert rebuilt(torch.randn(1, 3, 32, 32)).shape == (1, 10)
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(p.numel() for p in expected.parameters())
