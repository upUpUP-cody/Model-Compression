"""Compression-target derivation and CNN conv1-only compression limits."""
from pathlib import Path

import yaml

from src.experiments.compression_targets import (
    apply_compression_target,
    ensure_compression_target,
    parameter_count,
)
from src.experiments.model_factory import build_model_from_config
from src.pruning.pruning_backend import resolve_pruning_backend


RESNET_CFG = {"type": "resnet_cifar", "num_classes": 10, "base_width": 64}
CONV1_NAMES = {f"layer{stage}.{block}.conv1" for stage in range(1, 5) for block in range(2)}
CIFAR_TARGET_CONFIGS = [
    "configs/cifar_p12_gpu_study.yaml",
    "configs/cifar_p12_gpu_multiseed.yaml",
    "configs/cifar_p12_gpu_sweep.yaml",
    "configs/cifar_p12_gpu_smoke.yaml",
]


def _compression_at_uniform_ratio(prune_ratio: float) -> float:
    model = build_model_from_config({"model": dict(RESNET_CFG)})
    baseline = parameter_count(model)
    backend = resolve_pruning_backend(model, "resnet_cifar")
    ratios = {name: prune_ratio for name in backend.prunable_layer_names()}
    pruned = backend.create_pruned_model(ratios)
    return baseline / parameter_count(pruned)


def _actual_compression(ratios: dict) -> float:
    model = build_model_from_config({"model": dict(RESNET_CFG)})
    backend = resolve_pruning_backend(model, "resnet_cifar")
    return parameter_count(model) / parameter_count(backend.create_pruned_model(ratios))


def test_conv1_only_uniform_ratios_reach_targets():
    ratios = {
        0.3: _compression_at_uniform_ratio(0.3),
        0.5: _compression_at_uniform_ratio(0.5),
        0.8: _compression_at_uniform_ratio(0.8),
        0.95: _compression_at_uniform_ratio(0.95),
    }
    print(f"[INFO] conv1-only compression: {ratios}")
    assert ratios[0.5] >= 1.7
    assert ratios[0.8] >= 4.0
    assert ratios[0.95] >= 10.0


def test_apply_compression_target_hits_two_four_and_ten_x():
    for target, low, high in ((2.0, 1.7, 2.3), (4.0, 3.4, 4.6), (10.0, 8.5, 11.5)):
        updated = apply_compression_target(
            {"model": dict(RESNET_CFG), "comparison": {}, "search": {"candidate_ratios": [0.3]}},
            target,
        )
        oneshot = updated["comparison"]["oneshot_layer_ratios"]
        assert set(oneshot) == CONV1_NAMES
        assert updated["comparison"]["iterative_stage_ratios"] == [oneshot]
        assert updated["comparison"]["layer_ratios_derived_for"] == target
        actual = _actual_compression(oneshot)
        assert low <= actual <= high, f"target {target}x actual {actual:.3f}x"


def test_ensure_compression_target_is_idempotent():
    oneshot = {name: 0.5 for name in CONV1_NAMES}
    config = {
        "model": dict(RESNET_CFG),
        "comparison": {
            "target_compression_ratio": 2.0,
            "oneshot_layer_ratios": oneshot,
            "layer_ratios_derived_for": 2.0,
        },
        "search": {"candidate_ratios": [0.5]},
    }
    second = ensure_compression_target(config)
    assert second["comparison"]["oneshot_layer_ratios"] == oneshot
    assert second["comparison"]["layer_ratios_derived_for"] == 2.0


def test_cifar_study_configs_do_not_handwrite_single_layer_ratios():
    root = Path(__file__).resolve().parents[1]
    for relative in CIFAR_TARGET_CONFIGS:
        loaded = yaml.safe_load((root / relative).read_text(encoding="utf-8"))
        comparison = loaded["comparison"]
        assert comparison.get("oneshot_layer_ratios") in (None, {})
        assert comparison.get("iterative_stage_ratios") in (None, [])
        assert float(comparison["target_compression_ratio"]) > 1.0
