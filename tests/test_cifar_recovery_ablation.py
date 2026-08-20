"""Recovery ablation config + dispatch smoke tests."""
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_cifar_recovery_ablation import _mean_std
from src.experiments.compression_targets import ensure_compression_target
from src.models.dense_baseline import MLP
from src.recovery.recovery_dispatch import run_recovery

CONV1_NAMES = {f"layer{stage}.{block}.conv1" for stage in range(1, 5) for block in range(2)}


def test_ablation_config_derives_all_conv1_ratios():
    root = Path(__file__).resolve().parents[1]
    loaded = yaml.safe_load((root / "archive/configs_legacy/cifar_recovery_ablation.yaml").read_text(encoding="utf-8"))
    assert loaded["comparison"].get("oneshot_layer_ratios") in (None, {})
    assert float(loaded["comparison"]["target_compression_ratio"]) == 2.0
    assert loaded["comparison"]["recovery_levels"] == [1, 2, 3]
    assert loaded["seeds"] == [42, 43, 44]
    assert loaded["comparison"].get("report_test") is True

    updated = ensure_compression_target(loaded)
    ratios = updated["comparison"]["oneshot_layer_ratios"]
    assert set(ratios) == CONV1_NAMES
    assert all(0.0 < ratio < 1.0 for ratio in ratios.values())
    assert len(set(round(ratio, 6) for ratio in ratios.values())) == 1


def test_mean_std_aggregation():
    stats = _mean_std([1.0, 3.0])
    assert stats["n"] == 2
    assert abs(stats["mean"] - 2.0) < 1e-9
    assert abs(stats["std"] - 2.0 ** 0.5) < 1e-9
    assert _mean_std([])["mean"] is None


def test_run_recovery_levels_one_two_three_on_mlp():
    teacher = MLP(input_dim=4, hidden_dims=[8], num_classes=2, use_batch_norm=False)
    student = MLP(input_dim=4, hidden_dims=[8], num_classes=2, use_batch_norm=False)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=4)
    base_config = {
        "hardware": {"device": "cpu", "precision": "fp32"},
        "recovery": {
            "epochs": 1,
            "learning_rate": 0.01,
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "distill_temperature": 2.0,
            "distill_alpha": 0.5,
        },
    }
    for level in (1, 2, 3):
        cfg = {
            "hardware": dict(base_config["hardware"]),
            "recovery": {**base_config["recovery"], "level": level},
        }
        recovered, history = run_recovery(
            student,
            loader,
            loader,
            cfg,
            teacher_model=teacher,
            verbose=False,
        )
        assert history["recovery_level"] == level
        assert recovered(torch.randn(2, 4)).shape == (2, 2)
        assert history.get("best_validation_accuracy") is not None
