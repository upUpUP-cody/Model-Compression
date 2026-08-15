"""Smoke tests for crossover path ablation helpers."""
from experiments.run_cifar_crossover_path_ablation import _formal_study_dirname, _mean_std, _verdict


def test_mean_std_and_verdict_helpers():
    stats = _mean_std([1.0, 3.0])
    assert stats["n"] == 2
    assert abs(stats["mean"] - 2.0) < 1e-9
    rows = [
        {"arm": "uniform", "test_accuracy": {"mean": 83.0, "std": 0.1, "n": 3}},
        {"arm": "incremental_no_gate", "test_accuracy": {"mean": 86.0, "std": 0.1, "n": 3}},
        {"arm": "search_gated", "test_accuracy": {"mean": 86.2, "std": 0.1, "n": 3}},
    ]
    assert _verdict(rows) == "path_dominant"
    rows[1]["test_accuracy"]["mean"] = 83.2
    rows[2]["test_accuracy"]["mean"] = 86.0
    assert _verdict(rows) == "gate_dominant"


def test_formal_study_dirname_matches_sweep_layout():
    assert _formal_study_dirname(4.0, 42) == "ratio_4_seed_42"
    assert _formal_study_dirname(8.0, 42) == "ratio_8_seed_42"
    assert _formal_study_dirname(10.0, 43) == "ratio_10_seed_43"
    assert _formal_study_dirname(1.5, 44) == "ratio_1.5_seed_44"
