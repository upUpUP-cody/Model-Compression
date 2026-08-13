import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.run_p12_multiseed import run_multiseed
from src.experiments.compression_targets import apply_compression_target, derive_uniform_layer_ratios
from src.experiments.p12_aggregate import aggregate_root, load_study_row, summarize_rows
from src.models.dense_baseline import MLP
from src.utils.experiment_artifacts import config_hash, file_sha256


def test_derive_uniform_layer_ratios_moves_toward_target():
    model_config = {
        "input_dim": 16,
        "hidden_dims": [8, 4],
        "num_classes": 2,
        "dropout_rate": 0.0,
        "use_batch_norm": False,
    }
    baseline = MLP(**model_config)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    ratios = derive_uniform_layer_ratios(model_config, target_compression_ratio=2.0)
    pruned = __import__("src.pruning.structured_pruning", fromlist=["StructuredPruning"]).StructuredPruning(baseline)
    pruned_model = pruned.create_pruned_model(ratios)
    pruned_count = sum(parameter.numel() for parameter in pruned_model.parameters())
    assert pruned_count <= baseline_count / 2.0 + 50
    assert all(0.0 <= ratio < 1.0 for ratio in ratios.values())


def test_apply_compression_target_updates_comparison_fields():
    config = {
        "model": {
            "input_dim": 16,
            "hidden_dims": [8, 4],
            "num_classes": 2,
            "dropout_rate": 0.0,
            "use_batch_norm": False,
        },
        "comparison": {"target_compression_ratio": 1.0},
        "search": {"candidate_ratios": [0.1]},
    }
    updated = apply_compression_target(config, 4.0)
    assert updated["comparison"]["target_compression_ratio"] == 4.0
    assert updated["comparison"]["oneshot_layer_ratios"]
    assert updated["comparison"]["iterative_stage_ratios"]


def test_aggregate_root_computes_mean_and_std(tmp_path):
    study = tmp_path / "study_a"
    study.mkdir()
    records = [{
        "method": "dense_baseline",
        "validation": {"accuracy": 90.0, "loss": 0.1},
        "target_compression_ratio": 2.0,
        "compression_ratio": 1.0,
        "parameter_count": 100,
    }]
    manifest = {
        "seed": 42,
        "config_hash": "abc",
        "records": records,
    }
    comparison = {"records": records}
    test_report = {"reports": [{"method": "dense_baseline", "test": {"accuracy": 88.0, "loss": 0.2}}]}
    (study / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (study / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (study / "final_test_report.json").write_text(json.dumps(test_report), encoding="utf-8")

    study_b = tmp_path / "study_b"
    study_b.mkdir()
    records_b = [{
        "method": "dense_baseline",
        "validation": {"accuracy": 92.0, "loss": 0.1},
        "target_compression_ratio": 2.0,
        "compression_ratio": 1.0,
        "parameter_count": 100,
    }]
    manifest_b = {"seed": 123, "config_hash": "def", "records": records_b}
    comparison_b = {"records": records_b}
    test_report_b = {"reports": [{"method": "dense_baseline", "test": {"accuracy": 90.0, "loss": 0.2}}]}
    (study_b / "manifest.json").write_text(json.dumps(manifest_b), encoding="utf-8")
    (study_b / "comparison.json").write_text(json.dumps(comparison_b), encoding="utf-8")
    (study_b / "final_test_report.json").write_text(json.dumps(test_report_b), encoding="utf-8")

    result = aggregate_root(tmp_path)
    summary = result["summary"]
    assert len(summary) == 1
    assert summary[0]["method"] == "dense_baseline"
    assert summary[0]["num_runs"] == 2
    assert summary[0]["validation_accuracy_mean"] == 91.0
    assert summary[0]["test_accuracy_mean"] == 89.0


def test_run_multiseed_invokes_study_and_report(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 7
seeds: [7, 8]
hardware:
  device: cpu
  precision: fp32
  mixed_precision: false
dataset:
  batch_size: 16
search:
  max_iterations: 1
  candidate_ratios: [0.5]
controller:
  max_accuracy_drop_points: 100.0
recovery:
  epochs: 0
logging:
  output_root: {root}
comparison:
  target_compression_ratio: 2.0
  methods: [dense_baseline]
""".format(root=tmp_path / "runs"),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "baseline.pth"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    calls = []

    def fake_study(config, checkpoint_source, command):
        calls.append(config["seed"])
        study_dir = Path(config["logging"]["output_root"]) / f"study_{config['seed']}"
        study_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "method": "dense_baseline",
            "validation": {"accuracy": 90.0, "loss": 0.1},
            "target_compression_ratio": 2.0,
            "compression_ratio": 1.0,
            "parameter_count": 10,
        }
        (study_dir / "manifest.json").write_text(json.dumps({"seed": config["seed"], "records": [record], "config_hash": config_hash(config)}), encoding="utf-8")
        (study_dir / "comparison.json").write_text(json.dumps({"records": [record]}), encoding="utf-8")
        (study_dir / "final_test_report.json").write_text(json.dumps({"reports": [{"method": "dense_baseline", "test": {"accuracy": 89.0, "loss": 0.2}}]}), encoding="utf-8")
        return study_dir

    monkeypatch.setattr("experiments.run_p12_multiseed.run_study", fake_study)
    monkeypatch.setattr("experiments.run_p12_multiseed.run_test_report", lambda *args, **kwargs: args[0] / "final_test_report.json")

    with patch("src.utils.experiment_artifacts.validate_config", return_value=None):
        manifest = run_multiseed(config_path, checkpoint, sweep=False)
    assert calls == [7, 8]
    assert len(manifest["study_dirs"]) == 2
