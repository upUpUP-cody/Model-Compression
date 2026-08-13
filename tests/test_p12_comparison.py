import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_p12_comparison import run_test_report, verify_frozen_study
from src.experiments.p12_comparison import METHOD_NAMES, results_to_records, run_comparison
from src.models.dense_baseline import MLP
from src.utils.experiment_artifacts import RunArtifacts, config_hash, file_sha256, git_sha


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_model():
    torch.manual_seed(7)
    return MLP(
        input_dim=4,
        hidden_dims=[4, 3],
        num_classes=2,
        dropout_rate=0.0,
        use_batch_norm=False,
    )


def make_loader():
    torch.manual_seed(8)
    return DataLoader(
        TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,))),
        batch_size=4,
        shuffle=False,
    )


def make_config():
    return {
        "seed": 4,
        "hardware": {"device": "cpu"},
        "dataset": {"split_seed": 4},
        "controller": {
            "max_accuracy_drop_points": 100.0,
            "max_failures": 3,
            "min_quality_score": 0.0,
            "regrow_ratio_multiplier": 0.8,
        },
        "search": {
            "max_iterations": 1,
            "candidate_ratios": [0.5],
            "candidates_per_round": 1,
            "cheap_eval_samples": 4,
            "recovery_top_k": 1,
        },
        "recovery": {"epochs": 0, "learning_rate": 0.001},
        "comparison": {
            "methods": list(METHOD_NAMES),
            "target_compression_ratio": 2.0,
            "dense_small_hidden_dims": [2, 2],
            "oneshot_layer_ratios": {"features.0": 0.5},
            "wanda_batches": 1,
            "iterative_stage_ratios": [{"features.0": 0.5}],
            "recovery_epochs": 0,
            "recovery_learning_rate": 0.001,
        },
    }


def make_frozen_study(tmp_path, config):
    checkpoint_source = tmp_path / "source.pth"
    torch.save({"model_state_dict": make_model().state_dict()}, checkpoint_source)
    study = tmp_path / "study"
    study.mkdir()
    records = []
    for method in METHOD_NAMES:
        selected = study / f"{method}.pth"
        torch.save({"model_state_dict": make_model().state_dict()}, selected)
        records.append({
            "method": method,
            "checkpoint": str(selected),
            "checkpoint_sha256": file_sha256(selected),
            "model_hidden_dims": [4, 3],
        })
    manifest = {
        "protocol": "p12_validation_only_study",
        "selection_frozen": True,
        "config_hash": config_hash(config),
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_source_sha256": file_sha256(checkpoint_source),
        "split": {"split_seed": 4, "train_size": 1, "validation_size": 1, "split_hash": "split"},
        "runtime": {"device": {"type": "cpu", "cuda_available": False}},
        "cuda_policy": {"device": "cpu"},
        "records": records,
    }
    (study / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint_source, study, manifest


def test_all_comparison_arms_are_validation_only_and_isolate_source_model():
    source = make_model()
    source_state = copy.deepcopy(source.state_dict())
    results, models = run_comparison(source, make_loader(), make_loader(), make_config())

    assert tuple(results) == METHOD_NAMES
    assert tuple(models) == METHOD_NAMES
    assert all(result.status == "completed" for result in results.values())
    assert all("accuracy" in result.validation for result in results.values())
    assert results["oneshot_magnitude"].parameter_count < results["dense_baseline"].parameter_count
    assert results["oneshot_wanda"].parameter_count < results["dense_baseline"].parameter_count
    assert models["oneshot_magnitude"].features[0].out_features == 2
    assert models["oneshot_wanda"].features[0].out_features == 2
    for name, expected in source_state.items():
        assert torch.equal(source.state_dict()[name], expected), name
    json.dumps(results_to_records(results))


def test_comparison_artifacts_write_json_and_flat_csv(tmp_path):
    results, _ = run_comparison(make_model(), make_loader(), make_loader(), make_config())
    artifacts = RunArtifacts(tmp_path, run_id="comparison")
    comparison_json, comparison_csv = artifacts.save_comparison(results_to_records(results))

    assert len(json.loads(comparison_json.read_text())["records"]) == 6
    assert "oneshot_wanda" in comparison_csv.read_text()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest, study: manifest.update(protocol="wrong"), "protocol"),
        (lambda manifest, study: manifest.update(git_sha="wrong"), "git SHA"),
        (lambda manifest, study: manifest["records"].pop(), "exactly six"),
        (lambda manifest, study: manifest["records"].__setitem__(0, {**manifest["records"][0], "method": "unknown"}), "method set"),
        (lambda manifest, study: manifest["split"].update(split_hash=""), "split hash"),
        (lambda manifest, study: manifest["records"][0].update(checkpoint=str(study.parent / "outside.pth")), "inside the study"),
    ],
)
def test_report_gate_rejects_invalid_manifest_before_loading_test(monkeypatch, tmp_path, mutation, message):
    config = make_config()
    checkpoint_source, study, manifest = make_frozen_study(tmp_path, config)
    mutation(manifest, study)
    (study / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "experiments.run_p12_comparison.get_mnist_loaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test loader was accessed")),
    )

    with pytest.raises(ValueError, match=message):
        run_test_report(study, config, checkpoint_source)


def test_report_gate_rejects_preexisting_final_report_before_loading_test(monkeypatch, tmp_path):
    config = make_config()
    checkpoint_source, study, _ = make_frozen_study(tmp_path, config)
    (study / "final_test_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "experiments.run_p12_comparison.get_mnist_loaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test loader was accessed")),
    )

    with pytest.raises(ValueError, match="already has a final test report"):
        run_test_report(study, config, checkpoint_source)


def test_verify_frozen_study_accepts_complete_manifest(tmp_path):
    config = make_config()
    checkpoint_source, study, _ = make_frozen_study(tmp_path, config)

    assert verify_frozen_study(study, config, checkpoint_source)["selection_frozen"] is True
