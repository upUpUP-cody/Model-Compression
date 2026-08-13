import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.frontier import FrontierPoint, ParetoFrontier
from src.utils import data_loader
from src.utils.experiment_artifacts import (
    RunArtifacts,
    config_hash,
    file_sha256,
    set_cpu_threads,
)


class FakeMNIST(TensorDataset):
    def __init__(self, root, train, download, transform):
        size = 20 if train else 8
        super().__init__(torch.zeros(size, 1, 28, 28), torch.arange(size) % 10)


def test_mnist_split_is_seeded_and_official_test_is_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader.datasets, "MNIST", FakeMNIST)
    first_train, first_validation, first_test = data_loader.get_mnist_loaders(
        data_dir=str(tmp_path), batch_size=4, validation_fraction=0.2, split_seed=7,
        return_test=True,
    )
    second_train, second_validation, _ = data_loader.get_mnist_loaders(
        data_dir=str(tmp_path), batch_size=4, validation_fraction=0.2, split_seed=7,
        return_test=True,
    )

    assert first_train.dataset.indices == second_train.dataset.indices
    assert first_validation.dataset.indices == second_validation.dataset.indices
    assert set(first_train.dataset.indices).isdisjoint(first_validation.dataset.indices)
    assert len(first_train.dataset) + len(first_validation.dataset) == 20
    assert len(first_test.dataset) == 8
    metadata = data_loader.mnist_split_metadata(first_train, first_validation, 7)
    assert metadata["train_size"] == 16
    assert metadata["validation_size"] == 4
    assert len(metadata["split_hash"]) == 64


def test_artifact_manifest_frontier_and_performance_outputs(tmp_path):
    artifacts = RunArtifacts(tmp_path, run_id="fixed")
    frontier = ParetoFrontier([FrontierPoint(90.0, 10, 2.0)])
    frontier_json, frontier_csv = artifacts.save_frontier(frontier)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    manifest = artifacts.save_manifest({"checkpoint_hash": file_sha256(checkpoint)})

    loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=4)
    performance = artifacts.measure_inference(nn.Linear(2, 2), loader)

    assert json.loads(frontier_json.read_text())["points"][0]["parameter_count"] == 10
    assert frontier_csv.stat().st_size > 0
    assert json.loads(manifest.read_text())["checkpoint_hash"] == file_sha256(checkpoint)
    assert performance["latency_seconds"] >= 0.0
    assert performance["throughput_samples_per_second"] > 0.0
    assert config_hash({"a": 1}) == config_hash({"a": 1})


def test_cpu_thread_setting_rejects_invalid_values():
    original = torch.get_num_threads()
    try:
        assert set_cpu_threads(1) == 1
        try:
            set_cpu_threads(0)
        except ValueError:
            pass
        else:
            raise AssertionError("zero CPU threads should fail")
    finally:
        torch.set_num_threads(original)
