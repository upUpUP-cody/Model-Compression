import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from src.models.dense_baseline import MLP
from src.utils.experiment_artifacts import (
    RunArtifacts,
    load_config,
    set_seed,
    to_json_safe,
    validate_config,
)
from src.utils.search_visualization import plot_search_history


CONFIG_PATH = Path("archive/configs_legacy/mnist_mlp_autonomous_cpu.yaml")


def test_cpu_config_loads_and_rejects_unavailable_cuda_and_cpu_amp():
    config = load_config(CONFIG_PATH)
    assert config["hardware"]["device"] == "cpu"
    assert config["hardware"]["mixed_precision"] is False

    invalid = json.loads(json.dumps(config))
    invalid["hardware"]["device"] = "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="invalid hardware configuration"):
            validate_config(invalid)

    invalid["hardware"]["device"] = "cpu"
    invalid["hardware"]["mixed_precision"] = True
    with pytest.raises(ValueError, match="mixed_precision"):
        validate_config(invalid)


def test_gpu_config_passes_validation_when_cuda_is_available():
    config = load_config(CONFIG_PATH)
    gpu_config = json.loads(json.dumps(config))
    gpu_config["hardware"]["device"] = "cuda:0"
    gpu_config["hardware"]["mixed_precision"] = False
    gpu_config["hardware"]["precision"] = "fp32"
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable on this host")
    validate_config(gpu_config)


def test_set_seed_reproduces_python_numpy_and_torch_values():
    set_seed(123)
    first = (np.random.rand(), torch.rand(3))
    set_seed(123)
    second = (np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_to_json_safe_converts_supported_scalar_values():
    value = {
        "path": Path("results/run"),
        "numpy": np.int64(4),
        "tensor": torch.tensor(2.5),
        "tuple": (1, "x"),
    }
    normalized = to_json_safe(value)
    assert normalized["path"] == str(Path("results/run"))
    assert normalized["numpy"] == 4
    assert normalized["tensor"] == 2.5
    assert normalized["tuple"] == [1, "x"]
    with pytest.raises(TypeError, match="only scalar tensors"):
        to_json_safe(torch.ones(2))


def test_run_artifacts_and_plot_are_serializable(tmp_path):
    artifacts = RunArtifacts(tmp_path)
    config = load_config(CONFIG_PATH)
    artifacts.save_config(config)
    event = {
        "iteration": 1,
        "fingerprint": "abc",
        "action": "accept",
        "final_action": "accept",
        "final_reason": "constraints_satisfied",
        "cheap_critic": {
            "accuracy": 90.0,
            "loss": 0.2,
            "parameter_count": 100,
        },
        "validation": {"accuracy": 89.5},
    }
    artifacts.append_event(event)
    artifacts.save_history_csv([event])
    model = MLP(
        input_dim=4,
        hidden_dims=[3],
        num_classes=2,
        dropout_rate=0.0,
        use_batch_norm=False,
    )
    checkpoint = artifacts.save_checkpoint(model)
    artifacts.save_summary({"events": [event], "checkpoint": checkpoint})
    chart = plot_search_history([event], artifacts.run_dir / "search_history.png")

    assert (artifacts.run_dir / "resolved_config.yaml").exists()
    assert json.loads((artifacts.run_dir / "resolved_config.json").read_text()) == config
    assert json.loads(artifacts.events_path.read_text().splitlines()[0]) == event
    assert (artifacts.run_dir / "history.csv").stat().st_size > 0
    assert json.loads((artifacts.run_dir / "summary.json").read_text())["checkpoint"] == str(checkpoint)
    assert torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    assert chart.exists() and chart.stat().st_size > 0


def test_plot_requires_accepted_event(tmp_path):
    with pytest.raises(ValueError, match="accepted event"):
        plot_search_history([], tmp_path / "empty.png")
