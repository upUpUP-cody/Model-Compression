import pytest
import torch

from src.utils.device import device_metadata, resolve_device


def test_cpu_device_resolves_without_cuda():
    assert resolve_device("cpu") == torch.device("cpu")
    assert device_metadata("cpu")["type"] == "cpu"


def test_cuda_request_fails_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA device requested"):
        resolve_device("cuda:0")


def test_unsupported_device_type_fails():
    with pytest.raises(ValueError, match="unsupported device type"):
        resolve_device("xpu")
