import pytest

from src.utils.precision import precision_metadata, resolve_precision


def test_fp32_is_valid_on_cpu():
    assert resolve_precision("fp32", "cpu") == "fp32"
    assert precision_metadata("cpu", "fp32")["grad_scaler"] is False


def test_amp_precision_requires_cuda():
    with pytest.raises(ValueError, match="requires a CUDA"):
        resolve_precision("fp16", "cpu")


def test_unknown_precision_fails():
    with pytest.raises(ValueError, match="unsupported precision"):
        resolve_precision("int8", "cpu")
