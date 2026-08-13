import torch
import pytest

from src.utils.gpu_benchmark import benchmark_callable, benchmark_model
from src.models.dense_baseline import MLP


def test_cpu_benchmark_schema():
    result = benchmark_callable(lambda: torch.ones(2), repeat=2, warmup=1, batch_size=2)
    assert result["device"] == "cpu"
    assert result["latency_p50_seconds"] >= 0
    assert result["peak_allocated_bytes"] == 0


def test_cpu_model_benchmark():
    model = MLP(4, [3], 2, dropout_rate=0.0, use_batch_norm=False)
    result = benchmark_model(model, torch.randn(2, 4), repeat=2, warmup=1)
    assert result["samples_per_second"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.no_grad()
def test_cuda_benchmark_schema_when_available():
    result = benchmark_callable(lambda: torch.ones(2, device="cuda"), device="cuda", repeat=2)
    assert result["peak_allocated_bytes"] >= 0
