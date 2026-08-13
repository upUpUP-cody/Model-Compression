"""Synchronized inference benchmarks for CPU and CUDA devices."""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Iterable

import torch

from src.utils.device import resolve_device


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_callable(
    callable_fn: Callable[[], Any],
    *,
    device: str | torch.device = "cpu",
    warmup: int = 2,
    repeat: int = 10,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Benchmark a callable with CUDA synchronization and memory accounting."""
    if warmup < 0 or repeat <= 0 or batch_size <= 0:
        raise ValueError("warmup must be non-negative; repeat and batch_size must be positive")
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        for _ in range(warmup):
            callable_fn()
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)
        latencies = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callable_fn()
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end) / 1000.0)
        torch.cuda.synchronize(resolved)
        allocated = torch.cuda.max_memory_allocated(resolved)
        reserved = torch.cuda.max_memory_reserved(resolved)
    else:
        for _ in range(warmup):
            callable_fn()
        latencies = []
        for _ in range(repeat):
            started = time.perf_counter()
            callable_fn()
            latencies.append(time.perf_counter() - started)
        allocated = reserved = 0
    mean = statistics.fmean(latencies)
    return {
        "device": str(resolved),
        "warmup": warmup,
        "repeat": repeat,
        "batch_size": batch_size,
        "latency_mean_seconds": mean,
        "latency_p50_seconds": _percentile(latencies, 50),
        "latency_p95_seconds": _percentile(latencies, 95),
        "samples_per_second": batch_size / max(mean, 1e-12),
        "peak_allocated_bytes": int(allocated),
        "peak_reserved_bytes": int(reserved),
    }


def benchmark_model(model: torch.nn.Module, data: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
    """Benchmark a fixed model input."""
    device = kwargs.pop("device", next(model.parameters()).device)
    resolved = resolve_device(device)
    model = model.to(resolved).eval()
    data = data.to(resolved)
    with torch.inference_mode():
        return benchmark_callable(lambda: model(data), device=resolved, batch_size=int(data.shape[0]), **kwargs)
