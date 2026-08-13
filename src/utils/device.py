"""Device resolution and CUDA reproducibility helpers."""
from __future__ import annotations

from typing import Any, Mapping

import torch


def resolve_device(value: str | torch.device, *, require_available: bool = True) -> torch.device:
    """Resolve and validate a configured device without silent fallback."""
    device = torch.device(value)
    if device.type == "cuda":
        if require_available and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        count = torch.cuda.device_count()
        index = 0 if device.index is None else device.index
        if index < 0:
            raise RuntimeError(f"CUDA device index {index} is invalid")
        if require_available and index >= count:
            raise RuntimeError(f"CUDA device index {index} is unavailable; count={count}")
        return torch.device("cuda", index)
    if device.type not in {"cpu", "mps"}:
        raise ValueError(f"unsupported device type: {device.type}")
    if device.type == "mps" and require_available and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but MPS is unavailable")
    return device


def move_to_device(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    """Move a model to a validated device and return it."""
    return model.to(resolve_device(device))


def configure_cuda(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply deterministic, cuDNN, and TF32 settings for a configured device."""
    hardware = config.get("hardware", config)
    device = resolve_device(str(hardware.get("device", "cpu")))
    deterministic = bool(hardware.get("deterministic", True))
    benchmark = bool(hardware.get("cudnn_benchmark", False))
    tf32 = bool(hardware.get("tf32", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("seed", 0)))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = tf32
    return {
        "device": str(device),
        "deterministic": deterministic,
        "cudnn_benchmark": benchmark,
        "tf32": tf32,
    }


def device_metadata(device: str | torch.device) -> dict[str, Any]:
    """Return serializable metadata for the selected device."""
    resolved = resolve_device(device)
    metadata: dict[str, Any] = {"device": str(resolved), "type": resolved.type}
    if resolved.type == "cuda":
        props = torch.cuda.get_device_properties(resolved)
        metadata.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
        })
    else:
        metadata["cuda_available"] = torch.cuda.is_available()
    return metadata
