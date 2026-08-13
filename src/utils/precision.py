"""Mixed-precision policy helpers for CPU and CUDA execution."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from src.utils.device import resolve_device


VALID_PRECISIONS = {"fp32", "fp16", "bf16"}


def resolve_precision(value: str | None, device: str | torch.device) -> str:
    """Validate a precision name against the selected device."""
    precision = str(value or "fp32").lower()
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"unsupported precision: {precision}")
    resolved = resolve_device(device)
    if precision != "fp32" and resolved.type != "cuda":
        raise ValueError(f"{precision} precision requires a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 precision is unavailable on this device")
    return precision


def autocast_context(device: str | torch.device, precision: str | None = None):
    """Return an autocast context, disabled for fp32."""
    resolved = resolve_device(device)
    selected = resolve_precision(precision, resolved)
    if selected == "fp32":
        return nullcontext()
    dtype = torch.float16 if selected == "fp16" else torch.bfloat16
    return torch.autocast(device_type=resolved.type, dtype=dtype)


def grad_scaler(device: str | torch.device, precision: str | None = None):
    """Create a scaler only for CUDA fp16; bf16 and fp32 do not need scaling."""
    resolved = resolve_device(device)
    selected = resolve_precision(precision, resolved)
    if selected != "fp16":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def precision_metadata(device: str | torch.device, precision: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe description of the precision policy."""
    resolved = resolve_device(device)
    selected = resolve_precision(precision, resolved)
    return {
        "precision": selected,
        "autocast": selected != "fp32",
        "grad_scaler": selected == "fp16",
        "device": str(resolved),
    }
