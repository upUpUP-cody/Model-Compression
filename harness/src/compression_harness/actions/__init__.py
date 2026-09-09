"""Action primitives (declarations for the finite action space)."""

from __future__ import annotations

from typing import Any


PRIMITIVES = (
    "profile_model",
    "quantize",
    "exclude_layers",
    "set_layer_precision",
    "calibrate",
    "serve",
    "evaluate",
    "compare",
    "diagnose",
)


def profile_model(model_ref: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "action": "profile_model",
        "model_ref": model_ref,
        "status": "dry_run",
        "profile": {"layers": [], "notes": "stub profile"},
        **kwargs,
    }


def quantize(**kwargs: Any) -> dict[str, Any]:
    return {"action": "quantize", "status": "dry_run", **kwargs}


def evaluate(**kwargs: Any) -> dict[str, Any]:
    return {"action": "evaluate", "status": "dry_run", **kwargs}


def diagnose(**kwargs: Any) -> dict[str, Any]:
    return {"action": "diagnose", "status": "dry_run", **kwargs}


def compare(run_a: str, run_b: str, **kwargs: Any) -> dict[str, Any]:
    return {"action": "compare", "run_a": run_a, "run_b": run_b, "status": "dry_run", **kwargs}
