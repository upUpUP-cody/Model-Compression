"""Derive structured-pruning ratios that approximate a target compression ratio."""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from src.experiments.model_factory import build_model_from_config, model_type_from_config
from src.pruning.pruning_backend import resolve_pruning_backend


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def derive_uniform_layer_ratios(model_config: Mapping[str, Any], target_compression_ratio: float) -> Dict[str, float]:
    """Binary-search a uniform prune ratio for all prunable layers."""
    if target_compression_ratio <= 1.0:
        raise ValueError("target_compression_ratio must be greater than 1.0")
    baseline = build_model_from_config({"model": dict(model_config)})
    return _derive_uniform_ratios_on_model(baseline, model_type_from_config({"model": model_config}), target_compression_ratio)


def derive_uniform_prune_ratio(
    model,
    model_type: str,
    target_compression_ratio: float,
) -> float:
    """Return a single uniform prune ratio that approximates target compression on ``model``."""
    ratios = _derive_uniform_ratios_on_model(model, model_type, target_compression_ratio)
    return float(max(ratios.values())) if ratios else 0.0


def _derive_uniform_ratios_on_model(model, model_type: str, target_compression_ratio: float) -> Dict[str, float]:
    if target_compression_ratio <= 1.0:
        raise ValueError("target_compression_ratio must be greater than 1.0")
    baseline_count = parameter_count(model)
    target_count = baseline_count / float(target_compression_ratio)
    backend = resolve_pruning_backend(model, model_type)
    layer_names = backend.prunable_layer_names()
    if not layer_names:
        raise ValueError("model has no prunable layers")

    low, high = 0.0, 0.95
    best = {name: 0.0 for name in layer_names}
    for _ in range(32):
        mid = (low + high) / 2.0
        ratios = {name: mid for name in layer_names}
        pruned = backend.create_pruned_model(ratios)
        count = parameter_count(pruned)
        if count > target_count:
            low = mid
        else:
            high = mid
            best = ratios
    return best


def apply_compression_target(config: Mapping[str, Any], target_compression_ratio: float) -> Dict[str, Any]:
    """Return a config copy with comparison ratios adjusted for one compression target."""
    updated = copy.deepcopy(dict(config))
    comparison = updated.setdefault("comparison", {})
    comparison["target_compression_ratio"] = float(target_compression_ratio)
    ratios = derive_uniform_layer_ratios(updated["model"], target_compression_ratio)
    comparison["oneshot_layer_ratios"] = dict(ratios)
    comparison["iterative_stage_ratios"] = [dict(ratios)]
    comparison["layer_ratios_derived_for"] = float(target_compression_ratio)
    max_ratio = max(ratios.values())
    search = updated.setdefault("search", {})
    search["candidate_ratios"] = [max_ratio]
    return updated


def ensure_compression_target(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive uniform layer ratios from comparison.target_compression_ratio when needed."""
    updated = copy.deepcopy(dict(config))
    comparison = updated.get("comparison") or {}
    target = float(comparison.get("target_compression_ratio", 1.0))
    if target <= 1.0:
        return updated
    derived_for = comparison.get("layer_ratios_derived_for")
    if derived_for is not None and abs(float(derived_for) - target) < 1e-9:
        return updated
    return apply_compression_target(updated, target)
