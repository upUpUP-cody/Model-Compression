"""Validation-only CIFAR P1.2 comparison arms using ResNet pruning backends."""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.autonomous_search import AutonomousSearch, full_evaluate
from src.controller.heuristic_controller import HeuristicController
from src.evaluation.frontier import ParetoFrontier
from src.experiments.model_factory import build_dense_small_model, model_type_from_config
from src.experiments.p12_comparison import (
    METHOD_NAMES,
    ComparisonResult,
    evaluate_validation,
    parameter_count,
    results_to_records,
)
from src.pruning.pruning_backend import resolve_pruning_backend
from src.pruning.sensitivity import SensitivityAnalyzer
from src.recovery.recovery_dispatch import run_recovery
from src.utils.device import resolve_device
from src.utils.experiment_artifacts import set_seed, to_json_safe


def run_comparison(
    baseline_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
) -> tuple[Dict[str, ComparisonResult], Dict[str, nn.Module]]:
    return _run_comparison_impl(baseline_model, train_loader, validation_loader, config)


def run_method(
    method: str,
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    baseline_parameter_count: int | None = None,
) -> tuple[ComparisonResult, nn.Module]:
    if method not in METHOD_NAMES:
        raise ValueError(f"unknown comparison method: {method}")
    baseline_parameter_count = baseline_parameter_count or parameter_count(source_model)
    device = str(config["hardware"]["device"])
    comparison = config.get("comparison", {})
    target_ratio = float(comparison.get("target_compression_ratio", 1.0))
    model_type = model_type_from_config(config)
    backend = resolve_pruning_backend(source_model, model_type)
    start = time.perf_counter()

    if method == "dense_baseline":
        model = copy.deepcopy(source_model)
        details: Dict[str, Any] = {}
        recovery_seconds = 0.0
    elif method == "dense_small":
        model, details, recovery_seconds = _train_dense_small(
            source_model, train_loader, validation_loader, config, device
        )
    elif method in {"oneshot_magnitude", "oneshot_wanda"}:
        importance_method = "magnitude" if method == "oneshot_magnitude" else "wanda"
        model, details = _oneshot_structured(
            source_model, backend, train_loader, comparison, importance_method, device, model_type
        )
        recovery_seconds = 0.0
    elif method == "iterative_structured_level1":
        model, details, recovery_seconds = _iterative_structured(
            source_model, backend, train_loader, validation_loader, config, device, model_type
        )
    else:
        model, details, recovery_seconds = _autonomous(
            source_model, train_loader, validation_loader, config, baseline_parameter_count, model_type
        )

    validation = evaluate_validation(model, validation_loader, device)
    selection_seconds = time.perf_counter() - start
    return ComparisonResult(
        method=method,
        status="completed",
        target_compression_ratio=target_ratio,
        baseline_parameter_count=baseline_parameter_count,
        parameter_count=parameter_count(model),
        validation=validation,
        selection_seconds=selection_seconds,
        recovery_seconds=recovery_seconds,
        details=details,
    ), model


def _run_comparison_impl(
    baseline_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
) -> tuple[Dict[str, ComparisonResult], Dict[str, nn.Module]]:
    baseline_count = parameter_count(baseline_model)
    methods = config.get("comparison", {}).get("methods", list(METHOD_NAMES))
    unknown = set(methods).difference(METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown comparison methods: {sorted(unknown)}")

    results: Dict[str, ComparisonResult] = {}
    models: Dict[str, nn.Module] = {}
    for offset, method in enumerate(methods):
        set_seed(int(config["seed"]) + offset)
        result, model = run_method(
            method,
            copy.deepcopy(baseline_model),
            train_loader,
            validation_loader,
            config,
            baseline_count,
        )
        results[method] = result
        models[method] = model
    return results, models


def _train_dense_small(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    device: str,
) -> tuple[nn.Module, Dict[str, Any], float]:
    comparison = config.get("comparison", {})
    model = build_dense_small_model(config, device)
    epochs = int(comparison.get("recovery_epochs", 0))
    learning_rate = float(comparison.get("recovery_learning_rate", 0.001))
    start = time.perf_counter()
    history: Dict[str, Any] = {}
    if epochs > 0:
        model, history = run_recovery(
            model,
            train_loader,
            validation_loader,
            config,
            teacher_model=source_model,
        )
    recovery_seconds = time.perf_counter() - start
    details = {"from_scratch": True, "training_epochs": epochs, "training": history}
    if hasattr(model, "base_width"):
        details["base_width"] = int(model.base_width)
    if hasattr(model, "hidden_dims"):
        details["hidden_dims"] = list(model.hidden_dims)
    return model, details, recovery_seconds


def _layer_ratios(model: nn.Module, configured_ratios: Mapping[str, Any], model_type: str) -> Dict[str, float]:
    if not isinstance(configured_ratios, Mapping) or not configured_ratios:
        raise ValueError("structured pruning ratios must be a non-empty mapping")
    known_layers = set(resolve_pruning_backend(model, model_type).prunable_layer_names())
    ratios = {str(name): float(value) for name, value in configured_ratios.items()}
    if set(ratios).difference(known_layers):
        raise ValueError("structured pruning ratios contain unknown prunable layers")
    if any(not 0.0 <= ratio < 1.0 for ratio in ratios.values()):
        raise ValueError("structured pruning ratios must be in [0.0, 1.0)")
    return ratios


def _keep_indices(
    backend,
    model: nn.Module,
    train_loader: DataLoader,
    ratios: Mapping[str, float],
    importance_method: str,
    device: str,
    comparison: Mapping[str, Any],
    model_type: str,
) -> Dict[str, list[int]]:
    if importance_method == "wanda":
        analyzer = SensitivityAnalyzer(model, device=device)
        if model_type == "mlp":
            scores = analyzer.compute_wanda_importance(
                train_loader, num_batches=int(comparison.get("wanda_batches", 1))
            )
        else:
            layer_names = backend.prunable_layer_names()
            scores = analyzer.compute_wanda_importance(
                train_loader,
                num_batches=int(comparison.get("wanda_batches", 1)),
                layer_names=layer_names,
            )
    else:
        scores = {}
        for name in ratios:
            layer = resolve_pruning_backend(model, model_type)
            if name not in layer.prunable_layer_names():
                continue
            module = model
            for part in name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            if isinstance(module, nn.Conv2d):
                scores[name] = torch.norm(module.weight.detach(), p=2, dim=(1, 2, 3))
            elif isinstance(module, nn.Linear):
                scores[name] = torch.norm(module.weight.detach(), p=2, dim=1)
            else:
                raise ValueError(f"unsupported layer type for magnitude: {name}")
    keep_indices: Dict[str, list[int]] = {}
    for name, ratio in ratios.items():
        output_size = backend.output_size(name)
        scores_tensor = scores[name].detach().reshape(-1)
        keep_count = max(1, int(output_size * (1.0 - float(ratio))))
        ranked = sorted(
            range(output_size),
            key=lambda index: (-float(scores_tensor[index]), index),
        )
        keep_indices[name] = sorted(ranked[:keep_count])
    return keep_indices


def _oneshot_structured(
    source_model: nn.Module,
    backend,
    train_loader: DataLoader,
    comparison: Mapping[str, Any],
    importance_method: str,
    device: str,
    model_type: str,
) -> tuple[nn.Module, Dict[str, Any]]:
    ratios = _layer_ratios(source_model, comparison.get("oneshot_layer_ratios", {}), model_type)
    keep_indices = _keep_indices(
        backend, source_model, train_loader, ratios, importance_method, device, comparison, model_type
    )
    model = backend.create_pruned_model_by_indices(keep_indices).to(resolve_device(device))
    return model, {"importance_method": importance_method, "layer_keep_indices": keep_indices}


def _iterative_structured(
    source_model: nn.Module,
    backend,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    device: str,
    model_type: str,
) -> tuple[nn.Module, Dict[str, Any], float]:
    comparison = config.get("comparison", {})
    stage_ratios = comparison.get("iterative_stage_ratios")
    if not isinstance(stage_ratios, list) or not stage_ratios:
        raise ValueError("comparison.iterative_stage_ratios must be a non-empty list")
    model = copy.deepcopy(source_model)
    recovery_seconds = 0.0
    stages = []
    iterative_epochs = comparison.get("iterative_recovery_epochs")
    recovery_config = dict(config.get("recovery") or {})
    if iterative_epochs is not None:
        recovery_config["epochs"] = int(iterative_epochs)
    stage_config = dict(config)
    stage_config["recovery"] = recovery_config
    for stage, ratio_config in enumerate(stage_ratios, start=1):
        current_backend = resolve_pruning_backend(model, model_type)
        ratios = _layer_ratios(model, ratio_config, model_type)
        keep_indices = _keep_indices(
            current_backend, model, train_loader, ratios, "magnitude", device, comparison, model_type
        )
        model = current_backend.create_pruned_model_by_indices(keep_indices).to(resolve_device(device))
        start = time.perf_counter()
        model, history = run_recovery(
            model, train_loader, validation_loader, stage_config, teacher_model=source_model
        )
        recovery_seconds += time.perf_counter() - start
        stages.append({
            "stage": stage,
            "layer_ratios": ratios,
            "layer_keep_indices": keep_indices,
            "recovery": history,
            "recovery_epochs": int(recovery_config.get("epochs", 0)),
        })
    merged_keep_indices: Dict[str, list[int]] = {}
    for stage_record in stages:
        merged_keep_indices.update(stage_record["layer_keep_indices"])
    return model, {"stages": stages, "layer_keep_indices": merged_keep_indices}, recovery_seconds


def _autonomous(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    baseline_parameter_count: int,
    model_type: str,
) -> tuple[nn.Module, Dict[str, Any], float]:
    search_config = config["search"]
    recovery_config = config["recovery"]
    controller = HeuristicController(**dict(config["controller"]))
    frontier = ParetoFrontier()
    search = AutonomousSearch(controller=controller, model_type=model_type)
    start = time.perf_counter()
    model, history = search.run(
        source_model,
        train_loader,
        validation_loader,
        max_iterations=int(search_config["max_iterations"]),
        candidate_ratios=list(search_config["candidate_ratios"]),
        candidates_per_round=int(search_config["candidates_per_round"]),
        cheap_eval_samples=int(search_config["cheap_eval_samples"]),
        recovery_epochs=int(recovery_config["epochs"]),
        recovery_learning_rate=float(recovery_config["learning_rate"]),
        device=str(config["hardware"]["device"]),
        precision=str(config["hardware"].get("precision", "fp32")),
        enable_two_layer_candidates=bool(search_config.get("enable_two_layer_candidates", False)),
        recovery_top_k=int(search_config.get("recovery_top_k", 1)),
        frontier_archive=frontier,
        target_compression_ratio=float(config.get("comparison", {}).get("target_compression_ratio", 0.0)) or None,
        max_step_compression=float(search_config.get("max_step_compression", 1.75)),
    )
    history_dict = history.to_dict()
    keep_indices: Dict[str, list[int]] = {}
    for event in reversed(history_dict.get("events") or []):
        if event.get("final_action") != "accept":
            continue
        spec = event.get("candidate_spec") or {}
        indices = spec.get("layer_keep_indices")
        if isinstance(indices, dict) and indices:
            keep_indices = {str(name): list(values) for name, values in indices.items()}
            break
    return model, {
        "history": history_dict,
        "frontier": frontier.to_dict(),
        "baseline_parameter_count": baseline_parameter_count,
        "layer_keep_indices": keep_indices,
    }, time.perf_counter() - start


__all__ = [
    "METHOD_NAMES",
    "ComparisonResult",
    "run_comparison",
    "run_method",
    "results_to_records",
]
