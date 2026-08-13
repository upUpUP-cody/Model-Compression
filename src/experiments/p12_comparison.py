"""Validation-only CPU comparison arms for the P1.2 MNIST study."""
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.autonomous_search import AutonomousSearch, full_evaluate
from src.controller.heuristic_controller import HeuristicController
from src.evaluation.frontier import ParetoFrontier
from src.pruning.sensitivity import SensitivityAnalyzer
from src.pruning.structured_pruning import StructuredPruning
from src.recovery.reconstruction import quick_recovery
from src.utils.experiment_artifacts import set_seed, to_json_safe

METHOD_NAMES = (
    "dense_baseline",
    "dense_small",
    "oneshot_magnitude",
    "oneshot_wanda",
    "iterative_structured_level1",
    "autonomous_search",
)


@dataclass
class ComparisonResult:
    """A serializable validation-only outcome for one comparison arm."""

    method: str
    status: str
    target_compression_ratio: float
    baseline_parameter_count: int
    parameter_count: int
    validation: Dict[str, float]
    selection_seconds: float
    recovery_seconds: float = 0.0
    details: Dict[str, Any] | None = None

    @property
    def compression_ratio(self) -> float:
        return self.baseline_parameter_count / max(1, self.parameter_count)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe({
            "method": self.method,
            "status": self.status,
            "target_compression_ratio": self.target_compression_ratio,
            "baseline_parameter_count": self.baseline_parameter_count,
            "parameter_count": self.parameter_count,
            "compression_ratio": self.compression_ratio,
            "validation": self.validation,
            "selection_seconds": self.selection_seconds,
            "recovery_seconds": self.recovery_seconds,
            "details": self.details or {},
        })


def parameter_count(model: nn.Module) -> int:
    """Return the physical parameter count of a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_validation(model: nn.Module, validation_loader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate a trial model exclusively against its validation loader."""
    return full_evaluate(model, validation_loader, device)


def run_comparison(
    baseline_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
) -> tuple[Dict[str, ComparisonResult], Dict[str, nn.Module]]:
    """Run all configured comparison arms without accepting a test loader."""
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


def run_method(
    method: str,
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    baseline_parameter_count: int | None = None,
) -> tuple[ComparisonResult, nn.Module]:
    """Run one method from an isolated source model and validation protocol."""
    if method not in METHOD_NAMES:
        raise ValueError(f"unknown comparison method: {method}")
    baseline_parameter_count = baseline_parameter_count or parameter_count(source_model)
    device = str(config["hardware"]["device"])
    comparison = config.get("comparison", {})
    target_ratio = float(comparison.get("target_compression_ratio", 1.0))
    start = time.perf_counter()

    if method == "dense_baseline":
        model = copy.deepcopy(source_model)
        details: Dict[str, Any] = {}
        recovery_seconds = 0.0
    elif method == "dense_small":
        model = _build_dense_small(source_model, comparison, device)
        details = {"hidden_dims": list(getattr(model, "hidden_dims", ())) }
        recovery_seconds = 0.0
    elif method in {"oneshot_magnitude", "oneshot_wanda"}:
        importance_method = "magnitude" if method == "oneshot_magnitude" else "wanda"
        model, details = _oneshot_structured(
            source_model,
            train_loader,
            comparison,
            importance_method,
            device,
        )
        recovery_seconds = 0.0
    elif method == "iterative_structured_level1":
        model, details, recovery_seconds = _iterative_structured(
            source_model, train_loader, validation_loader, comparison, device
        )
    else:
        model, details, recovery_seconds = _autonomous(
            source_model, train_loader, validation_loader, config, baseline_parameter_count
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


def _build_dense_small(source_model: nn.Module, comparison: Mapping[str, Any], device: str) -> nn.Module:
    hidden_dims = comparison.get("dense_small_hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims or any(int(width) < 1 for width in hidden_dims):
        raise ValueError("comparison.dense_small_hidden_dims must be a non-empty positive-width list")
    model_type = type(source_model)
    try:
        model = model_type(
            input_dim=source_model.input_dim,
            hidden_dims=[int(width) for width in hidden_dims],
            num_classes=source_model.num_classes,
            dropout_rate=_dropout_rate(source_model),
            use_batch_norm=source_model.use_batch_norm,
        )
    except AttributeError as error:
        raise ValueError("dense_small requires an MLP-compatible source model") from error
    return model.to(device)


def _dropout_rate(model: nn.Module) -> float:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            return float(module.p)
    return 0.0


def _oneshot_structured(
    source_model: nn.Module,
    train_loader: DataLoader,
    comparison: Mapping[str, Any],
    importance_method: str,
    device: str,
) -> tuple[nn.Module, Dict[str, Any]]:
    ratios = _layer_ratios(source_model, comparison.get("oneshot_layer_ratios", {}))
    pruner = StructuredPruning(source_model)
    keep_indices = _keep_indices(pruner, source_model, train_loader, ratios, importance_method, device, comparison)
    model = pruner.create_pruned_model_by_indices(keep_indices)
    return model, {"importance_method": importance_method, "layer_keep_indices": keep_indices}


def _iterative_structured(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    comparison: Mapping[str, Any],
    device: str,
) -> tuple[nn.Module, Dict[str, Any], float]:
    stage_ratios = comparison.get("iterative_stage_ratios")
    if not isinstance(stage_ratios, list) or not stage_ratios:
        raise ValueError("comparison.iterative_stage_ratios must be a non-empty list")
    model = copy.deepcopy(source_model)
    recovery_seconds = 0.0
    stages = []
    for stage, ratio_config in enumerate(stage_ratios, start=1):
        ratios = _layer_ratios(model, ratio_config)
        model = StructuredPruning(model).create_pruned_model(ratios)
        start = time.perf_counter()
        model, history = quick_recovery(
            model,
            train_loader,
            validation_loader,
            epochs=int(comparison.get("recovery_epochs", 0)),
            learning_rate=float(comparison.get("recovery_learning_rate", 0.001)),
            device=device,
            verbose=False,
        )
        recovery_seconds += time.perf_counter() - start
        stages.append({"stage": stage, "layer_ratios": ratios, "recovery": history})
    return model, {"stages": stages}, recovery_seconds


def _autonomous(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    baseline_parameter_count: int,
) -> tuple[nn.Module, Dict[str, Any], float]:
    search_config = config["search"]
    recovery_config = config["recovery"]
    controller = HeuristicController(**dict(config["controller"]))
    frontier = ParetoFrontier()
    search = AutonomousSearch(controller=controller)
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
        enable_two_layer_candidates=bool(search_config.get("enable_two_layer_candidates", False)),
        recovery_top_k=int(search_config.get("recovery_top_k", 1)),
        frontier_archive=frontier,
    )
    return model, {
        "history": history.to_dict(),
        "frontier": frontier.to_dict(),
        "baseline_parameter_count": baseline_parameter_count,
    }, time.perf_counter() - start


def _layer_ratios(model: nn.Module, configured_ratios: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(configured_ratios, Mapping) or not configured_ratios:
        raise ValueError("structured pruning ratios must be a non-empty mapping")
    known_layers = set(StructuredPruning(model)._hidden_linear_layer_names())
    ratios = {str(name): float(value) for name, value in configured_ratios.items()}
    if set(ratios).difference(known_layers):
        raise ValueError("structured pruning ratios contain unknown hidden layers")
    if any(not 0.0 <= ratio < 1.0 for ratio in ratios.values()):
        raise ValueError("structured pruning ratios must be in [0.0, 1.0)")
    return ratios


def _keep_indices(
    pruner: StructuredPruning,
    model: nn.Module,
    train_loader: DataLoader,
    ratios: Mapping[str, float],
    importance_method: str,
    device: str,
    comparison: Mapping[str, Any],
) -> Dict[str, List[int]]:
    if importance_method == "wanda":
        scores = SensitivityAnalyzer(model, device=device).compute_wanda_importance(
            train_loader, num_batches=int(comparison.get("wanda_batches", 1))
        )
    else:
        scores = {
            name: torch.norm(pruner._validate_hidden_linear(name).weight.detach(), p=2, dim=1)
            for name in ratios
        }
    return {
        name: pruner.prune_mlp_by_ratio(name, ratio, scores[name])
        for name, ratio in ratios.items()
    }


def results_to_records(results: Mapping[str, ComparisonResult] | Iterable[ComparisonResult]) -> List[Dict[str, Any]]:
    """Convert result mappings or sequences to JSON-ready artifact records."""
    values = results.values() if isinstance(results, Mapping) else results
    return [result.to_dict() for result in values]
