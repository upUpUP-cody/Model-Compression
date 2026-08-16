"""Phase KG Qwen+GLUE comparison: dense / oneshot / iterative / autonomous_search."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.controller.heuristic_controller import HeuristicController
from src.experiments.qwen_k5_comparison import (
    METHOD_NAMES,
    count_params,
    magnitude_importance_mlp,
    mlp_uniform_ratios,
    ratios_for_stage_target,
)
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.qwen_lm_recovery import QwenLmCheapCritic, evaluate_lm_loss, quick_lm_recovery
from src.utils.device import resolve_device
from src.utils.glue_protocol import assert_test_not_in_selection_path
from src.utils.qwen_glue_eval import evaluate_glue_split


@dataclass
class GlueMethodResult:
    method: str
    parameter_count: int
    compression_ratio: float
    val_loss: float
    val_proxy_accuracy: float
    accuracy: Optional[float]
    elapsed_sec: float
    details: Dict[str, Any]


def run_task_eval(
    model: nn.Module,
    tokenizer,
    examples,
    config: Mapping[str, Any],
    device: str,
) -> Dict[str, float]:
    assert_test_not_in_selection_path(["validation"])
    return evaluate_glue_split(
        model,
        tokenizer,
        examples,
        device=device,
        batch_size=int(config["hardware"].get("batch_size", 1)),
        max_seq_len=int(config["dataset"].get("max_seq_len", 128)),
        max_new_tokens=int(config["model"].get("max_new_tokens", 8)),
        max_samples=config["dataset"].get("eval_max_samples"),
        split_name="validation",
        task=str(config["dataset"].get("task", "sst2")),
    )


def run_method(
    method: str,
    source_model: nn.Module,
    tokenizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    val_examples,
    config: Mapping[str, Any],
    baseline_parameter_count: Optional[int] = None,
) -> Tuple[GlueMethodResult, nn.Module]:
    if method not in METHOD_NAMES:
        raise ValueError(f"unsupported method: {method}")
    device = str(config["hardware"]["device"])
    baseline = int(baseline_parameter_count or count_params(source_model))
    started = time.perf_counter()
    details: Dict[str, Any] = {}

    if method == "dense":
        model = source_model
        details["applied"] = False
    elif method == "oneshot":
        model, details = _oneshot(source_model, config, device)
    elif method == "iterative_level1":
        model, details = _iterative(source_model, train_loader, validation_loader, config, device)
    else:
        model, details = _search(source_model, train_loader, validation_loader, config, device)

    proxy = evaluate_lm_loss(model, validation_loader, device=device)
    task_metrics = None
    run_eval = bool(
        config.get("comparison", {}).get(
            "run_task_eval",
            config.get("comparison", {}).get("run_generative_eval", True),
        )
    )
    if run_eval:
        task_metrics = run_task_eval(model, tokenizer, val_examples, config, device)

    result = GlueMethodResult(
        method=method,
        parameter_count=count_params(model),
        compression_ratio=float(baseline) / max(count_params(model), 1),
        val_loss=float(proxy["loss"]),
        val_proxy_accuracy=float(proxy["accuracy"]),
        accuracy=None if task_metrics is None else float(task_metrics["accuracy"]),
        elapsed_sec=time.perf_counter() - started,
        details=details,
    )
    return result, model


def _oneshot(source_model: nn.Module, config: Mapping[str, Any], device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    # Prefer target-aligned ratios (KG.5/K6); fall back to fixed oneshot_mlp_ratio for smoke.
    target = float(config.get("comparison", {}).get("target_compression_ratio") or 0.0)
    backend = resolve_pruning_backend(source_model, "qwen")
    dense_count = count_params(source_model)
    if target > 1.0:
        ratios = ratios_for_stage_target(
            source_model,
            target,
            baseline_parameter_count=dense_count,
        )
        ratio_source = "target_compression_ratio"
    else:
        prune_ratio = float(config.get("pruning", {}).get("oneshot_mlp_ratio", 0.25))
        ratios = mlp_uniform_ratios(source_model, prune_ratio)
        ratio_source = "oneshot_mlp_ratio"
    before = dense_count
    model = backend.create_pruned_model(ratios).to(resolve_device(device))
    after = count_params(model)
    return model, {
        "applied": True,
        "param_before": before,
        "param_after": after,
        "compression_ratio": float(before) / max(after, 1),
        "compression_vs_dense": float(dense_count) / max(after, 1),
        "layer_ratios": ratios,
        "ratio_source": ratio_source,
        "target_compression_ratio": target if target > 1.0 else None,
        "importance_method": "magnitude",
        "recovery": "none",
    }


def _iterative(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    comparison = config.get("comparison", {})
    stage_targets = comparison.get("iterative_stage_targets")
    if not isinstance(stage_targets, list) or not stage_targets:
        raise ValueError("comparison.iterative_stage_targets must be a non-empty list")
    recovery_epochs = int(comparison.get("iterative_recovery_epochs", config.get("recovery", {}).get("epochs", 1)))
    recovery_lr = float(config.get("recovery", {}).get("learning_rate", 2e-5))
    precision = str(config.get("hardware", {}).get("precision", "fp16"))
    dense_count = count_params(source_model)
    model = copy.deepcopy(source_model).to(resolve_device(device))
    stages = []
    for stage, target in enumerate(stage_targets, start=1):
        target_ratio = float(target)
        ratios = ratios_for_stage_target(
            model,
            target_ratio,
            baseline_parameter_count=dense_count,
        )
        backend = resolve_pruning_backend(model, "qwen")
        before = count_params(model)
        model = backend.create_pruned_model(ratios).to(resolve_device(device))
        after = count_params(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, history = quick_lm_recovery(
            model,
            train_loader,
            validation_loader,
            epochs=recovery_epochs,
            learning_rate=recovery_lr,
            device=device,
            precision=precision,
            verbose=False,
            copy_model=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stages.append(
            {
                "stage": stage,
                "target_compression": target_ratio,
                "layer_ratios": ratios,
                "param_before": before,
                "param_after": after,
                "compression_ratio": float(before) / max(after, 1),
                "compression_vs_dense": float(dense_count) / max(after, 1),
                "recovery": history,
            }
        )
    return model, {
        "stages": stages,
        "importance_method": "magnitude",
        "dense_parameter_count": dense_count,
        "final_compression_vs_dense": float(dense_count) / max(count_params(model), 1),
    }


def _search(
    source_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    from src.controller.heuristic_controller import CandidateProfile

    search_config = config["search"]
    recovery_config = config["recovery"]
    controller = HeuristicController(**dict(config["controller"]))
    controller.reset()
    max_layers = int(search_config.get("search_mlp_layers", 2))
    resolved = resolve_device(device)
    target = float(config.get("comparison", {}).get("target_compression_ratio") or 0.0)
    max_step = float(search_config.get("max_step_compression") or 0.0)

    baseline = evaluate_lm_loss(source_model, validation_loader, device=device)
    parent_count = count_params(source_model)
    dense_count = parent_count

    # Align single-candidate prune with compression target when provided (KG.5/K6).
    if target > 1.0:
        step_target = min(target, max_step) if max_step > 1.0 else target
        ratios = ratios_for_stage_target(
            source_model,
            step_target,
            baseline_parameter_count=dense_count,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_parent = copy.deepcopy(source_model).cpu()
        candidate = (
            resolve_pruning_backend(cpu_parent, "qwen")
            .create_pruned_model(ratios)
            .to(resolved)
        )
        del cpu_parent
        layer_name = next(iter(ratios.keys())) if ratios else "mlp"
        ratio = float(next(iter(ratios.values()))) if ratios else 0.0
        search_mode = "kg5_target_aligned_single_candidate"
    else:
        ratio = float(list(search_config.get("candidate_ratios") or [0.25])[0])
        importance = magnitude_importance_mlp(source_model, max_layers=max_layers)
        layer_name = next(iter(importance.keys()))
        scores = importance[layer_name]
        output_size = int(scores.numel())
        keep_count = max(1, int(round(output_size * (1.0 - ratio))))
        ranked = sorted(range(output_size), key=lambda i: (-float(scores[i]), i))
        keep_indices = {layer_name: sorted(ranked[:keep_count])}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_parent = copy.deepcopy(source_model).cpu()
        candidate = (
            resolve_pruning_backend(cpu_parent, "qwen")
            .create_pruned_model_by_indices(keep_indices)
            .to(resolved)
        )
        del cpu_parent
        search_mode = "kg_smoke_single_candidate"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    critic = QwenLmCheapCritic()
    critic_result = critic.evaluate(
        candidate,
        validation_loader,
        max_samples=int(search_config.get("cheap_eval_samples", 2)),
        device=device,
    )
    recovered, recovery_history = quick_lm_recovery(
        candidate,
        train_loader,
        validation_loader,
        epochs=int(recovery_config["epochs"]),
        learning_rate=float(recovery_config["learning_rate"]),
        device=device,
        precision=str(config["hardware"].get("precision", "fp16")),
        verbose=False,
        copy_model=False,
    )
    del candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    validation = evaluate_lm_loss(recovered, validation_loader, device=device)
    recovered_count = count_params(recovered)
    quality = 1.0 / (1.0 + float(validation.get("loss", critic_result.loss)))
    profile = CandidateProfile(
        fingerprint=f"glue:{search_mode}:{layer_name}:{ratio}",
        quality=quality,
        accuracy=float(validation["accuracy"]),
        parent_accuracy=float(baseline["accuracy"]),
        parameter_count=recovered_count,
        parent_parameter_count=parent_count,
    )
    decision = controller.decide_action(profile)
    accepted = decision.action == "accept"
    model = recovered if accepted else source_model
    return model, {
        "mode": search_mode,
        "layer_name": layer_name,
        "prune_ratio": ratio,
        "target_compression_ratio": target if target > 1.0 else None,
        "critic": critic_result.to_dict(),
        "recovery": recovery_history,
        "validation": validation,
        "baseline": baseline,
        "controller_decision": decision.to_dict(),
        "accepted": accepted,
        "param_before": parent_count,
        "param_after": recovered_count,
        "search_mlp_layers": max_layers,
        "note": "GLUE single-candidate search; LM CE proxy gate; multi-candidate deferred",
    }


def results_to_records(results: Sequence[GlueMethodResult]) -> List[Dict[str, Any]]:
    return [asdict(result) for result in results]


__all__ = [
    "METHOD_NAMES",
    "GlueMethodResult",
    "count_params",
    "run_method",
    "results_to_records",
]
