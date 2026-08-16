"""Phase K5 Qwen comparison: dense / oneshot / iterative / autonomous_search."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.controller.heuristic_controller import HeuristicController
from src.pruning.pruning_backend import resolve_pruning_backend
from src.pruning.transformer_structured_pruning import TransformerStructuredPruning
from src.recovery.qwen_lm_recovery import (
    QwenLmCheapCritic,
    evaluate_lm_loss,
    run_configured_recovery,
)
from src.utils.device import resolve_device
from src.utils.qwen_squad_eval import evaluate_squad_split
from src.utils.squad_protocol import assert_test_not_in_selection_path


METHOD_NAMES = ("dense", "oneshot", "iterative_level1", "autonomous_search")


@dataclass
class QwenMethodResult:
    method: str
    parameter_count: int
    compression_ratio: float
    val_loss: float
    val_proxy_accuracy: float
    f1: Optional[float]
    exact_match: Optional[float]
    elapsed_sec: float
    details: Dict[str, Any]


def count_params(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def mlp_only_layer_names(model: nn.Module) -> List[str]:
    backend = resolve_pruning_backend(model, "qwen")
    return [name for name in backend.prunable_layer_names() if name.endswith(".mlp.intermediate")]


def magnitude_importance_mlp(
    model: nn.Module,
    dataloader: DataLoader | None = None,
    device: str = "cpu",
    backend=None,
    max_layers: int | None = 2,
) -> Dict[str, torch.Tensor]:
    """Amplitude importance on MLP intermediate dims only (K5 smoke default)."""
    del dataloader, device  # unused; magnitude is weight-only
    pruner = TransformerStructuredPruning(model)
    names = mlp_only_layer_names(model)
    if max_layers is not None:
        names = names[-int(max_layers) :]
    importance: Dict[str, torch.Tensor] = {}
    for name in names:
        kind, index = pruner._parse_name(name)
        if kind != "mlp":
            continue
        mlp = pruner.base.layers[index].mlp
        scores = torch.norm(mlp.gate_proj.weight.detach(), p=2, dim=1).float().cpu()
        importance[name] = scores
    if not importance:
        raise ValueError("no MLP intermediate layers found for magnitude importance")
    return importance


def mlp_uniform_ratios(model: nn.Module, prune_ratio: float) -> Dict[str, float]:
    ratio = float(prune_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("prune_ratio must satisfy 0 <= ratio < 1")
    return {name: ratio for name in mlp_only_layer_names(model)}


def max_mlp_prune_ratio_for_target(target_compression: float) -> float:
    """Higher targets need a wider binary-search range on MLP prune ratio."""
    target = float(target_compression)
    if target >= 3.0:
        return 0.90
    if target >= 1.9:
        return 0.75
    return 0.50


def ratios_for_stage_target(
    model: nn.Module,
    target_compression: float,
    *,
    max_ratio: float | None = None,
    baseline_parameter_count: int | None = None,
) -> Dict[str, float]:
    """Binary-search uniform MLP prune ratios for a *global* compression target.

    Compression is always ``baseline_parameter_count / pruned_params``.
    Pass the original dense parameter count so iterative stages do not compound
    (KG.5 bug: targeting 2.0x relative to an already-1.5x model overshot to ~2.8x).
    """
    dense_count = int(baseline_parameter_count or count_params(model))
    target = float(target_compression)
    if target <= 1.0:
        raise ValueError("target_compression must be > 1")
    if max_ratio is None:
        max_ratio = max_mlp_prune_ratio_for_target(target)
    current = count_params(model)
    current_comp = float(dense_count) / max(current, 1)
    if current_comp >= target - 1e-3:
        # Already at or past the global target; do not prune further.
        return mlp_uniform_ratios(model, 0.0)

    # Search on CPU copy to avoid stacking GPU graphs during binary search.
    cpu_model = copy.deepcopy(model).cpu()
    lo, hi = 0.0, float(max_ratio)
    best = mlp_uniform_ratios(cpu_model, lo)
    backend = resolve_pruning_backend(cpu_model, "qwen")
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        ratios = mlp_uniform_ratios(cpu_model, mid)
        pruned = backend.create_pruned_model(ratios)
        compression = float(dense_count) / max(count_params(pruned), 1)
        best = ratios
        del pruned
        if compression < target:
            lo = mid
        else:
            hi = mid
    del cpu_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def run_generative_eval(
    model: nn.Module,
    tokenizer,
    examples,
    config: Mapping[str, Any],
    device: str,
) -> Dict[str, float]:
    assert_test_not_in_selection_path(["validation"])
    return evaluate_squad_split(
        model,
        tokenizer,
        examples,
        device=device,
        batch_size=int(config["hardware"].get("batch_size", 2)),
        max_seq_len=int(config["dataset"].get("max_seq_len", 512)),
        max_new_tokens=int(config["model"].get("max_new_tokens", 64)),
        max_samples=config["dataset"].get("eval_max_samples"),
        split_name="validation",
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
) -> Tuple[QwenMethodResult, nn.Module]:
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
        model, details = _oneshot(
            source_model, config, device, train_loader=train_loader, validation_loader=validation_loader
        )
    elif method == "iterative_level1":
        model, details = _iterative(source_model, train_loader, validation_loader, config, device)
    else:
        model, details = _search(source_model, train_loader, validation_loader, config, device)

    proxy = evaluate_lm_loss(model, validation_loader, device=device)
    gen_metrics = None
    if bool(config.get("comparison", {}).get("run_generative_eval", True)):
        gen_metrics = run_generative_eval(model, tokenizer, val_examples, config, device)

    result = QwenMethodResult(
        method=method,
        parameter_count=count_params(model),
        compression_ratio=float(baseline) / max(count_params(model), 1),
        val_loss=float(proxy["loss"]),
        val_proxy_accuracy=float(proxy["accuracy"]),
        f1=None if gen_metrics is None else float(gen_metrics["f1"]),
        exact_match=None if gen_metrics is None else float(gen_metrics["exact_match"]),
        elapsed_sec=time.perf_counter() - started,
        details=details,
    )
    return result, model


def _oneshot(
    source_model: nn.Module,
    config: Mapping[str, Any],
    device: str,
    train_loader: DataLoader | None = None,
    validation_loader: DataLoader | None = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
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
    details: Dict[str, Any] = {
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
    apply_recovery = bool(config.get("comparison", {}).get("oneshot_recovery", False))
    if apply_recovery:
        if train_loader is None or validation_loader is None:
            raise ValueError("oneshot_recovery requires train_loader and validation_loader")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        recovery_epochs = int(
            config.get("comparison", {}).get(
                "oneshot_recovery_epochs",
                config.get("recovery", {}).get("epochs", 1),
            )
        )
        model, history = run_configured_recovery(
            model,
            train_loader,
            validation_loader,
            config,
            epochs=recovery_epochs,
            copy_model=False,
            verbose=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        details["recovery"] = history
        details["param_after_recovery"] = count_params(model)
        details["compression_vs_dense"] = float(dense_count) / max(count_params(model), 1)
    return model, details


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
        model, history = run_configured_recovery(
            model,
            train_loader,
            validation_loader,
            config,
            epochs=recovery_epochs,
            copy_model=False,
            verbose=False,
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
    """K5/K6 search: target-aligned single candidate + LM recovery + controller gate.

    Full AutonomousSearch enumerates many candidates and deep-copies the 1.5B model
    per candidate; that is too heavy for smoke. K6 keeps the single-candidate path.
    """
    from src.controller.heuristic_controller import CandidateProfile

    search_config = config["search"]
    recovery_config = config["recovery"]
    controller = HeuristicController(**dict(config["controller"]))
    controller.reset()
    max_layers = int(search_config.get("search_mlp_layers", 2))
    resolved = resolve_device(device)
    target = float(config.get("comparison", {}).get("target_compression_ratio") or 0.0)
    max_step = float(search_config.get("max_step_compression") or 0.0)
    dense_count = count_params(source_model)

    baseline = evaluate_lm_loss(source_model, validation_loader, device=device)
    parent_count = dense_count

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
        search_mode = "k6_target_aligned_single_candidate"
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
        search_mode = "k5_smoke_single_candidate"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    critic = QwenLmCheapCritic()
    critic_result = critic.evaluate(
        candidate,
        validation_loader,
        max_samples=int(search_config.get("cheap_eval_samples", 2)),
        device=device,
    )
    recovered, recovery_history = run_configured_recovery(
        candidate,
        train_loader,
        validation_loader,
        config,
        epochs=int(recovery_config["epochs"]),
        copy_model=False,
        verbose=False,
    )
    del candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    validation = evaluate_lm_loss(recovered, validation_loader, device=device)
    recovered_count = count_params(recovered)
    quality = 1.0 / (1.0 + float(validation.get("loss", critic_result.loss)))
    profile = CandidateProfile(
        fingerprint=f"{search_mode}:{layer_name}:{ratio}",
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
        "compression_vs_dense": float(dense_count) / max(count_params(model), 1),
        "critic": critic_result.to_dict(),
        "recovery": recovery_history,
        "validation": validation,
        "baseline": baseline,
        "controller_decision": decision.to_dict(),
        "accepted": accepted,
        "param_before": parent_count,
        "param_after": recovered_count,
        "search_mlp_layers": max_layers,
        "note": (
            "LLM capability gate uses LM CE proxy; full AutonomousSearch multi-candidate "
            "loop deferred for 1.5B due to deepcopy cost; vision 2pt narrative unvalidated"
        ),
    }


def results_to_records(results: Sequence[QwenMethodResult]) -> List[Dict[str, Any]]:
    return [asdict(result) for result in results]


__all__ = [
    "METHOD_NAMES",
    "QwenMethodResult",
    "count_params",
    "magnitude_importance_mlp",
    "max_mlp_prune_ratio_for_target",
    "mlp_only_layer_names",
    "mlp_uniform_ratios",
    "ratios_for_stage_target",
    "results_to_records",
    "run_method",
]
