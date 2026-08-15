"""Autonomous, CPU-oriented structured-pruning search."""
import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.controller.heuristic_controller import CandidateProfile, HeuristicController
from src.evaluation.cheap_critic import CheapCritic, CheapCriticResult
from src.evaluation.frontier import ParetoFrontier, FrontierPoint
from src.experiments.compression_targets import derive_uniform_prune_ratio
from src.pruning.pruning_backend import CnnBackend, MlpBackend, PruningBackend, resolve_pruning_backend
from src.pruning.sensitivity import SensitivityAnalyzer
from src.pruning.structured_pruning import StructuredPruning
from src.recovery.reconstruction import quick_recovery
from src.utils.device import resolve_device


@dataclass(frozen=True)
class CandidateSpec:
    """Immutable, JSON-safe description of a physically pruned candidate."""

    layer_ratios: Tuple[Tuple[str, float], ...]
    layer_keep_indices: Tuple[Tuple[str, Tuple[int, ...]], ...]
    importance_method: str
    fingerprint: str
    parameter_count: int
    parent_parameter_count: int

    @classmethod
    def create(
        cls,
        layer_ratios: Dict[str, float],
        layer_keep_indices: Dict[str, Sequence[int]],
        importance_method: str,
        parameter_count: int,
        parent_parameter_count: int,
    ) -> "CandidateSpec":
        ratios = tuple(sorted((name, float(ratio)) for name, ratio in layer_ratios.items()))
        keep_indices = tuple(
            sorted((name, tuple(sorted(int(index) for index in indices)))
                   for name, indices in layer_keep_indices.items())
        )
        payload = {
            "importance_method": importance_method,
            "layer_ratios": dict(ratios),
            "layer_keep_indices": {name: list(indices) for name, indices in keep_indices},
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(
            layer_ratios=ratios,
            layer_keep_indices=keep_indices,
            importance_method=str(importance_method),
            fingerprint=hashlib.sha256(serialized.encode("ascii")).hexdigest(),
            parameter_count=int(parameter_count),
            parent_parameter_count=int(parent_parameter_count),
        )

    @property
    def compression_ratio(self) -> float:
        if self.parameter_count == 0:
            return float("inf")
        return self.parent_parameter_count / self.parameter_count

    def ratios_dict(self) -> Dict[str, float]:
        return dict(self.layer_ratios)

    def keep_indices_dict(self) -> Dict[str, List[int]]:
        return {name: list(indices) for name, indices in self.layer_keep_indices}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_ratios": self.ratios_dict(),
            "layer_keep_indices": self.keep_indices_dict(),
            "importance_method": self.importance_method,
            "fingerprint": self.fingerprint,
            "parameter_count": self.parameter_count,
            "parent_parameter_count": self.parent_parameter_count,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class SearchHistory:
    """JSON-safe records from an autonomous pruning search."""

    baseline_accuracy: float
    accepted_parameter_count: int
    events: List[Dict[str, Any]] = field(default_factory=list)
    attempted_fingerprints: set[str] = field(default_factory=set, repr=False)
    consecutive_failures: int = 0
    ratio_multiplier: float = 1.0
    frontier_points: List[Dict[str, Any]] = field(default_factory=list)
    initial_parameter_count: int = 0

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_accuracy": float(self.baseline_accuracy),
            "accepted_parameter_count": int(self.accepted_parameter_count),
            "consecutive_failures": int(self.consecutive_failures),
            "ratio_multiplier": float(self.ratio_multiplier),
            "frontier_points": self.frontier_points,
            "initial_parameter_count": int(self.initial_parameter_count),
            "events": self.events,
        }


def candidate_fingerprint(config: Dict[str, float]) -> str:
    """Produce a deterministic fingerprint for a normalized pruning config."""
    normalized = {name: float(config[name]) for name in sorted(config)}
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def full_evaluate(
    model: nn.Module, dataloader: DataLoader, device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate all samples without changing the model's training mode."""
    critic = CheapCritic()
    result = critic.evaluate(model, dataloader, max_samples=_loader_sample_count(dataloader), device=device)
    return {"loss": result.loss, "accuracy": result.accuracy, "samples": result.samples}


class AutonomousSearch:
    """Generate, assess, recover, and selectively accept MLP pruning candidates."""

    def __init__(
        self,
        controller: HeuristicController,
        critic: Optional[CheapCritic] = None,
        recovery_fn: Callable[..., Tuple[nn.Module, Dict[str, Any]]] = quick_recovery,
        evaluator: Callable[[nn.Module, DataLoader, str], Dict[str, float]] = full_evaluate,
        importance_fn: Optional[Callable[[nn.Module, DataLoader, str], Dict[str, torch.Tensor]]] = None,
        pruning_backend: Optional[PruningBackend] = None,
        model_type: str = "mlp",
    ) -> None:
        self.controller = controller
        self.critic = critic if critic is not None else CheapCritic()
        self.recovery_fn = recovery_fn
        self.evaluator = evaluator
        self.pruning_backend = pruning_backend
        self.model_type = model_type
        self.importance_fn = importance_fn or self._default_importance_fn

    def run(
        self,
        parent_model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        max_iterations: int = 10,
        candidate_ratios: Sequence[float] = (0.3, 0.5),
        candidates_per_round: int = 5,
        cheap_eval_samples: int = 500,
        recovery_epochs: int = 1,
        recovery_learning_rate: float = 0.001,
        device: str = "cpu",
        precision: str = "fp32",
        enable_two_layer_candidates: bool = False,
        recovery_top_k: int = 1,
        frontier_archive: Optional[ParetoFrontier] = None,
        target_compression_ratio: float | None = None,
        max_step_compression: float = 1.75,
    ) -> Tuple[nn.Module, SearchHistory]:
        if max_iterations < 1 or candidates_per_round < 1 or recovery_epochs < 0:
            raise ValueError("iteration, candidate, and recovery limits must be valid")
        if recovery_top_k < 1:
            raise ValueError("recovery_top_k must be positive")
        if not candidate_ratios:
            raise ValueError("candidate_ratios must not be empty")
        if target_compression_ratio is not None and float(target_compression_ratio) <= 1.0:
            raise ValueError("target_compression_ratio must be greater than 1.0 when provided")
        if float(max_step_compression) <= 1.0:
            raise ValueError("max_step_compression must be greater than 1.0")

        resolved_device = resolve_device(device)
        current_model = copy.deepcopy(parent_model).to(resolved_device)
        self.controller.reset()
        baseline = self.evaluator(current_model, validation_loader, device)
        history = SearchHistory(
            baseline_accuracy=float(baseline["accuracy"]),
            accepted_parameter_count=_parameter_count(current_model),
            initial_parameter_count=_parameter_count(current_model),
        )
        ratio_multiplier = 1.0
        accepted_snapshot = copy.deepcopy(current_model)
        target_ratio = float(target_compression_ratio) if target_compression_ratio is not None else None
        step_cap = effective_max_step_compression(max_step_compression, target_ratio)
        max_allowed_compression = target_ratio * 1.15 if target_ratio is not None else None

        for iteration in range(max_iterations):
            history.ratio_multiplier = ratio_multiplier
            backend = self._backend_for_model(current_model)
            current_count = _parameter_count(current_model)
            current_compression = history.initial_parameter_count / max(current_count, 1)
            if target_ratio is not None and current_compression >= target_ratio:
                history.add_event({
                    "iteration": iteration,
                    "action": "stop",
                    "reason": "target_compression_reached",
                    "compression_ratio": current_compression,
                    "target_compression_ratio": target_ratio,
                })
                break

            round_ratios = self._round_candidate_ratios(
                current_model=current_model,
                configured_ratios=candidate_ratios,
                target_ratio=target_ratio,
                current_compression=current_compression,
                max_step_compression=step_cap,
            )
            if not round_ratios:
                history.add_event({"iteration": iteration, "action": "stop", "reason": "no_round_ratios"})
                break

            importance = self._call_importance_fn(current_model, train_loader, device, backend)
            candidates = self._generate_candidates(
                current_model,
                importance,
                round_ratios,
                ratio_multiplier,
                attempted=history.attempted_fingerprints,
                limit=None,
                enable_two_layer_candidates=enable_two_layer_candidates,
                backend=backend,
            )
            if not candidates:
                history.add_event({"iteration": iteration, "action": "stop", "reason": "no_new_candidates"})
                break

            audited = []
            critic_candidates = []
            cheap_evaluations = 0
            parent_parameter_count = _parameter_count(current_model)
            for order, spec in enumerate(candidates):
                record = {
                    "generation_order": order,
                    "candidate_type": "two_layer" if len(spec.layer_ratios) > 1 else "single_layer",
                    "candidate_spec": spec.to_dict(),
                    "fingerprint": spec.fingerprint,
                    "audit_status": "generated",
                }
                try:
                    candidate_model = backend.create_pruned_model_by_indices(
                        spec.keep_indices_dict()
                    ).to(resolved_device)
                    actual_count = _parameter_count(candidate_model)
                    spec = replace(
                        spec,
                        parameter_count=actual_count,
                        parent_parameter_count=parent_parameter_count,
                    )
                    record["candidate_spec"] = spec.to_dict()
                    record["actual_parameter_count"] = actual_count
                    record["parent_parameter_count"] = parent_parameter_count
                    record["parameter_reduction"] = parent_parameter_count - actual_count
                    candidate_compression = history.initial_parameter_count / max(actual_count, 1)
                    record["candidate_compression_ratio"] = candidate_compression
                    if spec.fingerprint in history.attempted_fingerprints:
                        record.update(audit_status="filtered", final_reason="duplicate_attempted_fingerprint")
                    elif actual_count >= parent_parameter_count:
                        record.update(audit_status="filtered", final_reason="no_parameter_reduction")
                    elif (
                        max_allowed_compression is not None
                        and candidate_compression > max_allowed_compression
                    ):
                        record.update(
                            audit_status="filtered",
                            final_reason="target_compression_overshoot",
                            max_allowed_compression=max_allowed_compression,
                        )
                    elif cheap_evaluations >= candidates_per_round:
                        record.update(audit_status="filtered", final_reason="cheap_critic_budget_exhausted")
                    else:
                        critic_result = self.critic.evaluate(
                            candidate_model, validation_loader, cheap_eval_samples, device
                        )
                        cheap_evaluations += 1
                        record["cheap_critic"] = critic_result.to_dict()
                        record["audit_status"] = "cheap_evaluated"
                        history.attempted_fingerprints.add(spec.fingerprint)
                        critic_candidates.append((spec, candidate_model, critic_result, record))
                except (TypeError, ValueError, RuntimeError) as error:
                    record.update(audit_status="filtered", final_reason="invalid_candidate_spec", error=str(error))
                audited.append(record)

            if not critic_candidates:
                history.add_event({"iteration": iteration, "action": "stop", "reason": "no_viable_candidates", "candidates": audited})
                break

            critic_candidates.sort(key=lambda item: (
                -(item[0].parent_parameter_count - item[0].parameter_count),
                item[2].loss,
                -item[0].compression_ratio,
                item[0].fingerprint,
            ))
            for rank, item in enumerate(critic_candidates):
                item[3]["shortlist_rank"] = rank + 1
            recovery_candidates = critic_candidates[:recovery_top_k]
            for rank, item in enumerate(recovery_candidates):
                item[3]["recovery_rank"] = rank + 1

            # Cheap Critic only ranks; capability gate uses post-recovery validation.
            recovered_shortlist = []
            for spec, candidate_model, critic_result, record in recovery_candidates:
                recovered_model, recovery_history = self.recovery_fn(
                    candidate_model,
                    train_loader,
                    validation_loader,
                    epochs=recovery_epochs,
                    learning_rate=recovery_learning_rate,
                    device=device,
                    precision=precision,
                    verbose=False,
                )
                validation = self.evaluator(recovered_model, validation_loader, device)
                record["recovery"] = _json_safe(recovery_history)
                record["validation"] = validation
                if frontier_archive is not None:
                    frontier_point = FrontierPoint(
                        validation_accuracy=validation["accuracy"],
                        parameter_count=_parameter_count(recovered_model),
                        compression_ratio=history.initial_parameter_count / max(
                            1, _parameter_count(recovered_model)
                        ),
                        candidate_spec=spec.to_dict(),
                        iteration=iteration,
                    )
                    frontier_archive.add(frontier_point)
                    history.frontier_points = [point.to_dict() for point in frontier_archive.points]
                recovered_shortlist.append(
                    (spec, recovered_model, critic_result, record, validation)
                )

            recovered_shortlist.sort(
                key=lambda item: (
                    -float(item[4]["accuracy"]),
                    -(item[0].parent_parameter_count - item[0].parameter_count),
                    item[0].fingerprint,
                )
            )
            best_spec, recovered_model, critic_result, best_record, validation = recovered_shortlist[0]
            fingerprint = best_spec.fingerprint
            recovered_count = _parameter_count(recovered_model)
            quality = 1.0 / (1.0 + float(validation.get("loss", critic_result.loss)))
            profile = CandidateProfile(
                fingerprint=fingerprint,
                quality=quality,
                accuracy=float(validation["accuracy"]),
                parent_accuracy=history.baseline_accuracy,
                parameter_count=recovered_count,
                parent_parameter_count=history.accepted_parameter_count,
            )
            decision = self.controller.decide_action(profile, history=history)
            best_record["decision"] = decision.to_dict()
            compression_ratio = (
                history.initial_parameter_count / recovered_count if recovered_count else 1.0
            )
            event = {
                "iteration": iteration,
                "candidates": audited,
                "fingerprint": fingerprint,
                "candidate_spec": best_spec.to_dict(),
                "importance_method": best_spec.importance_method,
                "actual_parameter_count": recovered_count,
                "parent_parameter_count": best_spec.parent_parameter_count,
                "compression_ratio": compression_ratio,
                "cheap_critic": critic_result.to_dict(),
                "recovery": best_record.get("recovery"),
                "validation": validation,
                "decision": decision.to_dict(),
            }

            if (
                decision.action == "accept"
                and max_allowed_compression is not None
                and compression_ratio > max_allowed_compression
            ):
                # Belt-and-suspenders: never accept an overshooting recovered model.
                history.consecutive_failures += 1
                best_record["final_action"] = "reject"
                event.update(
                    final_action="reject",
                    final_reason="target_compression_overshoot",
                    max_allowed_compression=max_allowed_compression,
                )
                history.add_event(event)
                continue
            if decision.action == "accept":
                current_model = recovered_model
                accepted_snapshot = copy.deepcopy(current_model)
                history.baseline_accuracy = float(validation["accuracy"])
                history.accepted_parameter_count = recovered_count
                history.consecutive_failures = 0
                best_record["final_action"] = "accept"
                event.update(final_action="accept", final_reason="constraints_satisfied")
                history.add_event(event)
                if target_ratio is not None and compression_ratio >= target_ratio:
                    history.add_event({
                        "iteration": iteration,
                        "action": "stop",
                        "reason": "target_compression_reached",
                        "compression_ratio": compression_ratio,
                        "target_compression_ratio": target_ratio,
                    })
                    break
                continue
            elif decision.action == "regrow":
                before = ratio_multiplier
                ratio_multiplier *= decision.next_ratio_multiplier
                history.ratio_multiplier = ratio_multiplier
                history.consecutive_failures += 1
                best_record.update(
                    final_action="regrow",
                    regrow_multiplier_before=before,
                    regrow_multiplier_after=ratio_multiplier,
                )
                event.update(final_action="regrow", final_reason=decision.reason)
            elif decision.action == "rollback":
                current_model = copy.deepcopy(accepted_snapshot)
                history.consecutive_failures += 1
                best_record.update(
                    final_action="rollback",
                    final_reason="rollback_failure_limit_reached",
                    rollback_restored=True,
                )
                event.update(
                    final_action="rollback",
                    final_reason="rollback_failure_limit_reached",
                    terminal=True,
                )
                history.add_event(event)
                break
            else:
                history.consecutive_failures += 1
                best_record["final_action"] = "reject"
                event.update(final_action="reject", final_reason=decision.reason)
            history.add_event(event)

        return current_model, history

    def _round_candidate_ratios(
        self,
        *,
        current_model: nn.Module,
        configured_ratios: Sequence[float],
        target_ratio: float | None,
        current_compression: float,
        max_step_compression: float,
    ) -> List[float]:
        """Choose prune ratios for this round; cap step size when chasing a target."""
        if target_ratio is None:
            return [float(ratio) for ratio in configured_ratios if 0.0 < float(ratio) < 1.0]

        remaining = float(target_ratio) / max(float(current_compression), 1e-9)
        if remaining <= 1.0 + 1e-6:
            return []
        step = min(float(max_step_compression), remaining)
        mild = min(step, 1.0 + 0.5 * (step - 1.0))
        model_type = self.model_type or "mlp"
        ratios: List[float] = []
        for step_compression in (mild, step):
            if step_compression <= 1.0 + 1e-6:
                continue
            try:
                prune_ratio = derive_uniform_prune_ratio(current_model, model_type, step_compression)
            except ValueError:
                continue
            # Allow tiny ratios near the target; discrete channel counts often need them.
            if 0.0 < prune_ratio < 1.0:
                ratios.append(float(prune_ratio))
        # Deduplicate while preserving order (mild then aggressive).
        ordered: List[float] = []
        for ratio in ratios:
            if all(abs(ratio - existing) > 1e-4 for existing in ordered):
                ordered.append(ratio)
        # Never fall back to full-target configured_ratios while chasing a target:
        # that overshoots on an already-compressed model (sweep_v3 4x seed43).
        return ordered

    def _backend_for_model(self, model: nn.Module) -> PruningBackend:
        if self.pruning_backend is not None:
            return self.pruning_backend
        return resolve_pruning_backend(model, self.model_type)

    def _call_importance_fn(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        backend: PruningBackend,
    ) -> Dict[str, torch.Tensor]:
        try:
            return self.importance_fn(model, dataloader, device, backend)
        except TypeError:
            return self.importance_fn(model, dataloader, device)

    def _default_importance_fn(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        backend: PruningBackend,
    ) -> Dict[str, torch.Tensor]:
        analyzer = SensitivityAnalyzer(model, device=device)
        layer_names = backend.prunable_layer_names()
        if isinstance(backend, MlpBackend):
            return analyzer.compute_wanda_importance(dataloader, num_batches=1)
        return analyzer.compute_wanda_importance(dataloader, num_batches=1, layer_names=layer_names)

    @staticmethod
    def _generate_candidates(
        model: nn.Module,
        importance: Dict[str, torch.Tensor],
        ratios: Sequence[float],
        multiplier: float,
        limit: Optional[int],
        attempted: set[str],
        enable_two_layer_candidates: bool = False,
        backend: Optional[PruningBackend] = None,
    ) -> List[CandidateSpec]:
        backend = backend or resolve_pruning_backend(model)
        parent_parameter_count = _parameter_count(model)
        candidates = []
        layer_candidates = {}
        if isinstance(backend, CnnBackend):
            for ratio in ratios:
                uniform = _uniform_all_layer_candidate(
                    backend, importance, ratio, multiplier, parent_parameter_count
                )
                if uniform is not None and uniform.fingerprint not in attempted:
                    candidates.append(uniform)
                if limit is not None and len(candidates) == limit:
                    return candidates
        for layer_name in backend.prunable_layer_names():
            scores = importance.get(layer_name)
            if scores is None:
                continue
            if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
                raise ValueError(f"importance scores for {layer_name} must be a one-dimensional tensor")
            output_size = backend.output_size(layer_name)
            if scores.numel() != output_size:
                continue
            scores = scores.detach().reshape(-1).cpu()
            if not torch.isfinite(scores).all().item():
                raise ValueError(f"importance scores for {layer_name} must be finite")
            for ratio in ratios:
                adjusted_ratio = min(float(ratio) * multiplier, 0.99)
                if not 0.0 < adjusted_ratio < 1.0:
                    continue
                num_keep = max(1, int(output_size * (1.0 - adjusted_ratio)))
                ranked_indices = sorted(
                    range(output_size), key=lambda index: (-float(scores[index]), index)
                )
                keep_indices = sorted(ranked_indices[:num_keep])
                spec = CandidateSpec.create(
                    {layer_name: adjusted_ratio},
                    {layer_name: keep_indices},
                    importance_method="wanda",
                    parameter_count=0,
                    parent_parameter_count=parent_parameter_count,
                )
                if spec.fingerprint not in attempted:
                    candidates.append(spec)
                    layer_candidates.setdefault(layer_name, []).append(spec)
                if limit is not None and len(candidates) == limit:
                    return candidates
        if enable_two_layer_candidates:
            layer_names = sorted(layer_candidates)
            for left_index, left_name in enumerate(layer_names):
                for right_name in layer_names[left_index + 1:]:
                    for left_spec in layer_candidates[left_name]:
                        for right_spec in layer_candidates[right_name]:
                            combined_ratios = left_spec.ratios_dict()
                            combined_ratios.update(right_spec.ratios_dict())
                            combined_indices = left_spec.keep_indices_dict()
                            combined_indices.update(right_spec.keep_indices_dict())
                            spec = CandidateSpec.create(
                                combined_ratios,
                                combined_indices,
                                importance_method="wanda",
                                parameter_count=0,
                                parent_parameter_count=parent_parameter_count,
                            )
                            if spec.fingerprint not in attempted:
                                candidates.append(spec)
                            if limit is not None and len(candidates) == limit:
                                return candidates
        return candidates


def _uniform_all_layer_candidate(
    backend: PruningBackend,
    importance: Dict[str, torch.Tensor],
    ratio: float,
    multiplier: float,
    parent_parameter_count: int,
) -> Optional[CandidateSpec]:
    adjusted_ratio = min(float(ratio) * multiplier, 0.99)
    if not 0.0 < adjusted_ratio < 1.0:
        return None
    layer_ratios: Dict[str, float] = {}
    keep_indices: Dict[str, List[int]] = {}
    for layer_name in backend.prunable_layer_names():
        scores = importance.get(layer_name)
        if scores is None or not isinstance(scores, torch.Tensor) or scores.ndim != 1:
            return None
        output_size = backend.output_size(layer_name)
        if scores.numel() != output_size:
            return None
        scores = scores.detach().reshape(-1).cpu()
        if not torch.isfinite(scores).all().item():
            raise ValueError(f"importance scores for {layer_name} must be finite")
        num_keep = max(1, int(output_size * (1.0 - adjusted_ratio)))
        ranked_indices = sorted(
            range(output_size), key=lambda index: (-float(scores[index]), index)
        )
        layer_ratios[layer_name] = adjusted_ratio
        keep_indices[layer_name] = sorted(ranked_indices[:num_keep])
    if not layer_ratios:
        return None
    return CandidateSpec.create(
        layer_ratios,
        keep_indices,
        importance_method="wanda",
        parameter_count=0,
        parent_parameter_count=parent_parameter_count,
    )


def config_to_json(config: Dict[str, float]) -> Dict[str, float]:
    return {name: float(config[name]) for name in sorted(config)}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _loader_sample_count(dataloader: DataLoader) -> int:
    try:
        return len(dataloader.dataset)
    except TypeError as error:
        raise ValueError("full evaluation requires a dataloader with a finite dataset") from error


def effective_max_step_compression(
    configured_max_step: float,
    target_compression_ratio: float | None,
) -> float:
    """Raise step cap to the target only in the very-low-compression regime.

    Empirically, allowing a one-shot jump helps at 2x but hurts at 4x because the
    final layer widths match iterative while intermediate recovery is skipped.
    """
    step_cap = float(configured_max_step)
    if target_compression_ratio is not None and float(target_compression_ratio) <= 2.0:
        step_cap = max(step_cap, float(target_compression_ratio))
    return step_cap


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize search value of type {type(value).__name__}")
