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
from src.pruning.sensitivity import SensitivityAnalyzer
from src.pruning.structured_pruning import StructuredPruning
from src.recovery.reconstruction import quick_recovery


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

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_accuracy": float(self.baseline_accuracy),
            "accepted_parameter_count": int(self.accepted_parameter_count),
            "consecutive_failures": int(self.consecutive_failures),
            "ratio_multiplier": float(self.ratio_multiplier),
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
    ) -> None:
        self.controller = controller
        self.critic = critic if critic is not None else CheapCritic()
        self.recovery_fn = recovery_fn
        self.evaluator = evaluator
        self.importance_fn = importance_fn or self._wanda_importance

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
        enable_two_layer_candidates: bool = False,
        recovery_top_k: int = 1,
    ) -> Tuple[nn.Module, SearchHistory]:
        if max_iterations < 1 or candidates_per_round < 1 or recovery_epochs < 0:
            raise ValueError("iteration, candidate, and recovery limits must be valid")
        if recovery_top_k < 1:
            raise ValueError("recovery_top_k must be positive")
        if not candidate_ratios:
            raise ValueError("candidate_ratios must not be empty")

        current_model = copy.deepcopy(parent_model)
        self.controller.reset()
        baseline = self.evaluator(current_model, validation_loader, device)
        history = SearchHistory(
            baseline_accuracy=float(baseline["accuracy"]),
            accepted_parameter_count=_parameter_count(current_model),
        )
        ratio_multiplier = 1.0
        accepted_snapshot = copy.deepcopy(current_model)

        for iteration in range(max_iterations):
            history.ratio_multiplier = ratio_multiplier
            importance = self.importance_fn(current_model, train_loader, device)
            candidates = self._generate_candidates(
                current_model,
                importance,
                candidate_ratios,
                ratio_multiplier,
                attempted=history.attempted_fingerprints,
                limit=None,
                enable_two_layer_candidates=enable_two_layer_candidates,
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
                    candidate_model = StructuredPruning(current_model).create_pruned_model_by_indices(
                        spec.keep_indices_dict()
                    )
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
                    if spec.fingerprint in history.attempted_fingerprints:
                        record.update(audit_status="filtered", final_reason="duplicate_attempted_fingerprint")
                    elif actual_count >= parent_parameter_count:
                        record.update(audit_status="filtered", final_reason="no_parameter_reduction")
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

            best_spec, candidate_model, critic_result, best_record = recovery_candidates[0]
            fingerprint = best_spec.fingerprint
            quality = 1.0 / (1.0 + critic_result.loss)
            profile = CandidateProfile(
                fingerprint=fingerprint,
                quality=quality,
                accuracy=critic_result.accuracy,
                parent_accuracy=history.baseline_accuracy,
                parameter_count=best_spec.parameter_count,
                parent_parameter_count=history.accepted_parameter_count,
            )
            decision = self.controller.decide_action(profile, history=history)
            best_record["decision"] = decision.to_dict()
            event = {
                "iteration": iteration,
                "candidates": audited,
                "fingerprint": fingerprint,
                "candidate_spec": best_spec.to_dict(),
                "importance_method": best_spec.importance_method,
                "actual_parameter_count": best_spec.parameter_count,
                "parent_parameter_count": best_spec.parent_parameter_count,
                "compression_ratio": best_spec.compression_ratio,
                "cheap_critic": critic_result.to_dict(),
                "decision": decision.to_dict(),
            }

            if decision.action == "accept":
                recovered_model, recovery_history = self.recovery_fn(
                    candidate_model, train_loader, validation_loader,
                    epochs=recovery_epochs, learning_rate=recovery_learning_rate,
                    device=device, verbose=False,
                )
                validation = self.evaluator(recovered_model, validation_loader, device)
                accepted = (
                    validation["accuracy"] >= history.baseline_accuracy - self.controller.max_accuracy_drop_points
                    and _parameter_count(recovered_model) < history.accepted_parameter_count
                )
                best_record["recovery"] = _json_safe(recovery_history)
                best_record["validation"] = validation
                if accepted:
                    current_model = recovered_model
                    accepted_snapshot = copy.deepcopy(current_model)
                    history.baseline_accuracy = float(validation["accuracy"])
                    history.accepted_parameter_count = _parameter_count(recovered_model)
                    history.consecutive_failures = 0
                    best_record["final_action"] = "accept"
                    event.update(final_action="accept", final_reason="constraints_satisfied")
                else:
                    history.consecutive_failures += 1
                    best_record.update(final_action="reject", final_reason="full_validation_failed")
                    event.update(final_action="reject", final_reason="full_validation_failed")
            elif decision.action == "regrow":
                before = ratio_multiplier
                ratio_multiplier *= decision.next_ratio_multiplier
                history.ratio_multiplier = ratio_multiplier
                history.consecutive_failures += 1
                best_record.update(final_action="regrow", regrow_multiplier_before=before, regrow_multiplier_after=ratio_multiplier)
                event.update(final_action="regrow", final_reason=decision.reason)
            elif decision.action == "rollback":
                current_model = copy.deepcopy(accepted_snapshot)
                history.consecutive_failures += 1
                best_record.update(final_action="rollback", final_reason="rollback_failure_limit_reached", rollback_restored=True)
                event.update(final_action="rollback", final_reason="rollback_failure_limit_reached", terminal=True)
                history.add_event(event)
                break
            else:
                history.consecutive_failures += 1
                best_record["final_action"] = "reject"
                event.update(final_action="reject", final_reason=decision.reason)
            history.add_event(event)

        return current_model, history

    @staticmethod
    def _wanda_importance(
        model: nn.Module, dataloader: DataLoader, device: str
    ) -> Dict[str, torch.Tensor]:
        return SensitivityAnalyzer(model, device=device).compute_wanda_importance(
            dataloader, num_batches=1
        )

    @staticmethod
    def _generate_candidates(
        model: nn.Module,
        importance: Dict[str, torch.Tensor],
        ratios: Sequence[float],
        multiplier: float,
        limit: Optional[int],
        attempted: set[str],
        enable_two_layer_candidates: bool = False,
    ) -> List[CandidateSpec]:
        pruner = StructuredPruning(model)
        parent_parameter_count = _parameter_count(model)
        candidates = []
        layer_candidates = {}
        for layer_name in pruner._hidden_linear_layer_names():
            scores = importance.get(layer_name)
            if scores is None:
                continue
            layer = pruner._validate_hidden_linear(layer_name)
            if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
                raise ValueError(f"importance scores for {layer_name} must be a one-dimensional tensor")
            if scores.numel() != layer.out_features:
                continue
            scores = pruner._validate_importance_scores(layer, scores)
            for ratio in ratios:
                adjusted_ratio = min(float(ratio) * multiplier, 0.99)
                if not 0.0 < adjusted_ratio < 1.0:
                    continue
                num_keep = max(1, int(layer.out_features * (1.0 - adjusted_ratio)))
                ranked_indices = sorted(
                    range(layer.out_features), key=lambda index: (-float(scores[index]), index)
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


def config_to_json(config: Dict[str, float]) -> Dict[str, float]:
    return {name: float(config[name]) for name in sorted(config)}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _loader_sample_count(dataloader: DataLoader) -> int:
    try:
        return len(dataloader.dataset)
    except TypeError as error:
        raise ValueError("full evaluation requires a dataloader with a finite dataset") from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize search value of type {type(value).__name__}")
