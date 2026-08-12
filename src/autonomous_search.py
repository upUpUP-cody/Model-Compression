"""Autonomous, CPU-oriented structured-pruning search."""
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.controller.heuristic_controller import CandidateProfile, HeuristicController
from src.evaluation.cheap_critic import CheapCritic, CheapCriticResult
from src.pruning.sensitivity import SensitivityAnalyzer
from src.pruning.structured_pruning import StructuredPruning
from src.recovery.reconstruction import quick_recovery


@dataclass
class SearchHistory:
    """JSON-safe records from an autonomous pruning search."""

    baseline_accuracy: float
    accepted_parameter_count: int
    events: List[Dict[str, Any]] = field(default_factory=list)
    attempted_fingerprints: set[str] = field(default_factory=set, repr=False)
    consecutive_failures: int = 0

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_accuracy": float(self.baseline_accuracy),
            "accepted_parameter_count": int(self.accepted_parameter_count),
            "consecutive_failures": int(self.consecutive_failures),
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
    ) -> Tuple[nn.Module, SearchHistory]:
        if max_iterations < 1 or candidates_per_round < 1 or recovery_epochs < 0:
            raise ValueError("iteration, candidate, and recovery limits must be valid")
        if not candidate_ratios:
            raise ValueError("candidate_ratios must not be empty")

        current_model = copy.deepcopy(parent_model)
        baseline = self.evaluator(current_model, validation_loader, device)
        history = SearchHistory(
            baseline_accuracy=float(baseline["accuracy"]),
            accepted_parameter_count=_parameter_count(current_model),
        )
        ratio_multiplier = 1.0

        for iteration in range(max_iterations):
            importance = self.importance_fn(current_model, train_loader, device)
            candidates = self._generate_candidates(
                current_model,
                importance,
                candidate_ratios,
                ratio_multiplier,
                candidates_per_round,
                history.attempted_fingerprints,
            )
            if not candidates:
                history.add_event({"iteration": iteration, "action": "stop", "reason": "no_new_candidates"})
                break

            critic_candidates = []
            for config in candidates:
                fingerprint = candidate_fingerprint(config)
                candidate_model = StructuredPruning(current_model).create_pruned_model(config)
                critic_result = self.critic.evaluate(
                    candidate_model, validation_loader, cheap_eval_samples, device
                )
                critic_candidates.append((config, fingerprint, candidate_model, critic_result))

            best_config, fingerprint, candidate_model, critic_result = min(
                critic_candidates, key=lambda item: item[3].loss
            )
            history.attempted_fingerprints.add(fingerprint)
            quality = 1.0 / (1.0 + critic_result.loss)
            profile = CandidateProfile(
                fingerprint=fingerprint,
                quality=quality,
                accuracy=critic_result.accuracy,
                parent_accuracy=history.baseline_accuracy,
                parameter_count=critic_result.parameter_count,
                parent_parameter_count=history.accepted_parameter_count,
            )
            decision = self.controller.decide_action(profile, history=history)
            event = {
                "iteration": iteration,
                "config": config_to_json(best_config),
                "fingerprint": fingerprint,
                "cheap_critic": critic_result.to_dict(),
                "decision": decision.to_dict(),
            }

            if decision.action == "accept":
                recovered_model, recovery_history = self.recovery_fn(
                    candidate_model,
                    train_loader,
                    validation_loader,
                    epochs=recovery_epochs,
                    learning_rate=recovery_learning_rate,
                    device=device,
                    verbose=False,
                )
                validation = self.evaluator(recovered_model, validation_loader, device)
                accepted = (
                    validation["accuracy"] >= history.baseline_accuracy - self.controller.max_accuracy_drop_points
                    and _parameter_count(recovered_model) < history.accepted_parameter_count
                )
                event["recovery"] = _json_safe(recovery_history)
                event["validation"] = validation
                if accepted:
                    current_model = recovered_model
                    history.baseline_accuracy = float(validation["accuracy"])
                    history.accepted_parameter_count = _parameter_count(recovered_model)
                    history.consecutive_failures = 0
                    event["final_action"] = "accept"
                else:
                    history.consecutive_failures += 1
                    event["final_action"] = "reject"
                    event["final_reason"] = "full_validation_failed"
            elif decision.action == "regrow":
                ratio_multiplier *= decision.next_ratio_multiplier
                history.consecutive_failures += 1
                event["final_action"] = "regrow"
            elif decision.action == "rollback":
                history.consecutive_failures += 1
                event["final_action"] = "rollback"
            else:
                history.consecutive_failures += 1
                event["final_action"] = "reject"
            history.add_event(event)

        return current_model, history

    @staticmethod
    def _wanda_importance(
        model: nn.Module, dataloader: DataLoader, device: str
    ) -> Dict[str, torch.Tensor]:
        analyzer = SensitivityAnalyzer(model, device=device)
        pruner = StructuredPruning(model)
        return {
            name: analyzer.get_neuron_importance(name, dataloader, method="wanda", num_batches=1)
            for name in pruner._hidden_linear_layer_names()
        }

    @staticmethod
    def _generate_candidates(
        model: nn.Module,
        importance: Dict[str, torch.Tensor],
        ratios: Sequence[float],
        multiplier: float,
        limit: int,
        attempted: set[str],
    ) -> List[Dict[str, float]]:
        pruner = StructuredPruning(model)
        candidates = []
        for layer_name in pruner._hidden_linear_layer_names():
            if layer_name not in importance or importance[layer_name] is None:
                continue
            for ratio in ratios:
                adjusted_ratio = min(float(ratio) * multiplier, 0.99)
                if adjusted_ratio <= 0.0:
                    continue
                config = {layer_name: adjusted_ratio}
                if candidate_fingerprint(config) not in attempted:
                    candidates.append(config)
                if len(candidates) == limit:
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
