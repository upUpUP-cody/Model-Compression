"""Deterministic heuristic decisions for autonomous pruning."""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class CandidateProfile:
    """Metrics used to decide whether a candidate may replace its parent."""

    fingerprint: str
    quality: float
    accuracy: float
    parent_accuracy: float
    parameter_count: int
    parent_parameter_count: int
    capability_gap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "quality": float(self.quality),
            "accuracy": float(self.accuracy),
            "parent_accuracy": float(self.parent_accuracy),
            "parameter_count": int(self.parameter_count),
            "parent_parameter_count": int(self.parent_parameter_count),
            "capability_gap": (
                None if self.capability_gap is None else float(self.capability_gap)
            ),
        }


@dataclass(frozen=True)
class ControllerDecision:
    """A JSON-safe controller action and its explanation."""

    action: str
    reason: str
    fingerprint: str
    next_ratio_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "next_ratio_multiplier": float(self.next_ratio_multiplier),
        }


class HeuristicController:
    """Apply deterministic compression and capability rules."""

    ACTIONS = {"accept", "reject", "rollback", "regrow"}

    def __init__(
        self,
        max_accuracy_drop_points: float = 5.0,
        max_failures: int = 3,
        min_quality_score: float = 0.6,
        regrow_ratio_multiplier: float = 0.8,
    ) -> None:
        if max_accuracy_drop_points < 0:
            raise ValueError("max_accuracy_drop_points must be non-negative")
        if max_failures < 1:
            raise ValueError("max_failures must be positive")
        if not 0.0 <= min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be in [0.0, 1.0]")
        if not 0.0 < regrow_ratio_multiplier < 1.0:
            raise ValueError("regrow_ratio_multiplier must be in (0.0, 1.0)")

        self.max_accuracy_drop_points = float(max_accuracy_drop_points)
        self.max_failures = int(max_failures)
        self.min_quality_score = float(min_quality_score)
        self.regrow_ratio_multiplier = float(regrow_ratio_multiplier)
        self._seen_fingerprints: Set[str] = set()

    @property
    def seen_fingerprints(self) -> Set[str]:
        return set(self._seen_fingerprints)

    def decide_action(
        self,
        profile: CandidateProfile | Dict[str, Any],
        candidate: Optional[Dict[str, Any]] = None,
        history: Optional[Any] = None,
    ) -> ControllerDecision:
        """Return one of accept, reject, rollback, or regrow.

        ``candidate`` is accepted for compatibility with the original plan;
        fields in it override fields in a dictionary profile.
        """
        normalized = self._normalize_profile(profile, candidate)
        fingerprint = normalized.fingerprint
        if fingerprint in self._seen_fingerprints:
            return ControllerDecision("reject", "duplicate_candidate", fingerprint)
        self._seen_fingerprints.add(fingerprint)

        if normalized.parameter_count >= normalized.parent_parameter_count:
            return ControllerDecision("reject", "no_parameter_reduction", fingerprint)

        failures = self._failure_count(history)
        if failures >= self.max_failures:
            return ControllerDecision("rollback", "failure_limit_reached", fingerprint)

        gap = normalized.capability_gap
        if gap is None:
            gap = normalized.parent_accuracy - normalized.accuracy
        if gap > self.max_accuracy_drop_points:
            return ControllerDecision(
                "regrow",
                "capability_gap_exceeded",
                fingerprint,
                self.regrow_ratio_multiplier,
            )
        if normalized.quality < self.min_quality_score:
            return ControllerDecision(
                "regrow",
                "quality_below_threshold",
                fingerprint,
                self.regrow_ratio_multiplier,
            )
        return ControllerDecision("accept", "constraints_satisfied", fingerprint)

    def reset(self) -> None:
        """Clear legacy direct-call duplicate tracking."""
        self._seen_fingerprints.clear()

    @staticmethod
    def _normalize_profile(
        profile: CandidateProfile | Dict[str, Any], candidate: Optional[Dict[str, Any]]
    ) -> CandidateProfile:
        if isinstance(profile, CandidateProfile):
            values = profile.to_dict()
        elif isinstance(profile, dict):
            values = dict(profile)
        else:
            raise TypeError("profile must be CandidateProfile or dict")
        if candidate:
            values.update(candidate)
        required = {
            "fingerprint",
            "quality",
            "accuracy",
            "parent_accuracy",
            "parameter_count",
            "parent_parameter_count",
        }
        missing = required.difference(values)
        if missing:
            raise ValueError(f"profile is missing fields: {sorted(missing)}")
        return CandidateProfile(
            fingerprint=str(values["fingerprint"]),
            quality=float(values["quality"]),
            accuracy=float(values["accuracy"]),
            parent_accuracy=float(values["parent_accuracy"]),
            parameter_count=int(values["parameter_count"]),
            parent_parameter_count=int(values["parent_parameter_count"]),
            capability_gap=(
                None
                if values.get("capability_gap") is None
                else float(values["capability_gap"])
            ),
        )

    def _seen_from_history(self, history: Optional[Any]) -> Set[str]:
        """Read run-owned duplicate state, retaining direct-call compatibility."""
        if history is None:
            return self._seen_fingerprints
        if isinstance(history, dict):
            return set(history.get("attempted_fingerprints", ()))
        return set(getattr(history, "attempted_fingerprints", ()))

    @staticmethod
    def _failure_count(history: Optional[Any]) -> int:
        if history is None:
            return 0
        if isinstance(history, dict):
            if "consecutive_failures" in history:
                return int(history["consecutive_failures"])
            failures = history.get("failures", [])
            return len(failures) if isinstance(failures, Iterable) else int(failures)
        if hasattr(history, "consecutive_failures"):
            return int(history.consecutive_failures)
        if hasattr(history, "failures"):
            failures = history.failures
            return len(failures) if isinstance(failures, Iterable) else int(failures)
        return 0
