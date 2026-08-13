"""Pareto archive for validation-accuracy and structured-model size."""
from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FrontierPoint:
    """A JSON-safe result from one fully validated candidate."""

    validation_accuracy: float
    parameter_count: int
    compression_ratio: float
    recovery_seconds: float = 0.0
    candidate_spec: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    run: Optional[str] = None
    iteration: Optional[int] = None

    def __post_init__(self) -> None:
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if self.recovery_seconds < 0:
            raise ValueError("recovery_seconds must be non-negative")
        object.__setattr__(self, "validation_accuracy", float(self.validation_accuracy))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "compression_ratio", float(self.compression_ratio))
        object.__setattr__(self, "recovery_seconds", float(self.recovery_seconds))
        if self.candidate_spec is not None:
            object.__setattr__(self, "candidate_spec", _json_safe(self.candidate_spec))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_accuracy": self.validation_accuracy,
            "parameter_count": self.parameter_count,
            "compression_ratio": self.compression_ratio,
            "recovery_seconds": self.recovery_seconds,
            "candidate_spec": self.candidate_spec,
            "seed": self.seed,
            "run": self.run,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FrontierPoint":
        return cls(**{key: value.get(key) for key in (
            "validation_accuracy", "parameter_count", "compression_ratio",
            "recovery_seconds", "candidate_spec", "seed", "run", "iteration"
        )})


class ParetoFrontier:
    """Maintain the non-dominated points for accuracy maximization and size minimization."""

    def __init__(self, points: Iterable[FrontierPoint] = ()) -> None:
        self._points: List[FrontierPoint] = []
        self.extend(points)

    @property
    def points(self) -> List[FrontierPoint]:
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self.points)

    @staticmethod
    def dominates(first: FrontierPoint, second: FrontierPoint) -> bool:
        return (
            first.validation_accuracy >= second.validation_accuracy
            and first.parameter_count <= second.parameter_count
            and (
                first.validation_accuracy > second.validation_accuracy
                or first.parameter_count < second.parameter_count
            )
        )

    def add(self, point: FrontierPoint) -> bool:
        """Add a point if it is not dominated; return whether the archive changed."""
        if not isinstance(point, FrontierPoint):
            raise TypeError("point must be a FrontierPoint")
        if point in self._points:
            return False
        if any(self.dominates(existing, point) for existing in self._points):
            return False
        self._points = [existing for existing in self._points if not self.dominates(point, existing)]
        if point not in self._points:
            self._points.append(point)
        self._points.sort(key=lambda item: (item.parameter_count, -item.validation_accuracy))
        return True

    def extend(self, points: Iterable[FrontierPoint]) -> None:
        for point in points:
            self.add(point)

    def best_under_accuracy_drop(
        self, baseline_accuracy: float, max_drop: float
    ) -> Optional[FrontierPoint]:
        if max_drop < 0:
            raise ValueError("max_drop must be non-negative")
        eligible = [
            point for point in self._points
            if point.validation_accuracy >= float(baseline_accuracy) - max_drop
        ]
        return max(eligible, key=lambda point: (point.compression_ratio, point.validation_accuracy), default=None)

    def best_under_parameter_budget(self, parameter_budget: int) -> Optional[FrontierPoint]:
        if parameter_budget < 0:
            raise ValueError("parameter_budget must be non-negative")
        eligible = [point for point in self._points if point.parameter_count <= parameter_budget]
        return max(eligible, key=lambda point: (point.validation_accuracy, -point.parameter_count), default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [point.to_dict() for point in self._points]}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ParetoFrontier":
        return cls(FrontierPoint.from_dict(item) for item in value.get("points", []))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported candidate_spec value: {type(value).__name__}")


FrontierArchive = ParetoFrontier
