"""Plugin base types and short-step validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ModelState:
    """In-memory / path handle for iterative compression."""

    model_ref: str
    checkpoint_dir: str | None = None
    round_index: int = 0
    applied_steps: list[dict[str, Any]] = field(default_factory=list)
    layer_sparsity: dict[int, float] = field(default_factory=dict)
    layer_bits: dict[int, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        from compression_harness.model_io import strip_runtime_meta

        return {
            "model_ref": self.model_ref,
            "checkpoint_dir": self.checkpoint_dir,
            "round_index": self.round_index,
            "applied_steps": list(self.applied_steps),
            "layer_sparsity": {str(k): v for k, v in self.layer_sparsity.items()},
            "layer_bits": {str(k): v for k, v in self.layer_bits.items()},
            "meta": strip_runtime_meta(dict(self.meta)),
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "ModelState":
        return cls(
            model_ref=str(data.get("model_ref", "")),
            checkpoint_dir=data.get("checkpoint_dir"),
            round_index=int(data.get("round_index") or 0),
            applied_steps=list(data.get("applied_steps") or []),
            layer_sparsity={int(k): float(v) for k, v in (data.get("layer_sparsity") or {}).items()},
            layer_bits={int(k): int(v) for k, v in (data.get("layer_bits") or {}).items()},
            meta=dict(data.get("meta") or {}),
        )


# Type alias for step dicts
StepSpec = dict[str, Any]


def validate_step_spec(
    step: Mapping[str, Any],
    *,
    max_layers_per_step: int = 2,
    max_sparsity_delta: float = 0.05,
    min_weight_bits: int = 8,
) -> list[str]:
    """Reject giant one-shot steps. Returns error strings (empty = ok)."""
    errors: list[str] = []
    kind = str(step.get("kind") or "")
    if kind not in ("quantize", "prune", "evaluate"):
        errors.append(f"invalid kind {kind!r}")

    lr = step.get("layer_range")
    if lr is not None:
        if not (isinstance(lr, (list, tuple)) and len(lr) == 2):
            errors.append("layer_range must be [lo, hi]")
        else:
            lo, hi = int(lr[0]), int(lr[1])
            if hi < lo:
                errors.append("layer_range hi < lo")
            n_layers = hi - lo + 1
            if n_layers > max_layers_per_step:
                errors.append(
                    f"layer span {n_layers} exceeds max_layers_per_step={max_layers_per_step}"
                )

    if kind == "prune":
        delta = step.get("sparsity_delta")
        if delta is None and step.get("sparsity") is not None:
            # absolute sparsity without delta: treat as delta for first apply
            delta = step.get("sparsity")
        if delta is not None and float(delta) > max_sparsity_delta + 1e-12:
            errors.append(
                f"sparsity_delta {delta} exceeds max_sparsity_delta={max_sparsity_delta}"
            )
        if step.get("sparsity") is not None and float(step["sparsity"]) > 0.5 + 1e-12 and lr is None:
            errors.append("global prune sparsity > 0.5 without layer_range is forbidden")

    if kind == "quantize":
        bits = step.get("weight_bits")
        if bits is not None and int(bits) < min_weight_bits:
            errors.append(f"weight_bits {bits} below min_weight_bits={min_weight_bits}")
        if bits is not None and int(bits) <= 4 and lr is None:
            errors.append("global INT4-or-lower without layer_range is forbidden")

    return errors


class CompressionPlugin(ABC):
    name: str = "base"
    kind: str = "base"

    @abstractmethod
    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        """Apply step; return updated state and metrics."""
