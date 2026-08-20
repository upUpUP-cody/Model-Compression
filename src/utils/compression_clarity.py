"""Clarify cell_target vs actual compression for dense/pruned method reports.

dense is an unpruned baseline (actual_compression=1.0). cell_target only
groups the comparison matrix and must not be read as "dense was pruned".
"""
from __future__ import annotations

from typing import Any, Dict


def compression_clarity_fields(
    method: str,
    cell_target: float,
    actual_compression: float,
) -> Dict[str, Any]:
    """Return explicit fields for metrics JSON / summary rows."""
    pruning_applied = str(method) != "dense"
    return {
        "cell_target": float(cell_target),
        "actual_compression": float(actual_compression),
        "pruning_applied": pruning_applied,
    }


def format_method_ok_line(
    *,
    method: str,
    cell_target: float,
    actual_compression: float,
    metric_parts: str,
    prefix: str = "",
) -> str:
    """Human-readable OK line that does not imply dense was compressed."""
    head = f"{prefix}cell={float(cell_target):g}x {method}"
    if str(method) == "dense":
        return (
            f"[OK] {head} (baseline, no prune): "
            f"actual={float(actual_compression):.3f}x {metric_parts}"
        )
    return (
        f"[OK] {head}: actual={float(actual_compression):.3f}x {metric_parts}"
    )


SUMMARY_NOTE_DENSE_BASELINE = (
    "dense is unpruned baseline (actual_compression=1.0); "
    "cell_target only groups the comparison matrix"
)
