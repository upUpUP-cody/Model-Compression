"""Diagnose quality regressions (stub)."""

from __future__ import annotations

from typing import Any


class Diagnoser:
    def diagnose(self, artifact: dict[str, Any], goal: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = goal
        drop = artifact.get("relative_drop")
        suggestion = "global_int8.yaml"
        if drop is not None and float(drop) > 0.02:
            suggestion = "conservative_mixed.yaml"
        return {
            "status": "dry_run",
            "hypothesis": (
                "Stub diagnosis: quality outside near-lossless band; "
                "prefer milder recipe or denser sensitive layers."
            ),
            "next_recipe_suggestion": suggestion,
            "run_id": artifact.get("run_id"),
        }
