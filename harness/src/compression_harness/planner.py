"""Planner: milder_first queue; stop when near-lossless satisfied."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compression_harness.paths import DEFAULT_RECIPE_ORDER, RECIPES_DIR
from compression_harness.schema_util import load_yaml


class Planner:
    def __init__(self, recipes_dir: Path | None = None, order: tuple[str, ...] | None = None) -> None:
        self.recipes_dir = Path(recipes_dir) if recipes_dir else RECIPES_DIR
        self.order = order or DEFAULT_RECIPE_ORDER

    def candidate_paths(self) -> list[Path]:
        paths: list[Path] = []
        for name in self.order:
            p = self.recipes_dir / name
            if p.exists():
                paths.append(p)
        return paths

    def next_recipe(
        self,
        tried: set[str],
        *,
        last_ok: bool | None = None,
    ) -> dict[str, Any] | None:
        """Return next recipe dict, or None to stop.

        If last_ok is True, stop immediately (do not chase higher compression).
        """
        if last_ok:
            return None
        for path in self.candidate_paths():
            recipe = load_yaml(path)
            rid = str(recipe.get("recipe_id") or path.stem)
            if rid in tried:
                continue
            recipe["_recipe_path"] = str(path)
            return recipe
        return None

    def should_stop(self, near_lossless_ok: bool, trials: int, max_trials: int) -> str | None:
        if near_lossless_ok:
            return "accepted_near_lossless"
        if trials >= max_trials:
            return "budget_exhausted"
        return None
