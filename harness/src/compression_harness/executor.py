"""Execute a Recipe via the selected adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compression_harness.adapters import get_adapter


class Executor:
    def run(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        *,
        force_stub: bool = False,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        backend = "stub" if force_stub else str(recipe.get("backend", "stub"))
        adapter = get_adapter(backend)
        result = adapter.compress(model_ref, recipe, output_dir=output_dir)
        result["executor"] = "compression_harness.executor"
        return result
