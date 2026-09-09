"""Compression adapters (interfaces + stub / real implementations)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class CompressionAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def compress(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return a result dict with at least status and artifact paths."""


class StubAdapter(CompressionAdapter):
    """Dry-run adapter: does not touch weights."""

    name = "stub"

    def compress(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        out = str(output_dir) if output_dir else None
        return {
            "status": "dry_run",
            "backend": recipe.get("backend", "stub"),
            "method": recipe.get("method"),
            "model_ref": model_ref,
            "recipe_id": recipe.get("recipe_id"),
            "output_dir": out,
            "message": "[OK] Stub compress; no weights modified",
        }


class LLMCompressorAdapter(StubAdapter):
    name = "llm_compressor"


class AutoRoundAdapter(StubAdapter):
    name = "autoround"


def _torchao_adapter() -> CompressionAdapter:
    from compression_harness.adapters.torchao_adapter import TorchAOAdapter

    return TorchAOAdapter()  # type: ignore[return-value]


ADAPTERS: dict[str, CompressionAdapter] = {
    "stub": StubAdapter(),
    "llm_compressor": LLMCompressorAdapter(),
    "autoround": AutoRoundAdapter(),
}


def get_adapter(backend: str) -> CompressionAdapter:
    if backend == "torchao":
        return _torchao_adapter()
    if backend not in ADAPTERS:
        raise KeyError(f"Unknown backend {backend!r}; known={sorted(list(ADAPTERS) + ['torchao'])}")
    return ADAPTERS[backend]
