"""Experiment memory: JSON run artifacts under EXPERIMENTS_DIR."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compression_harness.paths import EXPERIMENTS_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentMemory:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else EXPERIMENTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_artifact(self, run_id: str, artifact: dict[str, Any]) -> Path:
        artifact = dict(artifact)
        artifact.setdefault("run_id", run_id)
        artifact.setdefault("created_at", _utc_now())
        path = self.run_dir(run_id) / "artifact.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return path

    def read_artifact(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "artifact.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("run_"))
