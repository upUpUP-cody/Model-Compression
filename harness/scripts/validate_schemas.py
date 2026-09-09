#!/usr/bin/env python3
"""Validate harness Goal/Recipe YAML against JSON Schema."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compression_harness.paths import RECIPES_DIR  # noqa: E402
from compression_harness.schema_util import load_yaml, validate_instance  # noqa: E402


def main() -> int:
    failures = 0

    goals = (
        "goal_near_lossless_small.yaml",
        "goal_lossless_iter.yaml",
        "goal_lossy5_iter.yaml",
    )
    for name in goals:
        path = RECIPES_DIR / name
        errs = validate_instance(load_yaml(path), "goal.schema.json")
        if errs:
            print(f"[FAIL] {name}: {errs}")
            failures += 1
        else:
            print(f"[OK] {name}")

    for name in (
        "global_int8.yaml",
        "global_int8_llmcompressor.yaml",
        "conservative_mixed.yaml",
    ):
        path = RECIPES_DIR / name
        errs = validate_instance(load_yaml(path), "recipe.schema.json")
        if errs:
            print(f"[FAIL] {name}: {errs}")
            failures += 1
        else:
            print(f"[OK] {name}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
