"""Paths and constants for the harness tree."""

from __future__ import annotations

import os
from pathlib import Path

# harness/src/compression_harness -> parents: package, src, harness, repo
HARNESS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = HARNESS_ROOT.parent
SCHEMAS_DIR = HARNESS_ROOT / "schemas"
RECIPES_DIR = HARNESS_ROOT / "recipes"
# Prefer large disk for multi-GB iterative checkpoints (root FS is often tiny).
_DEFAULT_EXPERIMENTS = Path("/mnt/data2/results/harness_experiments")
if os.environ.get("HARNESS_EXPERIMENTS_DIR"):
    EXPERIMENTS_DIR = Path(os.environ["HARNESS_EXPERIMENTS_DIR"])
elif _DEFAULT_EXPERIMENTS.parent.exists():
    EXPERIMENTS_DIR = _DEFAULT_EXPERIMENTS
else:
    EXPERIMENTS_DIR = HARNESS_ROOT / "experiments"
try:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
KNOWLEDGE_DIR = HARNESS_ROOT / "knowledge"

DEFAULT_GOAL_PATH = RECIPES_DIR / "goal_near_lossless_small.yaml"
# Phase B: only torchao global INT8 in the auto queue.
DEFAULT_RECIPE_ORDER = (
    "global_int8.yaml",
)
