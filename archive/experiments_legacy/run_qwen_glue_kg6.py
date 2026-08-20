#!/usr/bin/env python3
"""Phase KG.6: budget-aligned GLUE three-task scan (wrapper around KG.5 runner)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_qwen_glue_kg5 import main as kg5_main


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--config", str(ROOT / "configs" / "qwen_glue_kg6.yaml")]
    kg5_main()
