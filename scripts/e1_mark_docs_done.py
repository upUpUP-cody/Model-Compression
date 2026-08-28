#!/usr/bin/env python3
"""Update E1 docs after formal dual-GPU run completes."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E_MAP = ROOT / "docs/project/EXPERIMENT_E_MAP.md"
CHECKLIST = ROOT / "docs/process/PDF_SETUP_CHECKLIST.md"


def main() -> None:
    e_map = E_MAP.read_text(encoding="utf-8")
    e_map = e_map.replace(
        "| E1 | One-shot curve | 3B base | **aligned，未开跑** |",
        "| E1 | One-shot curve | 3B base | **done** |",
    )
    e_map = e_map.replace(
        "E1 **已对齐、未开跑**。你确认验收后再下载 base 并执行。",
        "E1 **正式跑完**（双卡 3+3，Wanda，batch=2）。产物见 `/mnt/data2/results/E1_oneshot_sparsity_curve/`。",
    )
    E_MAP.write_text(e_map, encoding="utf-8")

    checklist = CHECKLIST.read_text(encoding="utf-8")
    checklist = checklist.replace(
        "## E1 One-shot sparsity curve（已对齐 · **未开跑**）",
        "## E1 One-shot sparsity curve（**done**）",
    )
    checklist = checklist.replace(
        "| E1 | **aligned，未开跑** | （开跑后）`/mnt/data2/results/E1_oneshot_sparsity_curve/` |",
        "| E1 | **done** | `/mnt/data2/results/E1_oneshot_sparsity_curve/` |",
    )
    CHECKLIST.write_text(checklist, encoding="utf-8")
    print("[OK] updated E-MAP and PDF_SETUP_CHECKLIST for E1 done")


if __name__ == "__main__":
    main()
