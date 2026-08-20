#!/usr/bin/env python3
"""E8 Random recovery — cell-resumable (PDF Stage C)."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_c_common import (
    build_pool_loaders,
    cell_done,
    ensure_dir,
    eval_proxy,
    load_all_cells,
    load_model,
    load_yaml,
    mirror_docs_report,
    prune_mlp_ratio,
    recovery_gain,
    run_lora_on_subset,
    save_cell,
    score_pool,
    select_indices,
    subset_loader,
    write_json,
    write_progress,
    write_report,
)


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_c/e8_random_recovery.yaml"))
    parser.add_argument("--sizes", default=None, help="comma sizes, e.g. 256")
    parser.add_argument("--seeds", default=None, help="comma seeds, e.g. 42")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    resume = bool(config.get("resume", True)) and not args.no_resume
    sizes = _parse_int_list(args.sizes) if args.sizes else list(config.get("data_sizes", [256]))
    seeds = _parse_int_list(args.seeds) if args.seeds else list(config.get("seeds", [42]))
    recovery_cfg = dict(config.get("recovery") or {})
    if args.max_steps is not None:
        recovery_cfg["max_steps"] = int(args.max_steps)

    max_steps = int(recovery_cfg.get("max_steps") or 0)
    planned = [f"random_n{n}_seed{s}_steps{max_steps}" for n in sizes for s in seeds]
    print(
        f"[INFO] E8 Random recovery start sizes={sizes} seeds={seeds} "
        f"max_steps={max_steps} resume={resume}"
    )
    print(f"[INFO] output={out} cells={len(planned)}")
    write_progress(
        out,
        "E8",
        [
            f"- planned cells: {len(planned)}",
            f"- sizes={sizes} seeds={seeds} max_steps={max_steps}",
            f"- resume={resume}",
            "- status: starting",
        ],
    )

    model, tokenizer, device = load_model(config)
    pool_loader, val_loader, _meta = build_pool_loaders(config, tokenizer, device)
    dense_metrics = eval_proxy(model, val_loader, device)

    child_base = prune_mlp_ratio(copy.deepcopy(model), float(config.get("child_sparsity", 0.5)))
    child_base = child_base.to(device)
    compressed_metrics = eval_proxy(child_base, val_loader, device)

    # Score pool once (parent vs child) for random sampling consistency
    scored = score_pool(model, child_base, pool_loader, device)
    write_json(out / "pool_scores_preview.json", {"n": len(scored), "head": scored[:5]})

    done = 0
    skipped = 0
    for n in sizes:
        for seed in seeds:
            cell_id = f"random_n{n}_seed{seed}_steps{max_steps}"
            if resume and cell_done(out, cell_id):
                print(f"[SKIP] {cell_id}")
                skipped += 1
                continue
            print(f"[RUN] {cell_id}")
            t0 = time.time()
            indices = select_indices(scored, "random", n, seed)
            train_sub = subset_loader(pool_loader, indices, batch_size=1)
            recovered, hist = run_lora_on_subset(
                child_base, train_sub, val_loader, recovery_cfg, device
            )
            rec_metrics = eval_proxy(recovered, val_loader, device)
            gain = recovery_gain(compressed_metrics, rec_metrics)
            payload: Dict[str, Any] = {
                "experiment_id": "E8",
                "strategy": "random",
                "n_examples": n,
                "seed": seed,
                "dense": dense_metrics,
                "compressed": compressed_metrics,
                "recovered": rec_metrics,
                "recovery_gain": gain,
                "lora_history": {
                    "max_steps": hist.get("max_steps"),
                    "steps_ran": hist.get("steps_ran"),
                    "best_validation_loss": hist.get("best_validation_loss"),
                },
                "elapsed_sec": time.time() - t0,
                "spec": config.get("model", {}).get("spec", "proxy_1.5B"),
            }
            save_cell(out, cell_id, payload)
            done += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cells = load_all_cells(out)
            write_progress(
                out,
                "E8",
                [
                    f"- planned={len(planned)} done_files={len(cells)} this_session_new={done} skipped={skipped}",
                    f"- last_cell={cell_id} gain={gain:.6f} elapsed={payload['elapsed_sec']:.1f}s",
                    f"- resume_cmd: python experiments/stage_c/run_e8_random_recovery.py --config {args.config}",
                ],
            )

    cells = load_all_cells(out)
    summary = {
        "experiment_id": "E8",
        "title": "Random recovery",
        "spec": config.get("model", {}).get("spec", "proxy_1.5B"),
        "status": "partial" if len(cells) < len(planned) else "done_proxy",
        "planned_cells": planned,
        "completed_cells": [c.get("cell_id") for c in cells],
        "records": cells,
        "dense": dense_metrics,
        "compressed": compressed_metrics,
    }
    write_json(out / "e8_summary.json", summary)

    rows = [
        "| cell | n | seed | gain | comp_loss | rec_loss |",
        "|------|---|------|------|-----------|----------|",
    ]
    for c in cells:
        rows.append(
            f"| {c.get('cell_id')} | {c.get('n_examples')} | {c.get('seed')} | "
            f"{c.get('recovery_gain'):.6f} | {c.get('compressed', {}).get('loss'):.4f} | "
            f"{c.get('recovered', {}).get('loss'):.4f} |"
        )
    report = "\n".join(
        [
            "# E8. Recovery Data Baseline (Random)",
            "",
            "## 项目 / 设置",
            "",
            f"| Experiment ID | E8 |",
            f"| spec | {summary['spec']} |",
            f"| status | {summary['status']} |",
            f"| completed | {len(cells)}/{len(planned)} |",
            "",
            "## 记录表",
            "",
            *rows,
            "",
            "## 续跑",
            "",
            "已完成 cell 在 `cells/*.json`；再次运行同命令会自动 skip。",
            "",
        ]
    )
    write_report(out / "e8_report.md", report)
    mirror_docs_report("E8", "random_recovery", report)
    print(f"[OK] E8 session finished new={done} skipped={skipped} total_files={len(cells)}")


if __name__ == "__main__":
    main()
