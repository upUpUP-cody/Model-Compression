#!/usr/bin/env python3
"""E9 Hard/Gap recovery strategies — cell-resumable (PDF Stage C)."""
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
    STRATEGIES,
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


def _parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_c/e9_high_gap_recovery.yaml"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--strategies", default=None, help="comma strategies")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    resume = bool(config.get("resume", True)) and not args.no_resume
    n_examples = int(args.n if args.n is not None else config.get("n_examples", 512))
    seeds = _parse_int_list(args.seeds) if args.seeds else list(config.get("seeds", [42]))
    strategies = (
        _parse_str_list(args.strategies)
        if args.strategies
        else list(config.get("strategies") or list(STRATEGIES))
    )
    recovery_cfg = dict(config.get("recovery") or {})
    if args.max_steps is not None:
        recovery_cfg["max_steps"] = int(args.max_steps)

    max_steps = int(recovery_cfg.get("max_steps") or 0)
    planned = [f"{st}_n{n_examples}_seed{s}_steps{max_steps}" for st in strategies for s in seeds]
    print(
        f"[INFO] E9 start n={n_examples} seeds={seeds} strategies={strategies} "
        f"max_steps={max_steps} resume={resume}"
    )
    write_progress(
        out,
        "E9",
        [
            f"- planned cells: {len(planned)}",
            f"- n={n_examples} strategies={strategies} max_steps={max_steps}",
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
    scored = score_pool(model, child_base, pool_loader, device)
    write_json(out / "pool_scores_preview.json", {"n": len(scored), "head": scored[:5]})

    done = 0
    skipped = 0
    for strategy in strategies:
        for seed in seeds:
            cell_id = f"{strategy}_n{n_examples}_seed{seed}_steps{max_steps}"
            if resume and cell_done(out, cell_id):
                print(f"[SKIP] {cell_id}")
                skipped += 1
                continue
            print(f"[RUN] {cell_id}")
            t0 = time.time()
            indices = select_indices(scored, strategy, n_examples, seed)
            train_sub = subset_loader(pool_loader, indices, batch_size=1)
            recovered, hist = run_lora_on_subset(
                child_base, train_sub, val_loader, recovery_cfg, device
            )
            rec_metrics = eval_proxy(recovered, val_loader, device)
            gain = recovery_gain(compressed_metrics, rec_metrics)
            payload: Dict[str, Any] = {
                "experiment_id": "E9",
                "strategy": strategy,
                "n_examples": n_examples,
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
                "E9",
                [
                    f"- planned={len(planned)} done_files={len(cells)} this_session_new={done} skipped={skipped}",
                    f"- last_cell={cell_id} gain={gain:.6f}",
                    f"- resume_cmd: python experiments/stage_c/run_e9_high_gap_recovery.py --config {args.config}",
                ],
            )

    cells = load_all_cells(out)
    summary = {
        "experiment_id": "E9",
        "title": "Hard Data vs Compression Gap Data",
        "spec": config.get("model", {}).get("spec", "proxy_1.5B"),
        "status": "partial" if len(cells) < len(planned) else "done_proxy",
        "planned_cells": planned,
        "completed_cells": [c.get("cell_id") for c in cells],
        "records": cells,
    }
    write_json(out / "e9_summary.json", summary)

    rows = [
        "| cell | strategy | seed | gain |",
        "|------|----------|------|------|",
    ]
    for c in cells:
        rows.append(
            f"| {c.get('cell_id')} | {c.get('strategy')} | {c.get('seed')} | {c.get('recovery_gain'):.6f} |"
        )
    report = "\n".join(
        [
            "# E9. Hard Data vs Compression Gap Data",
            "",
            f"| status | {summary['status']} |",
            f"| completed | {len(cells)}/{len(planned)} |",
            "",
            *rows,
            "",
            "续跑：同命令自动 skip 已完成 `cells/*.json`。",
            "",
        ]
    )
    write_report(out / "e9_report.md", report)
    mirror_docs_report("E9", "high_gap_recovery", report)
    print(f"[OK] E9 session finished new={done} skipped={skipped} total_files={len(cells)}")


if __name__ == "__main__":
    main()
