#!/usr/bin/env python3
"""Soft-eval side ladder: Sentiment + easy/soft/lite dims on E1 Wanda curve (resumable).

Not Gate A. Does not rewrite formal E1/E2 six-dim results.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import eval_capability_vector
from src.experiments.soft_eval_common import (
    apply_easy_capability_overrides,
    build_soft_checkpoint_payload,
    cell_is_completed,
    empty_scores_table,
    load_soft_checkpoint,
    resolve_soft_dim_specs,
    save_soft_checkpoint,
    soft_cell_key,
    soft_eval_checkpoint_filename,
    soft_protocol_digest,
    sparsity_label,
    utc_now,
)
from src.experiments.stage_a_common import (
    ensure_dir,
    load_model,
    load_yaml,
    mirror_docs_report,
    prepare_wanda_calibration_loader,
    stage_a_prune_mlp,
    write_json,
    write_report,
)
from src.utils.qwen_glue_eval import evaluate_glue_split, prepare_glue_from_config


def _cuda_oom_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def _eval_capability_with_batch_fallback(
    model_path: str,
    dim_config: Dict[str, Any],
    *,
    model,
    tokenizer,
) -> Dict[str, Any]:
    """Try configured batch, then 8→4→2→1 on CUDA OOM."""
    cap0 = (dim_config.get("evaluation") or {}).get("capability") or {}
    primary = int(cap0.get("batch_size") or dim_config.get("hardware", {}).get("batch_size") or 16)
    cascade: List[int] = []
    for bs in (primary, 8, 4, 2, 1):
        if bs not in cascade:
            cascade.append(bs)
    last_exc: Optional[Exception] = None
    for idx, bs in enumerate(cascade):
        cfg = copy.deepcopy(dim_config)
        cfg.setdefault("evaluation", {}).setdefault("capability", {})["batch_size"] = bs
        cfg.setdefault("hardware", {})["batch_size"] = bs
        try:
            return eval_capability_vector(model_path, cfg, model=model, tokenizer=tokenizer)
        except Exception as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)
            if not is_oom:
                raise
            last_exc = exc
            _cuda_oom_cleanup()
            if idx + 1 >= len(cascade):
                break
            nxt = cascade[idx + 1]
            print(f"[WARNING] soft-eval batch={bs} OOM; retrying batch={nxt}: {exc}", flush=True)
    assert last_exc is not None
    raise last_exc


def _parse_sparsities_arg(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _eval_glue_classification(
    model,
    tokenizer,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: str,
    *,
    soft_dim: str,
) -> Tuple[float, Dict[str, float]]:
    """GLUE verbalizer classification (SST-2 / RTE / …).

    Returns (primary_score in 0-1, extras in 0-1 including accuracy / balanced / majority).
    """
    glue_task = str(spec.get("glue_task") or ("sst2" if spec.get("kind") == "glue_sst2" else "rte"))
    cfg = dict(config)
    ds = dict(cfg.get("dataset") or {})
    ds["task"] = glue_task
    cfg["dataset"] = ds
    splits = prepare_glue_from_config(cfg)
    val = splits.validation
    sent_cfg = dict((config.get("evaluation") or {}).get("sentiment") or {})
    max_samples = int(spec.get("max_samples", sent_cfg.get("max_samples", 256)))
    batch_size = int(sent_cfg.get("batch_size", config.get("hardware", {}).get("batch_size", 8)))
    metrics = evaluate_glue_split(
        model,
        tokenizer,
        val,
        device=device,
        batch_size=batch_size,
        max_seq_len=int(spec.get("max_seq_len", 256)),
        max_new_tokens=int(spec.get("max_new_tokens", 8)),
        max_samples=max_samples,
        split_name="validation",
        task=glue_task,
    )

    def _unit(key: str) -> Optional[float]:
        if key not in metrics or metrics[key] is None:
            return None
        val_f = float(metrics[key])
        if key.startswith("pred_rate"):
            return val_f
        if val_f > 1.0 + 1e-9:
            return val_f / 100.0
        return val_f

    extras: Dict[str, float] = {}
    for key in (
        "accuracy",
        "balanced_accuracy",
        "majority_baseline",
        "macro_f1",
        "pred_rate_0",
        "pred_rate_1",
        "n_examples",
    ):
        u = _unit(key) if key != "n_examples" else (
            float(metrics[key]) if metrics.get(key) is not None else None
        )
        if u is not None:
            extras[key] = float(u)

    metric_name = str(spec.get("metric") or "classification_acc").lower()
    primary_key = "accuracy"
    if "balanced" in metric_name:
        primary_key = "balanced_accuracy"
    elif metric_name in ("classification_acc", "accuracy", "acc", "glue_accuracy"):
        primary_key = "accuracy"

    primary = extras.get(primary_key)
    if primary is None:
        for key in ("accuracy", "acc", "glue_accuracy", "balanced_accuracy"):
            if key in metrics and metrics[key] is not None:
                score = float(metrics[key])
                if score > 1.0 + 1e-9:
                    score = score / 100.0
                primary = score
                break
    if primary is None:
        raise KeyError(f"{soft_dim} metrics missing accuracy keys: {list(metrics)}")
    return float(primary), extras


def _eval_capability_soft(
    model,
    tokenizer,
    model_path: str,
    base_config: Mapping[str, Any],
    soft_dim: str,
    spec: Mapping[str, Any],
    *,
    smoke: bool,
) -> float:
    dim_cfg = apply_easy_capability_overrides(base_config, soft_dim, spec, smoke=smoke)
    base_dim = str(spec["base_dim"])
    cap = _eval_capability_with_batch_fallback(
        model_path,
        dim_cfg,
        model=model,
        tokenizer=tokenizer,
    )
    vector = cap.get("vector") or {}
    score = vector.get(base_dim)
    if score is None:
        raise RuntimeError(f"{soft_dim}: capability returned no score for {base_dim}: {vector}")
    return float(score)


def _write_ckpt(
    path: Path,
    *,
    digest: str,
    smoke: bool,
    grid: List[float],
    soft_dims: List[str],
    completed: List[str],
    scores: Dict[str, Dict[str, Optional[float]]],
    started_at: str,
    t0: float,
    status: str,
    scores_extra: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> None:
    payload = build_soft_checkpoint_payload(
        protocol_digest=digest,
        smoke=smoke,
        sparsity_grid=grid,
        soft_dims=soft_dims,
        completed_cells=completed,
        scores=scores,
        started_at=started_at,
        elapsed_sec=time.time() - t0,
        status=status,
        scores_extra=scores_extra,
    )
    save_soft_checkpoint(path, payload)


def _build_report(
    config: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Optional[float]]],
    soft_dims: Sequence[str],
    grid: Sequence[float],
    *,
    status: str,
    scores_extra: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
) -> str:
    headers = ["Sparsity"] + list(soft_dims)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["------"] * len(headers)) + "|"]
    for sp in grid:
        lab = sparsity_label(sp)
        row = scores.get(lab) or {}
        cells = [f"{sp*100:.0f}%" if sp > 0 else "dense"]
        for d in soft_dims:
            v = row.get(d)
            cells.append("n/a" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    extras_block = ""
    if scores_extra:
        glue_dims = [
            d
            for d in soft_dims
            if any(d in (scores_extra.get(sparsity_label(sp)) or {}) for sp in grid)
        ]
        if glue_dims:
            eh = ["Sparsity"]
            for d in glue_dims:
                eh.extend([f"{d}_acc", f"{d}_bal", f"{d}_maj"])
            elines = [
                "| " + " | ".join(eh) + " |",
                "|" + "|".join(["------"] * len(eh)) + "|",
            ]
            for sp in grid:
                lab = sparsity_label(sp)
                extra_row = scores_extra.get(lab) or {}
                cells = [f"{sp*100:.0f}%" if sp > 0 else "dense"]
                for d in glue_dims:
                    m = extra_row.get(d) or {}
                    for key in ("accuracy", "balanced_accuracy", "majority_baseline"):
                        v = m.get(key)
                        cells.append("n/a" if v is None else f"{float(v):.4f}")
                elines.append("| " + " | ".join(cells) + " |")
            extras_block = (
                "\n## GLUE extras (acc / balanced / majority)\n\n"
                + "\n".join(elines)
                + "\n\n"
                "For Instruction_lite, Scores primary is **balanced_accuracy**. "
                "If accuracy rises toward majority while balanced stays ~0.5, that is "
                "class collapse, not recovery.\n"
            )

    exp_id = str(config.get("experiment_id") or "E1_soft_eval_side")
    return f"""# {exp_id} Soft-Eval Ladder (NOT Gate A)

## Setup

| Item | Value |
|------|-------|
| Model | {config.get("model", {}).get("pdf_model")} ({config.get("model", {}).get("path")}) |
| Weights | Wanda oneshot re-applied (same pruning as E1); Recovery None |
| Soft dims | {", ".join(soft_dims)} |
| Sparsity | {", ".join(f"{s*100:.0f}%" for s in grid)} |
| status | {status} |
| Gate | **Does not enter Gate A**; formal E1/E2 unchanged |

## Scores

{table}
{extras_block}
## Notes

- Sentiment = SST-2 **classification accuracy** (not Wanda LM calibration NLL).
- Instruction_lite primary = **balanced_accuracy** (RTE); see GLUE extras for acc/majority.
- `*_easy` = same-bank softer limits; `*_soft` / `*_lite` = hybrid easier tasks (维名不变).
- See `docs/process/NLP_EVAL_TASK_SURVEY.md` and `docs/results/E1_eval_ladder_all_banks.md`.
- Resume file: `soft_eval_checkpoint.json` (cell = sparsity|dim).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_soft_eval_side.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--sparsities", default=None, help="comma subset e.g. 0,0.1,0.2")
    parser.add_argument("--only-dims", default=None, help="comma soft dim subset")
    args = parser.parse_args()

    if os.environ.get("SOFT_EVAL_FRESH", "").strip() in {"1", "true", "True"}:
        args.fresh = True
    if args.fresh:
        args.resume = False

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    smoke = bool(args.smoke)
    specs = resolve_soft_dim_specs(config, smoke=smoke)
    soft_dims = list(specs.keys())
    if args.only_dims:
        want = {x.strip() for x in args.only_dims.split(",") if x.strip()}
        soft_dims = [d for d in soft_dims if d in want]
        if not soft_dims:
            raise SystemExit("only-dims matched nothing")

    full_grid = [float(x) for x in config.get("sparsity_grid", [0.0, 0.1, 0.2, 0.3, 0.4])]
    subset = _parse_sparsities_arg(args.sparsities)
    if subset is not None:
        grid = [s for s in full_grid if any(abs(s - t) < 1e-9 for t in subset)]
    else:
        grid = list(full_grid)
    if smoke and subset is None:
        grid = [0.0, 0.3]

    digest = soft_protocol_digest(config, smoke=smoke)
    ckpt_path = out / soft_eval_checkpoint_filename(smoke=smoke)

    if args.dry_run:
        print("[INFO] soft-eval dry-run")
        print(f"  out={out}")
        print(f"  dims={soft_dims}")
        print(f"  grid={grid}")
        print(f"  digest={digest[:16]}...")
        print(f"  checkpoint={ckpt_path}")
        return

    started_at = utc_now()
    t0 = time.time()
    completed: List[str] = []
    scores = empty_scores_table(grid, soft_dims)
    scores_extra: Dict[str, Dict[str, Dict[str, float]]] = {
        sparsity_label(sp): {} for sp in grid
    }

    if args.resume and not args.fresh:
        try:
            prev = load_soft_checkpoint(ckpt_path, digest)
        except ValueError as exc:
            print(f"[WARNING] checkpoint not loaded ({exc}); starting fresh scores table")
            prev = None
        if prev:
            completed = list(prev.get("completed_cells") or [])
            prev_scores = prev.get("scores") or {}
            for lab, row in prev_scores.items():
                if lab not in scores:
                    scores[lab] = {d: None for d in soft_dims}
                for d, v in dict(row).items():
                    if d in soft_dims:
                        scores[lab][d] = v
            prev_extra = prev.get("scores_extra") or {}
            for lab, row in dict(prev_extra).items():
                scores_extra.setdefault(str(lab), {})
                for d, metrics in dict(row).items():
                    if d in soft_dims and isinstance(metrics, dict):
                        scores_extra[str(lab)][str(d)] = {
                            str(k): float(v) for k, v in metrics.items()
                        }
            started_at = str(prev.get("started_at") or started_at)
            print(f"[INFO] soft-eval resume cells={len(completed)} from {ckpt_path}")

    print(
        f"[INFO] soft-eval start smoke={smoke} fresh={args.fresh} "
        f"dims={soft_dims} grid={grid}"
    )

    parent, tokenizer, device = load_model(config)
    model_path = str(config["model"]["path"])
    calib_loader = None
    if any(abs(s) > 1e-12 for s in grid):
        calib_loader = prepare_wanda_calibration_loader(config, tokenizer, device)

    for sp in grid:
        lab = sparsity_label(sp)
        scores.setdefault(lab, {d: None for d in soft_dims})
        scores_extra.setdefault(lab, {})

        pending_dims = [
            d for d in soft_dims if not cell_is_completed(completed, sp, d)
        ]
        if not pending_dims:
            print(f"[INFO] soft-eval skip sparsity={sp:.2f} (all dims done)")
            continue

        if abs(sp) < 1e-12:
            child = parent
        else:
            print(f"[INFO] soft-eval prune sparsity={sp:.2f}")
            parent.to("cpu")
            _cuda_oom_cleanup()
            child = stage_a_prune_mlp(
                copy.deepcopy(parent),
                sp,
                config,
                calib_loader=calib_loader,
                device=device,
            ).to(device)

        for dim in pending_dims:
            spec = specs[dim]
            print(f"[INFO] soft-eval cell sparsity={sp:.2f} dim={dim}")
            extras: Optional[Dict[str, float]] = None
            if str(spec.get("kind", "")).startswith("glue_"):
                score, extras = _eval_glue_classification(
                    child, tokenizer, config, spec, device, soft_dim=dim
                )
            elif spec.get("kind") == "capability":
                score = _eval_capability_soft(
                    child,
                    tokenizer,
                    model_path,
                    config,
                    dim,
                    spec,
                    smoke=smoke,
                )
            else:
                raise ValueError(f"unknown soft dim kind: {spec.get('kind')}")

            scores[lab][dim] = float(score)
            if extras:
                scores_extra[lab][dim] = dict(extras)
            completed.append(soft_cell_key(sp, dim))
            _write_ckpt(
                ckpt_path,
                digest=digest,
                smoke=smoke,
                grid=grid,
                soft_dims=soft_dims,
                completed=completed,
                scores=scores,
                started_at=started_at,
                t0=t0,
                status="in_progress",
                scores_extra=scores_extra,
            )
            extra_msg = ""
            if extras:
                extra_msg = (
                    f" acc={extras.get('accuracy', float('nan')):.4f}"
                    f" bal={extras.get('balanced_accuracy', float('nan')):.4f}"
                    f" maj={extras.get('majority_baseline', float('nan')):.4f}"
                )
            print(f"[OK] soft-eval {lab} {dim}={score:.4f}{extra_msg}")
            _cuda_oom_cleanup()

        if child is not parent:
            del child
            _cuda_oom_cleanup()
            parent.to(device)
            _cuda_oom_cleanup()

    status = "done"
    _write_ckpt(
        ckpt_path,
        digest=digest,
        smoke=smoke,
        grid=grid,
        soft_dims=soft_dims,
        completed=completed,
        scores=scores,
        started_at=started_at,
        t0=t0,
        status=status,
        scores_extra=scores_extra,
    )
    exp_id = str(config.get("experiment_id") or "E1_soft_eval_side")
    summary = {
        "experiment_id": exp_id,
        "status": status,
        "protocol_digest": digest,
        "soft_dims": soft_dims,
        "sparsity_grid": grid,
        "scores": scores,
        "scores_extra": scores_extra,
        "completed_cells": completed,
        "notes": "Side ladder only; formal E1/E2 and Gate A unchanged.",
    }
    write_json(out / "soft_eval_summary.json", summary)
    report = _build_report(
        config, scores, soft_dims, grid, status=status, scores_extra=scores_extra
    )
    write_report(out / "soft_eval_report.md", report)
    exp_l = exp_id.lower()
    if "lite" in exp_l:
        mirror_slug = "soft_eval_lite"
    elif "soft" in exp_l and "side" not in exp_l:
        mirror_slug = "soft_eval_soft"
    else:
        mirror_slug = "soft_eval_side"
    mirror_docs_report("E1", mirror_slug, report)
    print(f"[OK] soft-eval wrote {out}")


if __name__ == "__main__":
    main()
