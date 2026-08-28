#!/usr/bin/env python3
"""E1 Code-only patch: re-eval HumanEval after protocol fix; write back into checkpoints.

Process name must NOT be run_e1_oneshot_curve.py (merge watcher uses that pgrep).
Requires E0 Code gate pass unless --force.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector
from src.experiments.qwen_k5_comparison import count_params
from src.experiments.stage_a_common import (
    build_e1_checkpoint_payload,
    compute_capability_deltas,
    e1_checkpoint_digest_candidates,
    e1_checkpoint_filename,
    e1_config_digest,
    load_e1_checkpoint,
    load_model,
    load_yaml,
    prepare_wanda_calibration_loader,
    save_e1_checkpoint,
    stage_a_prune_mlp,
    write_json,
)


def _parse_sparsities(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None or not str(raw).strip():
        return None
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_ckpt(path: Path, config: Dict[str, Any], *, smoke: bool) -> Dict[str, Any]:
    primary = e1_config_digest(config, smoke=smoke)
    alts = [d for d in e1_checkpoint_digest_candidates(config, smoke=smoke) if d != primary]
    data = load_e1_checkpoint(path, primary, alternate_digests=alts)
    if data is None:
        raise FileNotFoundError(f"E1 checkpoint missing: {path}")
    return data


def _patch_vector_code(row_vector: Dict[str, Any], code_score: Optional[float]) -> Dict[str, Any]:
    out = dict(row_vector or {d: None for d in DIM_ORDER})
    out["Code"] = code_score
    return out


def _require_e0_gate(gate_path: Path, *, force: bool) -> None:
    if force:
        print("[WARNING] --force: skipping E0 Code gate check")
        return
    if not gate_path.is_file():
        raise SystemExit(
            f"[ERROR] E0 Code gate not passed (missing {gate_path}). "
            "Run experiments/stage_a/run_e0_code_gate.py first."
        )
    print(f"[INFO] E0 Code gate OK: {gate_path}")


def _save_code_patched_checkpoint(
    ckpt_path: Path,
    *,
    cfg: Dict[str, Any],
    smoke: bool,
    digest: str,
    started_at: float,
    dense_cap: Dict[str, Any],
    rows: List[Dict[str, Any]],
    grid: Sequence[float],
    fallback_status: str,
) -> Dict[str, Any]:
    """Write Code patches while preserving concurrent GPU1 `partial` / status."""
    latest = _load_ckpt(ckpt_path, cfg, smoke=smoke)
    partial = latest.get("partial")
    status = str(latest.get("status") or fallback_status or "in_progress")
    # Disk curve first (may include rows we did not patch, e.g. 0.7); our rows win.
    by_sp = {float(r["sparsity"]): dict(r) for r in (latest.get("curve") or [])}
    for r in rows:
        by_sp[float(r["sparsity"])] = dict(r)
    curve_out = [by_sp[sp] for sp in sorted(by_sp)]
    payload = build_e1_checkpoint_payload(
        config_digest=digest,
        smoke=smoke,
        dense_capability=dense_cap,
        curve=curve_out,
        grid=grid,
        started_at=float(latest.get("started_at") or started_at),
        elapsed_sec=time.time() - float(latest.get("started_at") or started_at),
        status=status,
        partial=partial,
    )
    save_e1_checkpoint(ckpt_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_oneshot_curve.yaml"))
    parser.add_argument(
        "--shard-config",
        default=str(ROOT / "configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml"),
        help="Shard config for 50/60/70 checkpoints",
    )
    parser.add_argument("--sparsities", default=None, help="Comma list; default=all completed in target ckpt")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--main-only", action="store_true", help="Only patch main checkpoint")
    parser.add_argument("--shard-only", action="store_true", help="Only patch shard checkpoint")
    parser.add_argument("--force", action="store_true", help="Skip E0 gate file check")
    parser.add_argument(
        "--e0-gate-flag",
        default="/mnt/data2/results/E0_dense_baseline/e0_code_gate_passed.flag",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _require_e0_gate(Path(args.e0_gate_flag), force=args.force)

    main_cfg = load_yaml(Path(args.config))
    shard_cfg = load_yaml(Path(args.shard_config))
    smoke = False
    want = _parse_sparsities(args.sparsities)

    main_out = Path(main_cfg["logging"]["output_root"])
    shard_out = Path(shard_cfg["logging"]["output_root"])
    main_ckpt = main_out / e1_checkpoint_filename(smoke=smoke)
    shard_ckpt = shard_out / e1_checkpoint_filename(smoke=smoke)

    targets: List[tuple[str, Dict[str, Any], Path]] = []
    if not args.shard_only:
        targets.append(("main", main_cfg, main_ckpt))
    if not args.main_only:
        targets.append(("shard", shard_cfg, shard_ckpt))

    # Shared eval config: only Code on GPU0.
    # Clear skip_dimensions so shard YAML skip_dimensions:[Code] does not still skip Code
    # (resolve keeps skips that intersect only_dimensions).
    def _code_run_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        run = copy.deepcopy(cfg)
        cap = run.setdefault("evaluation", {}).setdefault("capability", {})
        cap["only_dimensions"] = ["Code"]
        cap["skip_dimensions"] = []
        cap["device"] = args.device
        run.setdefault("hardware", {})["device"] = args.device
        return run

    model_path = str(main_cfg["model"]["path"])
    t0 = time.time()
    model, tokenizer, device = load_model({**main_cfg, "hardware": {**main_cfg.get("hardware", {}), "device": args.device}})
    calib_loader = prepare_wanda_calibration_loader(main_cfg, tokenizer, device)

    for label, cfg, ckpt_path in targets:
        if not ckpt_path.is_file():
            print(f"[WARNING] skip {label}: missing {ckpt_path}")
            continue
        data = _load_ckpt(ckpt_path, cfg, smoke=smoke)
        dense_cap = dict(data.get("dense_capability") or {})
        rows: List[Dict[str, Any]] = [dict(r) for r in (data.get("curve") or [])]
        grid = [float(x) for x in cfg.get("sparsity_grid", [])]
        digest = str(data.get("config_digest") or e1_config_digest(cfg, smoke=smoke))
        started_at = float(data.get("started_at") or t0)
        run_cfg = _code_run_config(cfg)

        if not args.skip_dense:
            print(f"[INFO] E1 Code patch dense ({label})")
            code_cap = eval_capability_vector(
                model_path,
                run_cfg,
                model=model,
                tokenizer=tokenizer,
            )
            dense_vec = _patch_vector_code(dense_cap.get("vector") or {}, code_cap["vector"].get("Code"))
            dense_cap["vector"] = dense_vec
            dense_cap["vector_list"] = [dense_vec.get(d) for d in DIM_ORDER]
            details = dict(dense_cap.get("details") or {})
            details["Code"] = code_cap.get("details", {}).get("Code")
            dense_cap["details"] = details
            print(f"[INFO] dense Code={dense_cap['vector'].get('Code')}")

        completed = [float(r["sparsity"]) for r in rows]
        todo = [s for s in completed if want is None or any(abs(s - w) < 1e-6 for w in want)]
        print(f"[INFO] {label} Code patch sparsities={todo}")

        for sp in todo:
            idx = next(i for i, r in enumerate(rows) if abs(float(r["sparsity"]) - sp) < 1e-6)
            print(f"[INFO] E1 Code patch sparsity={sp:.2f} ({label})")
            child = stage_a_prune_mlp(
                copy.deepcopy(model),
                sp,
                cfg,
                calib_loader=calib_loader,
                device=device,
            ).to(device)
            code_cap = eval_capability_vector(
                model_path,
                run_cfg,
                model=child,
                tokenizer=tokenizer,
            )
            code_score = code_cap["vector"].get("Code")
            row = dict(rows[idx])
            row["vector"] = _patch_vector_code(row.get("vector") or {}, code_score)
            row["delta"] = compute_capability_deltas(dense_cap["vector"], row["vector"])
            row["params"] = count_params(child)
            # Keep prior capability payload but refresh Code slice when present.
            prev_cap = dict(row.get("capability") or {})
            prev_cap["vector"] = row["vector"]
            prev_details = dict(prev_cap.get("details") or {})
            prev_details["Code"] = code_cap.get("details", {}).get("Code")
            prev_cap["details"] = prev_details
            prev_cap["only_dimensions"] = ["Code"]
            row["capability"] = prev_cap
            rows[idx] = row
            del child
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[INFO] sparsity={sp:.2f} Code={code_score}")

            _save_code_patched_checkpoint(
                ckpt_path,
                cfg=cfg,
                smoke=smoke,
                digest=digest,
                started_at=started_at,
                dense_cap=dense_cap,
                rows=rows,
                grid=grid,
                fallback_status=str(data.get("status") or "in_progress"),
            )

        _save_code_patched_checkpoint(
            ckpt_path,
            cfg=cfg,
            smoke=smoke,
            digest=digest,
            started_at=started_at,
            dense_cap=dense_cap,
            rows=rows,
            grid=grid,
            fallback_status=str(data.get("status") or "in_progress"),
        )
        write_json(
            Path(cfg["logging"]["output_root"]) / "e1_code_patch_status.json",
            {
                "label": label,
                "patched_sparsities": todo,
                "dense_code": (dense_cap.get("vector") or {}).get("Code"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        print(f"[OK] E1 Code patch wrote {ckpt_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OK] E1 Code patch finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
