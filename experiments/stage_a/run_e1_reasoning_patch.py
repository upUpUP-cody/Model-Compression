#!/usr/bin/env python3
"""E1 Reasoning-only patch: re-eval BBH with max_gen_toks=1024; write back into checkpoints.

Process name must NOT be run_e1_oneshot_curve.py.
Requires E0 Reasoning gate pass unless --force.
J0: base model path only (no Instruct).
J2: dense score must align with Phase1 calib (|delta|<=0.02).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector
from src.experiments.qwen_k5_comparison import count_params
from src.experiments.stage_a_common import (
    E1_CHECKPOINT_VERSION,
    build_e1_checkpoint_payload,
    compute_capability_deltas,
    e1_checkpoint_digest_candidates,
    e1_checkpoint_filename,
    e1_config_digest,
    load_model,
    load_yaml,
    prepare_wanda_calibration_loader,
    save_e1_checkpoint,
    stage_a_prune_mlp,
    write_json,
)

BASE_PATH = "/mnt/data/models/Qwen2.5-3B"
CALIB_JSON = Path("/mnt/data2/results/bbh_maxtok1024_calib/bbh1024_calib.json")
MAX_ABS_DELTA = 0.02


def _parse_sparsities(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None or not str(raw).strip():
        return None
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _require_base_path(model_path: str) -> None:
    if "Instruct" in model_path or "instruct" in model_path:
        raise SystemExit(f"[ERROR] E1 Reasoning patch forbids Instruct model: {model_path}")
    if model_path != BASE_PATH and not model_path.rstrip("/").endswith("Qwen2.5-3B"):
        raise SystemExit(f"[ERROR] E1 Reasoning patch expects base path {BASE_PATH}, got {model_path}")


def _require_e0_gate(gate_path: Path, *, force: bool) -> None:
    if force:
        print("[WARNING] --force: skipping E0 Reasoning gate check (debug only)")
        return
    if not gate_path.is_file():
        raise SystemExit(
            f"[ERROR] E0 Reasoning gate not passed (missing {gate_path}). "
            "Run experiments/stage_a/run_e0_reasoning_gate.py first."
        )
    print(f"[INFO] E0 Reasoning gate OK: {gate_path}")


def _load_ckpt_allow_migrate(path: Path, config: Dict[str, Any], *, smoke: bool) -> Dict[str, Any]:
    """Load E1 checkpoint; allow digest migration from pre-1024 protocol."""
    if not path.is_file():
        raise FileNotFoundError(f"E1 checkpoint missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if int(data.get("version", 0)) != E1_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E1 checkpoint version at {path}")
    primary = e1_config_digest(config, smoke=smoke)
    alts = set(e1_checkpoint_digest_candidates(config, smoke=smoke))
    alts.add(primary)
    stored = str(data.get("config_digest") or "")
    if stored and stored not in alts:
        print(f"[WARNING] E1 digest migrate {stored[:12]}... -> {primary[:12]}... at {path}")
    return data


def _patch_vector_reasoning(row_vector: Dict[str, Any], score: Optional[float]) -> Dict[str, Any]:
    out = dict(row_vector or {d: None for d in DIM_ORDER})
    out["Reasoning"] = score
    return out


def _reasoning_details_1024(details: Optional[Mapping[str, Any]]) -> bool:
    if not details:
        return False
    r = details.get("Reasoning") or {}
    gk = r.get("gen_kwargs") or {}
    return int(gk.get("max_gen_toks") or 0) == 1024 and gk.get("do_sample") is False


def _row_reasoning_is_1024(row: Mapping[str, Any]) -> bool:
    if _reasoning_details_1024(row.get("details")):
        return True
    cap = row.get("capability") or {}
    return _reasoning_details_1024(cap.get("details"))


def _dense_reasoning_is_1024(dense_cap: Mapping[str, Any]) -> bool:
    return _reasoning_details_1024(dense_cap.get("details"))


def _eval_reasoning_with_batch_fallback(
    model_path: str,
    run_cfg: Dict[str, Any],
    *,
    model,
    tokenizer,
) -> tuple[Dict[str, Any], bool]:
    """Try batch=4; on CUDA OOM retry batch=2. Returns (cap, used_fallback)."""
    cfg4 = copy.deepcopy(run_cfg)
    cap = cfg4.setdefault("evaluation", {}).setdefault("capability", {})
    cap["batch_size"] = 4
    cfg4.setdefault("hardware", {})["batch_size"] = 4
    try:
        return (
            eval_capability_vector(model_path, cfg4, model=model, tokenizer=tokenizer),
            False,
        )
    except Exception as exc:
        msg = str(exc).lower()
        is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg
        if not is_oom:
            raise
        print(f"[WARNING] batch=4 OOM; retrying batch=2: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cfg2 = copy.deepcopy(run_cfg)
        cap2 = cfg2.setdefault("evaluation", {}).setdefault("capability", {})
        cap2["batch_size"] = 2
        cfg2.setdefault("hardware", {})["batch_size"] = 2
        return (
            eval_capability_vector(model_path, cfg2, model=model, tokenizer=tokenizer),
            True,
        )


def _save_reasoning_patched_checkpoint(
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
    latest = _load_ckpt_allow_migrate(ckpt_path, cfg, smoke=smoke)
    partial = latest.get("partial")
    status = str(latest.get("status") or fallback_status or "in_progress")
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


def _load_calib_score() -> float:
    if not CALIB_JSON.is_file():
        raise FileNotFoundError(f"missing calib json: {CALIB_JSON}")
    payload = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    return float(payload["score_1024"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_oneshot_curve.yaml"))
    parser.add_argument(
        "--shard-config",
        default=str(ROOT / "configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml"),
    )
    parser.add_argument("--sparsities", default=None, help="Comma list; default=all completed in target ckpt")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--shard-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Skip E0 Reasoning gate file check")
    parser.add_argument(
        "--force-repatch",
        action="store_true",
        help="Re-eval sparsities even if Reasoning gen_kwargs already max_gen_toks=1024",
    )
    parser.add_argument(
        "--e0-gate-flag",
        default="/mnt/data2/results/E0_dense_baseline/e0_reasoning_gate_passed.flag",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-abs-delta", type=float, default=MAX_ABS_DELTA)
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

    def _reason_run_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        run = copy.deepcopy(cfg)
        cap = run.setdefault("evaluation", {}).setdefault("capability", {})
        cap["only_dimensions"] = ["Reasoning"]
        cap["skip_dimensions"] = []
        cap["device"] = args.device
        dim = cap.setdefault("dimensions", {}).setdefault("Reasoning", {})
        gk = dict(dim.get("gen_kwargs") or {})
        gk["max_gen_toks"] = 1024
        gk["do_sample"] = False
        dim["gen_kwargs"] = gk
        run.setdefault("hardware", {})["device"] = args.device
        return run

    model_path = str(main_cfg["model"]["path"])
    _require_base_path(model_path)
    calib_score = _load_calib_score()
    t0 = time.time()
    model, tokenizer, device = load_model(
        {**main_cfg, "hardware": {**main_cfg.get("hardware", {}), "device": args.device}}
    )
    calib_loader = prepare_wanda_calibration_loader(main_cfg, tokenizer, device)

    any_fallback = False
    dense_checked = False

    for label, cfg, ckpt_path in targets:
        if not ckpt_path.is_file():
            print(f"[WARNING] skip {label}: missing {ckpt_path}")
            continue
        data = _load_ckpt_allow_migrate(ckpt_path, cfg, smoke=smoke)
        dense_cap = dict(data.get("dense_capability") or {})
        rows: List[Dict[str, Any]] = [dict(r) for r in (data.get("curve") or [])]
        grid = [float(x) for x in cfg.get("sparsity_grid", [])]
        # Write NEW protocol digest after Reasoning 1024 patch.
        digest = e1_config_digest(cfg, smoke=smoke)
        started_at = float(data.get("started_at") or t0)
        run_cfg = _reason_run_config(cfg)

        if not args.skip_dense:
            if not args.force_repatch and _dense_reasoning_is_1024(dense_cap):
                print(f"[INFO] skip dense ({label}): already Reasoning max_gen_toks=1024")
            else:
                print(f"[INFO] E1 Reasoning patch dense ({label})")
                reason_cap, fell = _eval_reasoning_with_batch_fallback(
                    model_path, run_cfg, model=model, tokenizer=tokenizer
                )
                any_fallback = any_fallback or fell
                dense_vec = _patch_vector_reasoning(
                    dense_cap.get("vector") or {}, reason_cap["vector"].get("Reasoning")
                )
                dense_cap["vector"] = dense_vec
                dense_cap["vector_list"] = [dense_vec.get(d) for d in DIM_ORDER]
                details = dict(dense_cap.get("details") or {})
                details["Reasoning"] = reason_cap.get("details", {}).get("Reasoning")
                dense_cap["details"] = details
                dense_score = dense_cap["vector"].get("Reasoning")
                print(f"[INFO] dense Reasoning={dense_score} fallback={fell}")

                # J2: dense vs calib (check once when main dense is patched)
                if label == "main" or (args.shard_only and not dense_checked):
                    if dense_score is None:
                        raise SystemExit("[ERROR] J2 fail: dense Reasoning is None")
                    delta = abs(float(dense_score) - float(calib_score))
                    print(
                        f"[INFO] J2 dense vs calib: score={dense_score} "
                        f"calib={calib_score} abs_delta={delta:.6f} thr={args.max_abs_delta}"
                    )
                    if delta > float(args.max_abs_delta):
                        fail_flag = Path(cfg["logging"]["output_root"]) / "e1_reasoning_patch_failed.flag"
                        fail_flag.write_text(
                            f"J2 fail abs_delta={delta} score={dense_score} calib={calib_score}\n",
                            encoding="utf-8",
                        )
                        raise SystemExit(
                            f"[ERROR] J2 fail: |dense-calib|={delta} > {args.max_abs_delta}; "
                            "refuse E2 sync"
                        )
                    dense_checked = True
                    Path(cfg["logging"]["output_root"]).joinpath("e1_reasoning_j2_passed.flag").write_text(
                        time.strftime("%Y-%m-%dT%H:%M:%S%z") + f" score={dense_score}\n",
                        encoding="utf-8",
                    )

        completed = [float(r["sparsity"]) for r in rows]
        todo = [s for s in completed if want is None or any(abs(s - w) < 1e-6 for w in want)]
        if not args.force_repatch:
            skipped_1024: List[float] = []
            kept: List[float] = []
            for sp in todo:
                idx = next(i for i, r in enumerate(rows) if abs(float(r["sparsity"]) - sp) < 1e-6)
                if _row_reasoning_is_1024(rows[idx]):
                    skipped_1024.append(sp)
                else:
                    kept.append(sp)
            if skipped_1024:
                print(f"[INFO] {label} skip already-1024 sparsities={skipped_1024}")
            todo = kept
        print(f"[INFO] {label} Reasoning patch sparsities={todo}")

        for sp in todo:
            idx = next(i for i, r in enumerate(rows) if abs(float(r["sparsity"]) - sp) < 1e-6)
            print(f"[INFO] E1 Reasoning patch sparsity={sp:.2f} ({label})")
            child = stage_a_prune_mlp(
                copy.deepcopy(model),
                sp,
                cfg,
                calib_loader=calib_loader,
                device=device,
            ).to(device)
            reason_cap, fell = _eval_reasoning_with_batch_fallback(
                model_path, run_cfg, model=child, tokenizer=tokenizer
            )
            any_fallback = any_fallback or fell
            reason_score = reason_cap["vector"].get("Reasoning")
            row = dict(rows[idx])
            row["vector"] = _patch_vector_reasoning(row.get("vector") or {}, reason_score)
            row["delta"] = compute_capability_deltas(dense_cap["vector"], row["vector"])
            row["params"] = count_params(child)
            prev_cap = dict(row.get("capability") or {})
            prev_cap["vector"] = row["vector"]
            prev_details = dict(prev_cap.get("details") or {})
            prev_details["Reasoning"] = reason_cap.get("details", {}).get("Reasoning")
            prev_cap["details"] = prev_details
            prev_cap["only_dimensions"] = ["Reasoning"]
            # Also keep top-level details if present (import path uses either).
            top_details = dict(row.get("details") or {})
            top_details["Reasoning"] = reason_cap.get("details", {}).get("Reasoning")
            row["details"] = top_details
            row["capability"] = prev_cap
            rows[idx] = row
            del child
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[INFO] sparsity={sp:.2f} Reasoning={reason_score} fallback={fell}")

            _save_reasoning_patched_checkpoint(
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

        _save_reasoning_patched_checkpoint(
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
            Path(cfg["logging"]["output_root"]) / "e1_reasoning_patch_status.json",
            {
                "label": label,
                "patched_sparsities": todo,
                "dense_reasoning": (dense_cap.get("vector") or {}).get("Reasoning"),
                "reasoning_batch_fallback": any_fallback,
                "protocol": "bbh max_gen_toks=1024",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        print(f"[OK] E1 Reasoning patch wrote {ckpt_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OK] E1 Reasoning patch finished in {time.time() - t0:.1f}s fallback={any_fallback}")


if __name__ == "__main__":
    main()
