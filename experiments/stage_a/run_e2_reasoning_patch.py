#!/usr/bin/env python3
"""E2 Reasoning sync after BBH max_gen_toks=1024 protocol lock.

- e1_import oneshot rows: copy Reasoning from patched E1 (same sparsity)
- e2_run oneshot / iterative completed cells: Reasoning-only re-eval
- clear incomplete Reasoning partials so resume re-runs that dim
- migrate config_digest to new e2 digest (gen_kwargs in protocol)

J0: base model only. Requires E1 J2/J3 (Reasoning details with max_gen_toks=1024).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector
from src.experiments.qwen_k5_comparison import count_params
from src.experiments.stage_a_common import (
    E1_CHECKPOINT_VERSION,
    build_e2_checkpoint_payload,
    compute_capability_deltas,
    e2_config_digest,
    incremental_prune_5pct,
    load_model,
    load_yaml,
    prepare_wanda_calibration_loader,
    pruning_eval_protocol_digest,
    save_e2_checkpoint,
    stage_a_prune_mlp,
    write_json,
)

BASE_PATH = "/mnt/data/models/Qwen2.5-3B"


def _require_base_path(model_path: str) -> None:
    if "Instruct" in model_path or "instruct" in model_path:
        raise SystemExit(f"[ERROR] E2 Reasoning patch forbids Instruct model: {model_path}")
    if model_path != BASE_PATH and not model_path.rstrip("/").endswith("Qwen2.5-3B"):
        raise SystemExit(f"[ERROR] E2 expects base path {BASE_PATH}, got {model_path}")


def _reasoning_gk_ok(details: Dict[str, Any]) -> bool:
    r = details.get("Reasoning") or {}
    gk = r.get("gen_kwargs") or {}
    return int(gk.get("max_gen_toks") or 0) == 1024 and gk.get("do_sample") is False


def _load_e1_reasoning_index(e1_path: Path) -> Dict[str, Any]:
    if not e1_path.is_file():
        raise FileNotFoundError(f"E1 checkpoint missing: {e1_path}")
    data = json.loads(e1_path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != E1_CHECKPOINT_VERSION:
        raise ValueError(f"bad E1 version at {e1_path}")
    dense = dict(data.get("dense_capability") or {})
    dense_details = dict(dense.get("details") or {})
    if not _reasoning_gk_ok(dense_details):
        raise SystemExit(
            f"[ERROR] J3 fail: E1 dense Reasoning missing max_gen_toks=1024 at {e1_path}"
        )
    by_sp: Dict[float, Dict[str, Any]] = {}
    for row in data.get("curve") or []:
        sp = float(row["sparsity"])
        details = dict((row.get("capability") or {}).get("details") or row.get("details") or {})
        if not _reasoning_gk_ok(details):
            raise SystemExit(
                f"[ERROR] J3 fail: E1 sparsity={sp} Reasoning missing max_gen_toks=1024"
            )
        by_sp[sp] = {
            "score": (row.get("vector") or {}).get("Reasoning"),
            "details": details.get("Reasoning"),
            "vector": dict(row.get("vector") or {}),
            "delta": dict(row.get("delta") or {}),
        }
    return {"dense": dense, "by_sp": by_sp}


def _eval_reasoning_with_batch_fallback(
    model_path: str,
    run_cfg: Dict[str, Any],
    *,
    model,
    tokenizer,
) -> tuple[Dict[str, Any], bool]:
    """Try batch=8; on CUDA OOM retry 4 then 2. Returns (cap, used_fallback)."""
    batches = (8, 4, 2)
    last_exc: Exception | None = None
    for idx, bs in enumerate(batches):
        cfg = copy.deepcopy(run_cfg)
        cap = cfg.setdefault("evaluation", {}).setdefault("capability", {})
        cap["only_dimensions"] = ["Reasoning"]
        cap["skip_dimensions"] = []
        cap["batch_size"] = bs
        cfg.setdefault("hardware", {})["batch_size"] = bs
        try:
            return (
                eval_capability_vector(model_path, cfg, model=model, tokenizer=tokenizer),
                idx > 0,
            )
        except Exception as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)
            if not is_oom:
                raise
            last_exc = exc
            nxt = batches[idx + 1] if idx + 1 < len(batches) else None
            if nxt is None:
                break
            print(f"[WARNING] batch={bs} OOM; retrying batch={nxt}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    assert last_exc is not None
    raise last_exc


def _patch_record_reasoning(rec: Dict[str, Any], score: Optional[float], reason_detail: Any, dense_vec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    vec = dict(out.get("vector") or {d: None for d in DIM_ORDER})
    vec["Reasoning"] = score
    out["vector"] = vec
    out["delta"] = compute_capability_deltas(dense_vec, vec)
    details = dict(out.get("details") or {})
    details["Reasoning"] = reason_detail
    out["details"] = details
    if "capability" in out:
        cap = dict(out["capability"] or {})
        cap["vector"] = vec
        cdet = dict(cap.get("details") or {})
        cdet["Reasoning"] = reason_detail
        cap["details"] = cdet
        out["capability"] = cap
    return out


def _clear_reasoning_from_partial(partial: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not partial:
        return None
    work = dict(partial)
    completed = list(work.get("completed_dimensions") or [])
    if "Reasoning" in completed:
        completed = [d for d in completed if d != "Reasoning"]
        work["completed_dimensions"] = completed
        vec = dict(work.get("vector") or {})
        vec["Reasoning"] = None
        work["vector"] = vec
        details = dict(work.get("details") or {})
        details.pop("Reasoning", None)
        work["details"] = details
        raw = dict(work.get("raw") or {})
        raw.pop("Reasoning", None)
        work["raw"] = raw
        print(
            f"[INFO] cleared Reasoning from partial "
            f"seed={work.get('seed')} t={work.get('target_sparsity')} m={work.get('method')}"
        )
    # If partial was mid-Reasoning (not yet in completed), still drop any stale Reasoning score.
    elif (work.get("vector") or {}).get("Reasoning") is not None:
        vec = dict(work.get("vector") or {})
        vec["Reasoning"] = None
        work["vector"] = vec
    return work


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Path to e2_checkpoint.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Skip E1 J2 flag check")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma seeds for this shard digest (e.g. 42,43 or 44). Default=all in YAML.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    model_path = str(config["model"]["path"])
    _require_base_path(model_path)

    seed_list = None
    if args.seeds:
        seed_list = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    e1_path = Path(config.get("e1_oneshot_checkpoint") or "")
    j2_flag = Path(e1_path).parent / "e1_reasoning_j2_passed.flag"
    if not args.force and not j2_flag.is_file():
        raise SystemExit(f"[ERROR] missing E1 J2 flag {j2_flag}; finish E1 Reasoning patch first")

    e1_idx = _load_e1_reasoning_index(e1_path)
    dense_from_e1 = e1_idx["dense"]
    by_sp = e1_idx["by_sp"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    new_digest = e2_config_digest(config, smoke=False, seeds=seed_list)
    proto = pruning_eval_protocol_digest(config)
    print(
        f"[INFO] E2 digest migrate {str(data.get('config_digest'))[:12]}... -> {new_digest[:12]}... "
        f"seeds={seed_list or config.get('seeds')}"
    )

    dense_cap = dict(data.get("dense_capability") or {})
    # Sync dense Reasoning from patched E1 (same base model).
    dense_vec = dict(dense_cap.get("vector") or dense_from_e1.get("vector") or {})
    dense_vec["Reasoning"] = (dense_from_e1.get("vector") or {}).get("Reasoning")
    dense_cap["vector"] = dense_vec
    dense_cap["vector_list"] = [dense_vec.get(d) for d in DIM_ORDER]
    ddet = dict(dense_cap.get("details") or {})
    ddet["Reasoning"] = (dense_from_e1.get("details") or {}).get("Reasoning")
    dense_cap["details"] = ddet

    records = [dict(r) for r in (data.get("records") or [])]
    completed_cells = [dict(c) for c in (data.get("completed_cells") or [])]
    partial = data.get("partial")
    if isinstance(partial, dict):
        partial = _clear_reasoning_from_partial(partial)
    else:
        partial = None

    started_at = float(data.get("started_at") or time.time())
    imported = bool(data.get("imported_oneshot_seed42"))
    smoke = bool(data.get("smoke"))

    # Partition work
    import_idxs: List[int] = []
    reeval_idxs: List[int] = []
    for i, rec in enumerate(records):
        src = str(rec.get("source") or "")
        method = str(rec.get("method") or "")
        if src == "e1_import" or (method == "oneshot" and int(rec.get("seed", -1)) == 42):
            import_idxs.append(i)
        else:
            reeval_idxs.append(i)

    for i in import_idxs:
        rec = records[i]
        sp = float(rec["target_sparsity"])
        match = None
        for k, v in by_sp.items():
            if abs(k - sp) < 1e-9:
                match = v
                break
        if match is None or match["score"] is None:
            raise SystemExit(f"[ERROR] E1 missing Reasoning for sparsity={sp}")
        records[i] = _patch_record_reasoning(rec, match["score"], match["details"], dense_vec)
        records[i]["protocol_digest"] = proto
        records[i]["source"] = "e1_import"
        print(f"[INFO] synced e1_import seed={rec.get('seed')} t={sp} R={match['score']}")

    # Save after import sync before long re-evals
    def _save(recs, part) -> None:
        payload = build_e2_checkpoint_payload(
            config_digest=new_digest,
            smoke=smoke,
            dense_capability=dense_cap,
            records=recs,
            completed_cells=completed_cells,
            started_at=started_at,
            elapsed_sec=time.time() - started_at,
            status=str(data.get("status") or "in_progress"),
            partial=part,
            imported_oneshot_seed42=imported,
        )
        save_e2_checkpoint(ckpt_path, payload)

    _save(records, partial)

    if not reeval_idxs:
        print("[OK] E2 Reasoning patch: no re-eval cells")
        write_json(
            ckpt_path.parent / "e2_reasoning_patch_status.json",
            {
                "checkpoint": str(ckpt_path),
                "imported": len(import_idxs),
                "reeval": 0,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        return

    # Group re-eval by seed for calib
    by_seed: Dict[int, List[int]] = {}
    for i in reeval_idxs:
        by_seed.setdefault(int(records[i]["seed"]), []).append(i)

    any_fallback = False
    for seed, idxs in sorted(by_seed.items()):
        cell_config = copy.deepcopy(config)
        cell_config["seed"] = int(seed)
        cell_config.setdefault("dataset", {})["split_seed"] = int(seed)
        cell_config.setdefault("hardware", {})["device"] = args.device
        cell_config.setdefault("evaluation", {}).setdefault("capability", {})["device"] = args.device

        model, tokenizer, device = load_model(cell_config)
        calib_loader = prepare_wanda_calibration_loader(cell_config, tokenizer, device)

        for i in idxs:
            rec = records[i]
            target = float(rec["target_sparsity"])
            method = str(rec["method"])
            print(f"[INFO] E2 re-eval Reasoning seed={seed} t={target:.2f} method={method}")
            if method == "oneshot":
                pruned = stage_a_prune_mlp(
                    copy.deepcopy(model),
                    target,
                    cell_config,
                    calib_loader=calib_loader,
                    device=device,
                ).to(device)
            else:
                pruned, _meta = incremental_prune_5pct(
                    copy.deepcopy(model),
                    target,
                    cell_config,
                    calib_loader=calib_loader,
                    device=device,
                )
            reason_cap, fell = _eval_reasoning_with_batch_fallback(
                model_path, cell_config, model=pruned, tokenizer=tokenizer
            )
            any_fallback = any_fallback or fell
            score = reason_cap["vector"].get("Reasoning")
            detail = (reason_cap.get("details") or {}).get("Reasoning")
            records[i] = _patch_record_reasoning(rec, score, detail, dense_vec)
            records[i]["protocol_digest"] = proto
            if "params" not in records[i] or records[i]["params"] is None:
                records[i]["params"] = count_params(pruned)
            print(f"[INFO] seed={seed} t={target:.2f} {method} R={score} fallback={fell}")
            del pruned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _save(records, partial)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save(records, partial)
    write_json(
        ckpt_path.parent / "e2_reasoning_patch_status.json",
        {
            "checkpoint": str(ckpt_path),
            "imported": len(import_idxs),
            "reeval": len(reeval_idxs),
            "reasoning_batch_fallback": any_fallback,
            "new_digest": new_digest,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"[OK] E2 Reasoning patch wrote {ckpt_path}")


if __name__ == "__main__":
    main()
