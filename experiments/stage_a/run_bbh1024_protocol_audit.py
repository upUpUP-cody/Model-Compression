#!/usr/bin/env python3
"""Protocol audit for BBH1024 lock (J3/J4/J5). Exit 0 only when all green.

Writes /mnt/data2/results/bbh_maxtok1024_calib/bbh1024_protocol_locked.flag on success.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_a_common import load_yaml, pruning_eval_protocol_digest

E1_MAIN = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json")
E1_SHARD = Path("/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json")
E2_GPU0 = Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json")
E2_GPU1 = Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json")
E0_FLAG = Path("/mnt/data2/results/E0_dense_baseline/e0_reasoning_gate_passed.flag")
J2_FLAG = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_reasoning_j2_passed.flag")
OUT_DIR = Path("/mnt/data2/results/bbh_maxtok1024_calib")


def _gk_ok(details: Dict[str, Any]) -> bool:
    r = details.get("Reasoning") or {}
    gk = r.get("gen_kwargs") or {}
    return int(gk.get("max_gen_toks") or 0) == 1024 and gk.get("do_sample") is False


def _e1_rows(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, Dict[str, Any]]] = []
    dense = data.get("dense_capability") or {}
    out.append(("dense", dict(dense.get("details") or {})))
    for row in data.get("curve") or []:
        details = dict((row.get("capability") or {}).get("details") or row.get("details") or {})
        out.append((f"sp={row.get('sparsity')}", details))
    return out


def _e2_scan(path: Path) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, Dict[str, Any]]] = []
    dense = data.get("dense_capability") or {}
    out.append(("dense", dict(dense.get("details") or {})))
    for rec in data.get("records") or []:
        label = f"seed={rec.get('seed')} t={rec.get('target_sparsity')} m={rec.get('method')}"
        out.append((label, dict(rec.get("details") or {})))
    return out, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml"),
    )
    args = parser.parse_args()
    e2_cfg = load_yaml(Path(args.config))
    expected_proto = pruning_eval_protocol_digest(e2_cfg)

    errors: List[str] = []
    notes: List[str] = []

    # J0 paths
    e0 = load_yaml(ROOT / "configs/stage_a/e0_dense.yaml")
    e1 = load_yaml(ROOT / "configs/stage_a/e1_oneshot_curve.yaml")
    if "Instruct" not in str(e0["model"]["path"]):
        errors.append(f"J0 E0 path missing Instruct: {e0['model']['path']}")
    if "Instruct" in str(e1["model"]["path"]) or "Instruct" in str(e2_cfg["model"]["path"]):
        errors.append("J0 E1/E2 path must not contain Instruct")
    else:
        notes.append("J0 model isolation OK")

    if not E0_FLAG.is_file():
        errors.append(f"J1 missing {E0_FLAG}")
    else:
        notes.append("J1 E0 Reasoning gate flag OK")

    if not J2_FLAG.is_file():
        errors.append(f"J2 missing {J2_FLAG}")
    else:
        notes.append("J2 E1 dense flag OK")

    for label, path in (("e1_main", E1_MAIN), ("e1_shard", E1_SHARD)):
        if not path.is_file():
            errors.append(f"J3 missing {path}")
            continue
        for row_label, details in _e1_rows(path):
            if not _gk_ok(details):
                errors.append(f"J3 {label} {row_label} Reasoning gen_kwargs not 1024")
        notes.append(f"J3 scanned {label}")

    for label, path in (("e2_gpu0", E2_GPU0), ("e2_gpu1", E2_GPU1)):
        if not path.is_file():
            errors.append(f"J4 missing {path}")
            continue
        rows, data = _e2_scan(path)
        for row_label, details in rows:
            if not _gk_ok(details):
                errors.append(f"J4 {label} {row_label} Reasoning gen_kwargs not 1024")
        for rec in data.get("records") or []:
            if str(rec.get("source")) == "e1_import":
                pd = rec.get("protocol_digest")
                if pd and pd != expected_proto:
                    errors.append(
                        f"J4 {label} import protocol_digest mismatch "
                        f"seed={rec.get('seed')} t={rec.get('target_sparsity')}"
                    )
        notes.append(f"J4 scanned {label}")

    passed = not errors
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": passed,
        "errors": errors,
        "notes": notes,
        "expected_protocol_digest": expected_proto,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (OUT_DIR / "bbh1024_protocol_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if passed:
        flag = OUT_DIR / "bbh1024_protocol_locked.flag"
        flag.write_text(
            time.strftime("%Y-%m-%dT%H:%M:%S%z")
            + "\nJ0-J5 locked; E2 resume allowed\n"
            + "\n".join(notes)
            + "\n",
            encoding="utf-8",
        )
        failed = OUT_DIR / "bbh1024_protocol_lock_failed.flag"
        if failed.is_file():
            failed.unlink()
        print(f"[OK] protocol locked: {flag}")
    else:
        fail = OUT_DIR / "bbh1024_protocol_lock_failed.flag"
        fail.write_text("\n".join(errors) + "\n", encoding="utf-8")
        locked = OUT_DIR / "bbh1024_protocol_locked.flag"
        if locked.is_file():
            locked.unlink()
        print("[ERROR] protocol audit failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
