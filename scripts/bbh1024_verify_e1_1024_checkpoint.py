#!/usr/bin/env python3
"""Read-only verify E1 checkpoint Reasoning BBH1024 protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _reasoning_1024(details: Optional[Dict[str, Any]]) -> bool:
    if not details:
        return False
    r = details.get("Reasoning") or {}
    gk = r.get("gen_kwargs") or {}
    return int(gk.get("max_gen_toks") or 0) == 1024 and gk.get("do_sample") is False


def _row_reasoning_1024(row: Dict[str, Any]) -> bool:
    if _reasoning_1024(row.get("details")):
        return True
    cap = row.get("capability") or {}
    return _reasoning_1024(cap.get("details"))


def verify_checkpoint(path: Path, *, label: str, required_sps: List[float]) -> List[str]:
    errors: List[str] = []
    if not path.is_file():
        return [f"{label}: missing {path}"]
    data = json.loads(path.read_text(encoding="utf-8"))
    dense = data.get("dense_capability") or {}
    if not _reasoning_1024(dense.get("details")):
        errors.append(f"{label}: dense Reasoning not 1024")
    by_sp = {float(r["sparsity"]): r for r in (data.get("curve") or [])}
    for sp in required_sps:
        if sp not in by_sp:
            errors.append(f"{label}: missing sparsity={sp}")
        elif not _row_reasoning_1024(by_sp[sp]):
            errors.append(f"{label}: sparsity={sp} Reasoning not 1024")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json")
    parser.add_argument(
        "--shard",
        default="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json",
    )
    parser.add_argument(
        "--phase",
        choices=("post-stop", "post-chain", "final"),
        default="post-stop",
        help="post-stop: main 0.1-0.3 + shard 0.1/0.5; post-chain: +main 0.4 +shard 0.6/0.7; final: main 0.1-0.7",
    )
    args = parser.parse_args()
    if args.phase == "post-stop":
        main_req = [0.1, 0.2, 0.3]
        shard_req = [0.1, 0.5]
    elif args.phase == "post-chain":
        main_req = [0.1, 0.2, 0.3, 0.4]
        shard_req = [0.1, 0.5, 0.6, 0.7]
    else:
        main_req = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        shard_req = [0.5, 0.6, 0.7]
    errors: List[str] = []
    errors.extend(verify_checkpoint(Path(args.main), label="main", required_sps=main_req))
    errors.extend(verify_checkpoint(Path(args.shard), label="shard", required_sps=shard_req))
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise SystemExit(1)
    print(f"[OK] E1 1024 verify passed phase={args.phase}")


if __name__ == "__main__":
    main()
