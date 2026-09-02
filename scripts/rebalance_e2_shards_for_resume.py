#!/usr/bin/env python3
"""Rebalance E2 shard checkpoints for dual-GPU wall-clock balance.

Rewrite:
  shard_gpu0 -> seeds [42, 44]
  shard_gpu1 -> seeds [43]

Migrates records/completed_cells by seed and routes at most one partial per
target shard (by partial.seed). Backs up existing checkpoints and rewrites
config_digest via e2_config_digest. Does not use --fresh / wipe scores.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_a_common import (
    build_e2_checkpoint_payload,
    e2_config_digest,
    load_e2_checkpoint_any,
    load_yaml,
    save_e2_checkpoint,
)

SHARD0 = Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json")
SHARD1 = Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json")
CFG = ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml"


def _cell_key(rec: Dict[str, Any]) -> Tuple[int, float, str]:
    return (
        int(rec["seed"]),
        float(rec["target_sparsity"]),
        str(rec["method"]),
    )


def _completed_key(c: Dict[str, Any]) -> Tuple[int, float, str]:
    return (
        int(c["seed"]),
        float(c["target_sparsity"]),
        str(c["method"]),
    )


def _filter_records(records: Sequence[Dict[str, Any]], seeds: Set[int]) -> List[Dict[str, Any]]:
    return [dict(r) for r in records if int(r["seed"]) in seeds]


def _filter_completed(
    completed: Sequence[Dict[str, Any]], seeds: Set[int]
) -> List[Dict[str, Any]]:
    return [dict(c) for c in completed if int(c["seed"]) in seeds]


def _assert_no_dupes(records: Sequence[Dict[str, Any]], label: str) -> None:
    seen: Set[Tuple[int, float, str]] = set()
    for r in records:
        k = _cell_key(r)
        if k in seen:
            raise SystemExit(f"[ERROR] duplicate record in {label}: {k}")
        seen.add(k)


def _active_partial(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not data:
        return None
    partial = data.get("partial")
    if isinstance(partial, dict) and partial.get("seed") is not None:
        return dict(partial)
    return None


def _assign_partials(
    partials: Sequence[Dict[str, Any]],
    seeds0: Set[int],
    seeds1: Set[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Route each partial to the shard that owns its seed; reject >1 per shard."""
    p0: Optional[Dict[str, Any]] = None
    p1: Optional[Dict[str, Any]] = None
    for partial in partials:
        seed = int(partial["seed"])
        if seed in seeds0:
            if p0 is not None:
                raise SystemExit(
                    "[ERROR] multiple partials for gpu0 seeds "
                    f"{sorted(seeds0)}: refuse rebalance"
                )
            p0 = dict(partial)
        elif seed in seeds1:
            if p1 is not None:
                raise SystemExit(
                    "[ERROR] multiple partials for gpu1 seeds "
                    f"{sorted(seeds1)}: refuse rebalance"
                )
            p1 = dict(partial)
        else:
            raise SystemExit(f"[ERROR] partial seed={seed} not in target shards")
    return p0, p1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    old0 = load_e2_checkpoint_any(SHARD0)
    old1 = load_e2_checkpoint_any(SHARD1)
    if old0 is None or old1 is None:
        raise SystemExit("[ERROR] both shard checkpoints required")

    seeds0 = [42, 44]
    seeds1 = [43]
    set0, set1 = set(seeds0), set(seeds1)

    partials = [p for p in (_active_partial(old0), _active_partial(old1)) if p is not None]
    partial0, partial1 = _assign_partials(partials, set0, set1)

    all_records = [dict(r) for r in (old0.get("records") or [])] + [
        dict(r) for r in (old1.get("records") or [])
    ]
    all_completed = [dict(c) for c in (old0.get("completed_cells") or [])] + [
        dict(c) for c in (old1.get("completed_cells") or [])
    ]
    completed_by: Dict[Tuple[int, float, str], Dict[str, Any]] = {}
    for c in all_completed:
        completed_by[_completed_key(c)] = c
    all_completed = list(completed_by.values())

    _assert_no_dupes(all_records, "merged-all")

    dense = dict(old0.get("dense_capability") or old1.get("dense_capability") or {})
    started = float(old0.get("started_at") or old1.get("started_at") or time.time())
    elapsed = float(old0.get("elapsed_sec") or 0) + float(old1.get("elapsed_sec") or 0)

    dig0 = e2_config_digest(config, smoke=False, seeds=seeds0)
    dig1 = e2_config_digest(config, smoke=False, seeds=seeds1)

    rec0 = _filter_records(all_records, set0)
    rec1 = _filter_records(all_records, set1)
    comp0 = _filter_completed(all_completed, set0)
    comp1 = _filter_completed(all_completed, set1)
    _assert_no_dupes(rec0, "gpu0")
    _assert_no_dupes(rec1, "gpu1")

    leftover = {int(r["seed"]) for r in all_records} - set0 - set1
    if leftover:
        raise SystemExit(f"[ERROR] unexpected seeds in records: {leftover}")

    payload0 = build_e2_checkpoint_payload(
        config_digest=dig0,
        smoke=False,
        dense_capability=dense,
        records=rec0,
        completed_cells=comp0,
        started_at=started,
        elapsed_sec=elapsed,
        status="in_progress",
        partial=partial0,
        imported_oneshot_seed42=True,
    )
    payload1 = build_e2_checkpoint_payload(
        config_digest=dig1,
        smoke=False,
        dense_capability=dense,
        records=rec1,
        completed_cells=comp1,
        started_at=started,
        elapsed_sec=elapsed,
        status="in_progress",
        partial=partial1,
        imported_oneshot_seed42=False,
    )

    print(
        f"[INFO] gpu0 seeds={seeds0} digest={dig0[:12]}... "
        f"records={len(rec0)} completed={len(comp0)} partial={partial0 is not None}"
    )
    print(
        f"[INFO] gpu1 seeds={seeds1} digest={dig1[:12]}... "
        f"records={len(rec1)} completed={len(comp1)} partial={partial1 is not None}"
    )
    if partial0:
        print(
            f"  gpu0 partial seed={partial0.get('seed')} "
            f"t={partial0.get('target_sparsity')} m={partial0.get('method')} "
            f"dims={partial0.get('completed_dimensions')}"
        )
    if partial1:
        print(
            f"  gpu1 partial seed={partial1.get('seed')} "
            f"t={partial1.get('target_sparsity')} m={partial1.get('method')} "
            f"dims={partial1.get('completed_dimensions')}"
        )
    for r in sorted(rec0, key=_cell_key):
        print(f"  gpu0 cell {_cell_key(r)}")
    for r in sorted(rec1, key=_cell_key):
        print(f"  gpu1 cell {_cell_key(r)}")

    if args.dry_run:
        print("[OK] dry-run only; no files written")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak0 = SHARD0.with_name(f"e2_checkpoint.pre_rebalance_{ts}.json")
    bak1 = SHARD1.with_name(f"e2_checkpoint.pre_rebalance_{ts}.json")
    shutil.copy2(SHARD0, bak0)
    shutil.copy2(SHARD1, bak1)
    print(f"[INFO] backed up -> {bak0.name} / {bak1.name}")

    save_e2_checkpoint(SHARD0, payload0)
    save_e2_checkpoint(SHARD1, payload1)
    print(f"[OK] rewrote {SHARD0}")
    print(f"[OK] rewrote {SHARD1}")


if __name__ == "__main__":
    main()
