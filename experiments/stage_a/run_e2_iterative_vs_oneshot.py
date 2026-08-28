#!/usr/bin/env python3
"""E2 Iterative vs One-shot (PDF Stage A) — formal six-dim capability scan."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector, resolve_capability_config
from src.experiments.qwen_k5_comparison import count_params
from src.experiments.stage_a_common import (
    build_e2_checkpoint_payload,
    cell_is_completed,
    compare_iterative_vs_oneshot,
    compute_capability_deltas,
    e2_cell_key,
    e2_checkpoint_filename,
    e2_config_digest,
    e2_partial_dims_complete,
    e2_partial_matches,
    ensure_dir,
    finalize_e2_partial,
    gate_a_iterative_advantage,
    import_e1_oneshot_rows,
    incremental_prune_5pct,
    load_e2_checkpoint,
    load_e2_checkpoint_any,
    load_model,
    load_yaml,
    merge_e2_capability_dim,
    merge_e2_records,
    mirror_docs_report,
    new_e2_partial,
    prepare_wanda_calibration_loader,
    pruning_eval_protocol_digest,
    save_e2_checkpoint,
    stage_a_prune_mlp,
    write_json,
    write_report,
)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None or not str(raw).strip():
        return None
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _write_ckpt(
    ckpt_path: Path,
    *,
    config_digest: str,
    smoke: bool,
    dense_cap: Dict[str, Any],
    records: List[Dict[str, Any]],
    completed_cells: List[Dict[str, Any]],
    started_at: float,
    imported_oneshot_seed42: bool,
    partial: Optional[Dict[str, Any]] = None,
    status: str = "in_progress",
) -> None:
    payload = build_e2_checkpoint_payload(
        config_digest=config_digest,
        smoke=smoke,
        dense_capability=dense_cap,
        records=records,
        completed_cells=completed_cells,
        started_at=started_at,
        elapsed_sec=time.time() - started_at,
        status=status,
        partial=partial,
        imported_oneshot_seed42=imported_oneshot_seed42,
    )
    save_e2_checkpoint(ckpt_path, payload)


def _eval_capability_with_batch_fallback(
    model_path: str,
    dim_config: Dict[str, Any],
    *,
    model,
    tokenizer,
) -> Dict[str, Any]:
    """Try configured batch (formal=8), then 4, then 2 on CUDA OOM."""
    cap0 = (dim_config.get("evaluation") or {}).get("capability") or {}
    primary = int(cap0.get("batch_size") or dim_config.get("hardware", {}).get("batch_size") or 8)
    cascade: List[int] = []
    for bs in (primary, 8, 4, 2):
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
            nxt = cascade[idx + 1] if idx + 1 < len(cascade) else None
            if nxt is None:
                break
            print(f"[WARNING] E2 batch={bs} OOM; retrying batch={nxt}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    assert last_exc is not None
    raise last_exc


def _eval_cell_dims(
    *,
    model_path: str,
    pruned,
    tokenizer,
    run_config: Dict[str, Any],
    seed: int,
    target: float,
    method: str,
    partial: Optional[Dict[str, Any]],
    dense_cap: Dict[str, Any],
    records: List[Dict[str, Any]],
    completed_cells: List[Dict[str, Any]],
    ckpt_path: Path,
    config_digest: str,
    smoke: bool,
    started_at: float,
    imported_oneshot_seed42: bool,
    use_checkpoint: bool,
    prune_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if e2_partial_matches(partial, seed, target, method):
        work = dict(partial)
        work["vector"] = dict(work.get("vector") or {})
        work["details"] = dict(work.get("details") or {})
        work["raw"] = dict(work.get("raw") or {})
        work["completed_dimensions"] = list(work.get("completed_dimensions") or [])
        print(
            f"[INFO] E2 resume partial seed={seed} target={target:.2f} method={method} "
            f"dims={work['completed_dimensions']}"
        )
    else:
        work = new_e2_partial(seed, target, method)
        if prune_meta is not None:
            work["params"] = prune_meta.get("final_params")
            work["actual_sparsity"] = prune_meta.get("actual_sparsity")
            work["prune_meta"] = prune_meta

    work["params"] = work.get("params") or count_params(pruned)
    done = set(work["completed_dimensions"])

    for dim in DIM_ORDER:
        if dim in done and work["vector"].get(dim) is not None:
            print(f"[INFO] E2 skip dim={dim} seed={seed} target={target:.2f} method={method}")
            continue
        dim_config = copy.deepcopy(run_config)
        # Harness eval seed stays fixed at capability.seed (42); experiment seed only for Wanda.
        dim_config.setdefault("evaluation", {}).setdefault("capability", {})["only_dimensions"] = [
            dim
        ]
        cap_slice = _eval_capability_with_batch_fallback(
            model_path,
            dim_config,
            model=pruned,
            tokenizer=tokenizer,
        )
        work = merge_e2_capability_dim(work, dim, cap_slice)
        if use_checkpoint:
            _write_ckpt(
                ckpt_path,
                config_digest=config_digest,
                smoke=smoke,
                dense_cap=dense_cap,
                records=records,
                completed_cells=completed_cells,
                started_at=started_at,
                imported_oneshot_seed42=imported_oneshot_seed42,
                partial=work,
            )
        print(
            f"[INFO] E2 partial seed={seed} target={target:.2f} method={method} "
            f"dims={','.join(work['completed_dimensions'])}"
        )

    if not e2_partial_dims_complete(work):
        raise RuntimeError(
            f"E2 partial incomplete seed={seed} target={target:.2f} method={method} "
            f"completed={work.get('completed_dimensions')}"
        )

    cap = finalize_e2_partial(
        work,
        model_path=model_path,
        mode=str((run_config.get("evaluation") or {}).get("capability", {}).get("mode", "scan")),
        seed=int((run_config.get("evaluation") or {}).get("capability", {}).get("seed", 42)),
    )
    delta = compute_capability_deltas(dense_cap["vector"], cap["vector"])
    return {
        "seed": int(seed),
        "target_sparsity": float(target),
        "method": str(method),
        "vector": cap["vector"],
        "delta": delta,
        "details": cap.get("details") or work.get("details") or {},
        "params": work.get("params"),
        "actual_sparsity": work.get("actual_sparsity"),
        "prune_meta": work.get("prune_meta"),
        "capability": cap,
        "source": "e2_run",
    }


def _pair_cells(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[int, float], Dict[str, Dict[str, Any]]] = {}
    for r in records:
        key = (int(r["seed"]), float(r["target_sparsity"]))
        by.setdefault(key, {})[str(r["method"])] = dict(r)
    out: List[Dict[str, Any]] = []
    for (seed, target), methods in sorted(by.items()):
        oneshot = methods.get("oneshot")
        iterative = methods.get("iterative")
        if not oneshot or not iterative:
            continue
        cmp = compare_iterative_vs_oneshot(oneshot["vector"], iterative["vector"])
        out.append(
            {
                "seed": seed,
                "target_sparsity": target,
                "oneshot": oneshot,
                "iterative": iterative,
                **cmp,
            }
        )
    return out


def _plot_comparison(pairs: Sequence[Mapping[str, Any]], figures: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    targets = sorted({float(p["target_sparsity"]) for p in pairs})
    if not targets:
        return paths
    for dim in DIM_ORDER:
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.35
        xs = list(range(len(targets)))
        # Mean across seeds per target
        one_means = []
        it_means = []
        for t in targets:
            o_vals = [
                float(p["oneshot"]["vector"][dim])
                for p in pairs
                if abs(float(p["target_sparsity"]) - t) < 1e-9
                and p["oneshot"]["vector"].get(dim) is not None
            ]
            i_vals = [
                float(p["iterative"]["vector"][dim])
                for p in pairs
                if abs(float(p["target_sparsity"]) - t) < 1e-9
                and p["iterative"]["vector"].get(dim) is not None
            ]
            one_means.append(sum(o_vals) / len(o_vals) if o_vals else float("nan"))
            it_means.append(sum(i_vals) / len(i_vals) if i_vals else float("nan"))
        ax.bar([x - width / 2 for x in xs], one_means, width, label="oneshot")
        ax.bar([x + width / 2 for x in xs], it_means, width, label="iterative")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t*100:.0f}%" for t in targets])
        ax.set_xlabel("Target sparsity")
        ax.set_ylabel(dim)
        ax.set_title(f"E2 {dim}: oneshot vs iterative (mean over seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = figures / f"iterative_vs_oneshot_{dim.lower()}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths[dim] = str(path)
    return paths


def _build_report(
    config: Mapping[str, Any],
    dense_cap: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    figure_paths: Mapping[str, str],
    elapsed: float,
    smoke: bool,
) -> str:
    steps = float(config.get("incremental_step_sparsity", 0.05))
    seeds = config.get("seeds") or [config.get("seed", 42)]
    targets = config.get("target_sparsities") or []
    status = "smoke" if smoke else ("done" if pairs else "aligned_not_run")

    table = [
        "| Seed | Target | One-shot PPL | Iterative PPL | Main wins (iter) | Cell winner |",
        "|------|--------|--------------|---------------|------------------|-------------|",
    ]
    for p in pairs:
        o = p["oneshot"]["vector"]
        i = p["iterative"]["vector"]
        w = "iterative" if p["cell_iterative_win"] else "oneshot/tie"
        table.append(
            f"| {p['seed']} | {p['target_sparsity']*100:.0f}% | {_fmt(o.get('PPL'))} | "
            f"{_fmt(i.get('PPL'))} | {p['main_dim_iterative_wins']}/4 | {w} |"
        )

    dim_lines = [
        "| Seed | Target | " + " | ".join(DIM_ORDER) + " |",
        "|------|--------|" + "|".join(["------"] * len(DIM_ORDER)) + "|",
    ]
    for p in pairs:
        winners = p.get("winner_per_dim") or {}
        dim_lines.append(
            f"| {p['seed']} | {p['target_sparsity']*100:.0f}% | "
            + " | ".join(winners.get(d, "n/a") for d in DIM_ORDER)
            + " |"
        )

    dense_vec = dense_cap.get("vector") or {}
    return f"""# E2. Iterative vs One-shot Compression

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E2 |
| 目的 | 验证逐步压缩是否优于一次性压缩 |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | Wanda；Baseline A One-shot；Baseline B {steps*100:.0f}% incremental；Recovery **None** |
| Target sparsity | {", ".join(f"{float(t)*100:.0f}%" for t in targets)} |
| Evaluation | 六维 scan（与 E1 相同 limit/seed）；Delta vs **base 3B dense** |
| Seeds | {seeds} |
| GPU | dual multi-process by seed (see launch_e2_dual.sh) |
| 优先级 | P0 |
| 成功条件 | iterative 稳定优于 one-shot（主四维 ≥3/4 且 ≥{config.get("success_min_wins", 7)}/9 cells） |
| status | {status} |

## Dense 参照（base 3B，来自 E1）

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| {" | ".join(_fmt(dense_vec.get(d)) for d in DIM_ORDER)} |

## Cell 对照（PPL + Gate）

{chr(10).join(table)}

## 逐维 winner（iterative / oneshot / tie）

{chr(10).join(dim_lines)}

## Gate A（E2 半）

- iterative cell wins: **{gate.get("iterative_cell_wins")}/{gate.get("total_cells")}**（阈值 ≥{gate.get("min_wins")}）
- {"**通过**：支持 iterative advantage。" if gate.get("passed") else "**未通过**：不支持「必须做 Agent」的 iterative 前提。"}
- 与 E1（capability-specific frontier）合判 Gate A。

## 输出图

{chr(10).join(f"- `{p}`" for p in figure_paths.values()) if figure_paths else "- （开跑后生成）"}

## 结论

- Dense / seed=42 oneshot 来自 E1 checkpoint；seed 43/44 oneshot 在 E2 内重剪。
- Incremental：累计 +{steps*100:.0f}pp 绝对 sparsity；remaining-relative Wanda；无 recovery。
- 墙钟（本报告生成时）：{elapsed:.1f}s
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from checkpoint if present (default: on)",
    )
    parser.add_argument("--seeds", default=None, help="comma-separated experiment seeds")
    parser.add_argument("--targets", default=None, help="comma-separated target sparsities")
    parser.add_argument("--output-root", default=None, help="override logging.output_root")
    parser.add_argument(
        "--merge-from",
        action="append",
        default=None,
        help="merge records from shard dir(s); no GPU (repeatable)",
    )
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    if args.output_root:
        config.setdefault("logging", {})["output_root"] = args.output_root
    out = ensure_dir(Path(config["logging"]["output_root"]))
    figures = ensure_dir(out / "figures")

    seeds = _parse_int_list(args.seeds) or [int(s) for s in config.get("seeds", [42])]
    targets = _parse_float_list(args.targets) or [float(t) for t in config.get("target_sparsities", [])]
    if args.smoke:
        seeds = [42]
        targets = [0.40]

    use_checkpoint = True  # smoke writes e2_checkpoint_smoke.json; formal writes e2_checkpoint.json
    ckpt_path = out / e2_checkpoint_filename(smoke=args.smoke)
    config_digest = e2_config_digest(config, smoke=args.smoke, seeds=seeds)
    merge_only = bool(args.merge_from)

    if args.fresh and ckpt_path.is_file() and not merge_only:
        ckpt_path.unlink()
        print(f"[INFO] E2 removed checkpoint {ckpt_path}")

    print(
        f"[INFO] E2 iterative vs oneshot "
        f"(smoke={args.smoke}, resume={args.resume and use_checkpoint}, "
        f"fresh={args.fresh}, merge_only={merge_only}, seeds={seeds}, targets={targets})"
    )
    print(f"[INFO] E2 protocol_digest={pruning_eval_protocol_digest(config)}")
    t0 = time.time()
    started_at = t0

    run_config = copy.deepcopy(config)
    # Keep capability.seed=42 for harness; experiment seed applied per-cell via config.seed / split_seed.
    if args.smoke:
        run_config.setdefault("evaluation", {}).setdefault("capability", {})["limit_override"] = 2

    dense_cap: Dict[str, Any] = {"vector": {d: None for d in DIM_ORDER}, "mode": "scan", "seed": 42}
    records: List[Dict[str, Any]] = []
    completed_cells: List[Dict[str, Any]] = []
    partial: Optional[Dict[str, Any]] = None
    imported_oneshot_seed42 = False

    if merge_only:
        shard_record_sets: List[List[Dict[str, Any]]] = []
        main_ckpt = load_e2_checkpoint_any(ckpt_path) if ckpt_path.is_file() else None
        if main_ckpt:
            dense_cap = dict(main_ckpt.get("dense_capability") or dense_cap)
            shard_record_sets.append([dict(r) for r in (main_ckpt.get("records") or [])])
            started_at = float(main_ckpt.get("started_at", t0))
            imported_oneshot_seed42 = bool(main_ckpt.get("imported_oneshot_seed42"))
        for shard_dir in args.merge_from:
            sp = Path(shard_dir) / e2_checkpoint_filename(smoke=False)
            data = load_e2_checkpoint_any(sp)
            if data is None:
                raise FileNotFoundError(f"E2 shard checkpoint missing: {sp}")
            if not dense_cap.get("vector") or all(
                dense_cap["vector"].get(d) is None for d in DIM_ORDER
            ):
                dense_cap = dict(data.get("dense_capability") or dense_cap)
            shard_record_sets.append([dict(r) for r in (data.get("records") or [])])
            imported_oneshot_seed42 = imported_oneshot_seed42 or bool(
                data.get("imported_oneshot_seed42")
            )
        # Flexible merge: allow overlapping only if identical — use require_disjoint
        records = merge_e2_records(*shard_record_sets, require_disjoint=True)
        completed_cells = [
            e2_cell_key(int(r["seed"]), float(r["target_sparsity"]), str(r["method"]))
            for r in records
        ]
        pairs = _pair_cells(records)
        min_wins = int(config.get("success_min_wins", 7))
        gate = gate_a_iterative_advantage(pairs, min_wins=min_wins)
        figure_paths = _plot_comparison(pairs, figures)
        elapsed = time.time() - started_at
        summary = {
            "experiment_id": "E2",
            "title": "Iterative vs One-shot Compression",
            "purpose": "验证逐步压缩是否优于一次性压缩",
            "model": config["model"].get("spec"),
            "pdf_model": config["model"].get("pdf_model"),
            "spec": config["model"].get("spec"),
            "method": "wanda; oneshot vs 5% incremental; Recovery None",
            "pruning": config.get("pruning"),
            "compression_or_sparsity": targets,
            "evaluation": "lm_eval six-dim scan; delta vs base-3B-dense (NOT E0 Instruct)",
            "evaluation_protocol": resolve_capability_config(config),
            "seeds": seeds,
            "gpu": "dual-shard-merge",
            "priority": "P0",
            "success_criteria": "iterative 稳定优于 one-shot",
            "status": "done",
            "outputs": {
                "pairs": pairs,
                "gate_a": gate,
                "figures": figure_paths,
            },
            "records": {
                "dense_capability": dense_cap,
                "comparison": records,
                "pairs": pairs,
            },
            "config_digest": config_digest,
            "protocol_digest": pruning_eval_protocol_digest(config),
        }
        write_json(out / "e2_summary.json", summary)
        report = _build_report(
            config, dense_cap, pairs, gate, figure_paths, elapsed, smoke=False
        )
        write_report(out / "e2_report.md", report)
        mirror_docs_report("E2", "iterative_vs_oneshot", report)
        _write_ckpt(
            ckpt_path,
            config_digest=config_digest,
            smoke=False,
            dense_cap=dense_cap,
            records=records,
            completed_cells=completed_cells,
            started_at=started_at,
            imported_oneshot_seed42=imported_oneshot_seed42,
            status="done",
        )
        print(f"[OK] E2 merge-only wrote {out}")
        return

    local_ckpt = None
    if use_checkpoint and args.resume and not args.fresh:
        # Seed-subset digest: use digest for this process's seeds
        local_ckpt = load_e2_checkpoint(ckpt_path, config_digest)

    if local_ckpt is not None:
        dense_cap = dict(local_ckpt.get("dense_capability") or dense_cap)
        records = [dict(r) for r in (local_ckpt.get("records") or [])]
        completed_cells = [dict(c) for c in (local_ckpt.get("completed_cells") or [])]
        raw_partial = local_ckpt.get("partial")
        partial = dict(raw_partial) if isinstance(raw_partial, dict) else None
        started_at = float(local_ckpt.get("started_at", t0))
        imported_oneshot_seed42 = bool(local_ckpt.get("imported_oneshot_seed42"))
        partial_desc = None
        if partial:
            partial_desc = e2_cell_key(
                int(partial["seed"]),
                float(partial["target_sparsity"]),
                str(partial["method"]),
            )
        print(
            f"[INFO] E2 resumed: completed_cells={len(completed_cells)} "
            f"records={len(records)} partial={partial_desc}"
        )

    # Import E1 dense + seed42 oneshot when needed
    e1_path = Path(config.get("e1_oneshot_checkpoint") or "")
    need_import = (42 in seeds) and (
        not imported_oneshot_seed42
        or any(
            not cell_is_completed(completed_cells, 42, t, "oneshot") for t in targets
        )
    )
    if need_import or not dense_cap.get("vector") or dense_cap["vector"].get("PPL") is None:
        if not e1_path.is_file():
            raise FileNotFoundError(f"E1 checkpoint required at {e1_path}")
        dense_cap, imported_rows = import_e1_oneshot_rows(e1_path, targets, e2_config=config)
        imported_oneshot_seed42 = True
        for row in imported_rows:
            if 42 not in seeds:
                break
            t = float(row["sparsity"])
            if cell_is_completed(completed_cells, 42, t, "oneshot"):
                continue
            rec = {
                "seed": 42,
                "target_sparsity": t,
                "method": "oneshot",
                "vector": row["vector"],
                "delta": row.get("delta")
                or compute_capability_deltas(dense_cap["vector"], row["vector"]),
                "details": row.get("details") or {},
                "params": row.get("params"),
                "actual_sparsity": t,
                "source": "e1_import",
            }
            records.append(rec)
            completed_cells.append(e2_cell_key(42, t, "oneshot"))
            print(f"[INFO] E2 imported oneshot seed=42 target={t:.2f} from E1")
        if use_checkpoint:
            _write_ckpt(
                ckpt_path,
                config_digest=config_digest,
                smoke=args.smoke,
                dense_cap=dense_cap,
                records=records,
                completed_cells=completed_cells,
                started_at=started_at,
                imported_oneshot_seed42=True,
                partial=partial,
            )

    model_path = str(config["model"]["path"])
    if not Path(model_path).joinpath("config.json").exists():
        raise FileNotFoundError(f"base model missing at {model_path}")

    # Determine remaining work
    work_items: List[Tuple[int, float, str]] = []
    for seed in seeds:
        for target in targets:
            for method in ("oneshot", "iterative"):
                if seed == 42 and method == "oneshot":
                    continue  # imported
                if cell_is_completed(completed_cells, seed, target, method):
                    continue
                work_items.append((seed, target, method))

    if not work_items and (partial is None or e2_partial_dims_complete(partial)):
        print("[INFO] E2 nothing left to run for this seed/target subset")
    else:
        # Group by seed to reload calib with correct split_seed
        for seed in seeds:
            seed_items = [w for w in work_items if w[0] == seed]
            # Also continue partial for this seed
            if partial and int(partial.get("seed", -1)) == seed:
                key = (
                    int(partial["seed"]),
                    float(partial["target_sparsity"]),
                    str(partial["method"]),
                )
                if key not in seed_items and not e2_partial_dims_complete(partial):
                    seed_items.insert(0, key)

            if not seed_items:
                continue

            cell_config = copy.deepcopy(run_config)
            cell_config["seed"] = int(seed)
            cell_config.setdefault("dataset", {})["split_seed"] = int(seed)

            model, tokenizer, device = load_model(cell_config)
            calib_loader = prepare_wanda_calibration_loader(cell_config, tokenizer, device)

            for seed_i, target, method in seed_items:
                if cell_is_completed(completed_cells, seed_i, target, method):
                    continue
                print(f"[INFO] E2 cell seed={seed_i} target={target:.2f} method={method}")

                prune_meta = None
                if method == "oneshot":
                    pruned = stage_a_prune_mlp(
                        copy.deepcopy(model),
                        float(target),
                        cell_config,
                        calib_loader=calib_loader,
                        device=device,
                    ).to(device)
                    prune_meta = {
                        "dense_params": count_params(model),
                        "final_params": count_params(pruned),
                        "target_sparsity": float(target),
                        "actual_sparsity": 1.0
                        - count_params(pruned) / max(count_params(model), 1),
                        "stages": [],
                    }
                else:
                    pruned, prune_meta = incremental_prune_5pct(
                        copy.deepcopy(model),
                        float(target),
                        cell_config,
                        calib_loader=calib_loader,
                        device=device,
                    )

                rec = _eval_cell_dims(
                    model_path=model_path,
                    pruned=pruned,
                    tokenizer=tokenizer,
                    run_config=cell_config,
                    seed=seed_i,
                    target=float(target),
                    method=method,
                    partial=partial if e2_partial_matches(partial, seed_i, target, method) else None,
                    dense_cap=dense_cap,
                    records=records,
                    completed_cells=completed_cells,
                    ckpt_path=ckpt_path,
                    config_digest=config_digest,
                    smoke=args.smoke,
                    started_at=started_at,
                    imported_oneshot_seed42=imported_oneshot_seed42,
                    use_checkpoint=use_checkpoint,
                    prune_meta=prune_meta,
                )
                # Replace any incomplete same cell
                records = [
                    r
                    for r in records
                    if not (
                        int(r["seed"]) == seed_i
                        and abs(float(r["target_sparsity"]) - float(target)) < 1e-9
                        and str(r["method"]) == method
                    )
                ]
                records.append(rec)
                completed_cells.append(e2_cell_key(seed_i, target, method))
                partial = None
                del pruned
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if use_checkpoint:
                    _write_ckpt(
                        ckpt_path,
                        config_digest=config_digest,
                        smoke=args.smoke,
                        dense_cap=dense_cap,
                        records=records,
                        completed_cells=completed_cells,
                        started_at=started_at,
                        imported_oneshot_seed42=imported_oneshot_seed42,
                        partial=None,
                    )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pairs = _pair_cells(records)
    min_wins = int(config.get("success_min_wins", 7))
    gate = gate_a_iterative_advantage(pairs, min_wins=min_wins)
    figure_paths = _plot_comparison(pairs, figures)
    elapsed = time.time() - started_at

    # Shard may be partial (subset of seeds) — status done when no pending for this subset
    pending = [
        (s, t, m)
        for s in seeds
        for t in targets
        for m in ("oneshot", "iterative")
        if not (s == 42 and m == "oneshot")
        and not cell_is_completed(completed_cells, s, t, m)
    ]
    status = "done" if not pending else "in_progress"

    summary = {
        "experiment_id": "E2",
        "title": "Iterative vs One-shot Compression",
        "purpose": "验证逐步压缩是否优于一次性压缩",
        "model": config["model"].get("spec"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": "wanda; oneshot vs 5% incremental; Recovery None",
        "pruning": config.get("pruning"),
        "compression_or_sparsity": targets,
        "evaluation": "lm_eval six-dim scan; delta vs base-3B-dense (NOT E0 Instruct)",
        "evaluation_protocol": {
            "capability_seed": int(
                (config.get("evaluation") or {}).get("capability", {}).get("seed", 42)
            ),
            "scan_limits": dict(
                resolve_capability_config(config).get("scan_limits") or {}
            ),
            "protocol_digest": pruning_eval_protocol_digest(config),
        },
        "seeds": seeds,
        "gpu": config.get("hardware", {}).get("device"),
        "priority": "P0",
        "success_criteria": "iterative 稳定优于 one-shot",
        "status": status,
        "outputs": {"pairs": pairs, "gate_a": gate, "figures": figure_paths},
        "records": {
            "dense_capability": dense_cap,
            "comparison": records,
            "pairs": pairs,
        },
        "config_digest": config_digest,
    }
    write_json(out / "e2_summary.json", summary)
    report = _build_report(config, dense_cap, pairs, gate, figure_paths, elapsed, smoke=args.smoke)
    write_report(out / "e2_report.md", report)
    if status == "done" and not args.smoke and set(seeds) >= {42, 43, 44}:
        mirror_docs_report("E2", "iterative_vs_oneshot", report)

    if use_checkpoint:
        _write_ckpt(
            ckpt_path,
            config_digest=config_digest,
            smoke=args.smoke,
            dense_cap=dense_cap,
            records=records,
            completed_cells=completed_cells,
            started_at=started_at,
            imported_oneshot_seed42=imported_oneshot_seed42,
            status=status,
        )
    print(f"[OK] E2 wrote {out} status={status} cells={len(pairs)} gate={gate}")


if __name__ == "__main__":
    main()
