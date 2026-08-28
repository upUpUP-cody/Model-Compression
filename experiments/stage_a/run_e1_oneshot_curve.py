#!/usr/bin/env python3
"""E1 One-shot sparsity curve (PDF Stage A)."""
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
    build_e1_checkpoint_payload,
    compute_capability_deltas,
    dense_capability_complete,
    detect_capability_cliffs,
    e1_checkpoint_digest_candidates,
    e1_checkpoint_filename,
    e1_config_digest,
    e1_partial_dims_complete,
    e1_required_dimensions,
    ensure_dir,
    filter_e1_sparsity_grid,
    finalize_e1_partial_capability,
    gpu_mem_gb,
    import_e1_checkpoint_seed,
    load_e1_checkpoint,
    load_model,
    load_yaml,
    merge_capability_dim_result,
    merge_e1_curves,
    merge_e1_main_with_shard,
    mirror_docs_report,
    new_e1_partial,
    prepare_wanda_calibration_loader,
    save_e1_checkpoint,
    sparsity_in_completed,
    stage_a_prune_mlp,
    write_json,
    write_report,
)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _plot_curves(rows: List[Dict[str, Any]], figures: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    xs = [r["sparsity"] * 100 for r in rows]
    for dim in DIM_ORDER:
        ys = [r["delta"].get(dim) for r in rows]
        if all(v is None for v in ys):
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Sparsity % (Wanda MLP structured)")
        plt.ylabel(f"Delta {dim} vs base dense")
        plt.title(f"E1 {dim} vs sparsity")
        plt.grid(True, alpha=0.3)
        fig_path = figures / f"performance_vs_sparsity_{dim.lower()}.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        paths[dim] = str(fig_path)
    plt.figure(figsize=(7, 4))
    for dim in ("Math", "Knowledge", "Reasoning"):
        ys = [r["delta"].get(dim) for r in rows]
        if any(v is not None for v in ys):
            plt.plot(xs, ys, marker="o", label=dim)
    plt.xlabel("Sparsity %")
    plt.ylabel("Delta vs base dense (acc dims: negative = worse)")
    plt.title("E1 capability-specific curve (Math/Knowledge/Reasoning)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    combo = figures / "performance_vs_sparsity_capability_specific.png"
    plt.savefig(combo, dpi=120, bbox_inches="tight")
    plt.close()
    paths["capability_specific"] = str(combo)
    return paths


def _capability_skip_dimensions(config: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(resolve_capability_config(config).get("skip_dimensions") or ())


def _build_report(config, dense_cap, rows, cliff_out, figure_paths, elapsed, smoke: bool) -> str:
    dense_vec = dense_cap["vector"]
    skip_dims = list(_capability_skip_dimensions(config))
    header = "| Sparsity | " + " | ".join(DIM_ORDER) + " |"
    sep = "|----------|" + "|".join(["------"] * len(DIM_ORDER)) + "|"
    vec_lines = [header, sep]
    for r in rows:
        vec = r["vector"]
        vec_lines.append(
            "| "
            + f"{r['sparsity']*100:.0f}% | "
            + " | ".join(_fmt(vec.get(d)) for d in DIM_ORDER)
            + " |"
        )
    delta_header = "| Sparsity | " + " | ".join(f"Delta {d}" for d in DIM_ORDER) + " |"
    delta_lines = [delta_header, sep]
    for r in rows:
        delta_lines.append(
            "| "
            + f"{r['sparsity']*100:.0f}% | "
            + " | ".join(_fmt(r["delta"].get(d)) for d in DIM_ORDER)
            + " |"
        )
    cliff_dims = cliff_out.get("cliff_dimensions") or []
    cap_specific = cliff_out.get("capability_specific", False)
    status = "smoke" if smoke else "aligned_not_run" if not rows else "done"
    skip_note = (
        f"| skip_dimensions | {', '.join(skip_dims)} |"
        if skip_dims
        else "| skip_dimensions | （无） |"
    )

    return f"""# E1. One-shot Sparsity Curve

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E1 |
| 目的 | 判断是否存在明显 performance cliff（capability-specific） |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | oneshot Wanda MLP structured prune |
| Sparsity | {", ".join(f"{s*100:.0f}%" for s in config.get("sparsity_grid", []))} |
| Recovery | None |
| Evaluation | 六维 scan（与 E0 相同 limit/seed）；Delta 相对 **base 3B dense**（非 E0 Instruct） |
{skip_note}
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | 明显非线性 / capability-specific degradation |
| status | {status} |

## Dense 参照（base 3B 未剪枝）

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| {" | ".join(_fmt(dense_vec.get(d)) for d in DIM_ORDER)} |

## 各档 P(s) 六维向量

{chr(10).join(vec_lines)}

## 各档 Delta vs dense

{chr(10).join(delta_lines)}

## Cliff 判定（启发式）

- capability-specific 信号：{"是" if cap_specific else "否"}（cliff 维：{", ".join(cliff_dims) if cliff_dims else "无"}）
- Gate A 输入：与 E2 一并判断是否暂缓 Agent
- 缺失 / skip 维不参与 cliff（报告为 n/a）

## 输出图

{chr(10).join(f"- `{p}`" for p in figure_paths.values()) if figure_paths else "- （开跑后生成）"}

## 结论

- Dense 参照为 **base Qwen2.5-3B** 自身未剪枝向量；**不得**与 E0 Instruct 混比。
- 方法为 **Wanda**（gate 权重 × MLP 中间激活校准）；校准集见 config `pruning.calibration`。
- 墙钟（本报告生成时）：{elapsed:.1f}s
"""


def _parse_sparsities_arg(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None or not str(raw).strip():
        return None
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _digest_alternates(config: Dict[str, Any], *, smoke: bool) -> List[str]:
    primary = e1_config_digest(config, smoke=smoke)
    return [d for d in e1_checkpoint_digest_candidates(config, smoke=smoke) if d != primary]


def _load_checkpoint_flexible(
    ckpt_path: Path,
    config: Dict[str, Any],
    *,
    smoke: bool,
) -> Optional[Dict[str, Any]]:
    primary = e1_config_digest(config, smoke=smoke)
    return load_e1_checkpoint(
        ckpt_path,
        primary,
        alternate_digests=_digest_alternates(config, smoke=smoke),
    )


def _merge_shard_checkpoints(
    main_rows: List[Dict[str, Any]],
    shard_dirs: List[Path],
    config: Dict[str, Any],
    *,
    smoke: bool,
    overwrite_sparsities: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    rows = list(main_rows)
    alts = list(e1_checkpoint_digest_candidates(config, smoke=smoke))
    overwrite = list(overwrite_sparsities or [])
    for shard_dir in shard_dirs:
        shard_ckpt = shard_dir / e1_checkpoint_filename(smoke=smoke)
        if not shard_ckpt.is_file():
            raise FileNotFoundError(f"shard checkpoint missing: {shard_ckpt}")
        _, shard_rows = import_e1_checkpoint_seed(shard_ckpt, alternate_digests=alts)
        if overwrite:
            rows = merge_e1_main_with_shard(rows, shard_rows, overwrite_sparsities=overwrite)
        else:
            main_sps = {float(r["sparsity"]) for r in rows}
            novel = [r for r in shard_rows if float(r["sparsity"]) not in main_sps]
            rows = merge_e1_curves(rows, novel)
    return rows


def _write_checkpoint(
    ckpt_path: Path,
    *,
    config_digest: str,
    smoke: bool,
    dense_cap: Dict[str, Any],
    rows: List[Dict[str, Any]],
    grid: List[float],
    started_at: float,
    elapsed_sec: float,
    status: str = "in_progress",
    partial: Optional[Dict[str, Any]] = None,
) -> None:
    payload = build_e1_checkpoint_payload(
        config_digest=config_digest,
        smoke=smoke,
        dense_capability=dense_cap,
        curve=rows,
        grid=grid,
        started_at=started_at,
        elapsed_sec=elapsed_sec,
        status=status,
        partial=partial,
    )
    save_e1_checkpoint(ckpt_path, payload)


def _partial_matches_sparsity(partial: Optional[Mapping[str, Any]], sparsity: float) -> bool:
    if not partial:
        return False
    try:
        return abs(float(partial.get("sparsity")) - float(sparsity)) <= 1e-6
    except (TypeError, ValueError):
        return False


def _eval_sparsity_with_dim_checkpoints(
    *,
    sp: float,
    model: Any,
    tokenizer: Any,
    device: Any,
    model_path: str,
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    calib_loader: Any,
    dense_cap: Mapping[str, Any],
    skip_dims: Tuple[str, ...],
    partial: Optional[Dict[str, Any]],
    ckpt_path: Path,
    use_checkpoint: bool,
    config_digest: str,
    smoke: bool,
    rows: List[Dict[str, Any]],
    full_grid: List[float],
    started_at: float,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Prune once, evaluate dims one-by-one, checkpoint after each dim."""
    if _partial_matches_sparsity(partial, sp):
        work = dict(partial)
        work["vector"] = dict(work.get("vector") or {})
        work["details"] = dict(work.get("details") or {})
        work["raw"] = dict(work.get("raw") or {})
        work["completed_dimensions"] = list(work.get("completed_dimensions") or [])
        done = set(work["completed_dimensions"])
        print(
            f"[INFO] E1 resume partial sparsity={sp:.2f} "
            f"completed_dims={work['completed_dimensions']}"
        )
    else:
        work = new_e1_partial(sp)
        done = set()

    child = stage_a_prune_mlp(
        copy.deepcopy(model),
        sp,
        config,
        calib_loader=calib_loader,
        device=device,
    ).to(device)
    work["params"] = count_params(child)

    required = e1_required_dimensions(skip_dimensions=skip_dims)
    for dim in required:
        if dim in done and work["vector"].get(dim) is not None:
            print(f"[INFO] E1 skip dim={dim} sparsity={sp:.2f} (partial checkpoint)")
            continue
        dim_config = copy.deepcopy(run_config)
        dim_config.setdefault("evaluation", {}).setdefault("capability", {})["only_dimensions"] = [
            dim
        ]
        cap_slice = eval_capability_vector(
            model_path,
            dim_config,
            model=child,
            tokenizer=tokenizer,
        )
        work = merge_capability_dim_result(work, dim, cap_slice)
        done = set(work["completed_dimensions"])
        if use_checkpoint:
            _write_checkpoint(
                ckpt_path,
                config_digest=config_digest,
                smoke=smoke,
                dense_cap=dict(dense_cap),
                rows=rows,
                grid=full_grid,
                started_at=started_at,
                elapsed_sec=time.time() - started_at,
                partial=work,
            )
        print(
            f"[INFO] E1 partial sparsity={sp:.2f} "
            f"dims={','.join(work['completed_dimensions'])}"
        )

    if not e1_partial_dims_complete(work, skip_dimensions=skip_dims):
        raise RuntimeError(
            f"E1 partial incomplete after dim loop sparsity={sp:.2f} "
            f"completed={work.get('completed_dimensions')}"
        )

    cap = finalize_e1_partial_capability(
        work,
        model_path=model_path,
        mode=str((run_config.get("evaluation") or {}).get("capability", {}).get("mode", "scan")),
        seed=int(run_config.get("seed", 42)),
        skip_dimensions=skip_dims,
    )
    delta = compute_capability_deltas(dense_cap["vector"], cap["vector"])
    row = {
        "sparsity": sp,
        "vector": cap["vector"],
        "delta": delta,
        "params": work.get("params"),
        "capability": cap,
    }
    del child
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_oneshot_curve.yaml"))
    parser.add_argument("--smoke", action="store_true", help="limit_override=2; sparsity 30% only")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config + write aligned report skeleton without GPU eval",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore/delete checkpoint and start over",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from checkpoint if present (formal runs only)",
    )
    parser.add_argument(
        "--sparsities",
        default=None,
        help="comma-separated sparsity subset of config sparsity_grid (e.g. 0.3,0.4,0.5)",
    )
    parser.add_argument(
        "--seed-checkpoint",
        default=None,
        help="import dense + completed curve from another checkpoint (shard GPU)",
    )
    parser.add_argument(
        "--merge-from",
        action="append",
        default=None,
        help="merge curve from shard output dir(s) into main; no GPU eval (repeatable)",
    )
    parser.add_argument(
        "--merge-overwrite-sparsities",
        default=None,
        help="comma sparsities to replace from shard during --merge-from (e.g. 0.5,0.6,0.7)",
    )
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    figures = ensure_dir(out / "figures")

    use_checkpoint = not args.dry_run and not args.smoke
    ckpt_path = out / e1_checkpoint_filename(smoke=args.smoke)
    config_digest = e1_config_digest(config, smoke=args.smoke)
    merge_only = bool(args.merge_from) and not args.dry_run and not args.smoke

    if args.fresh and ckpt_path.is_file() and not merge_only:
        ckpt_path.unlink()
        print(f"[INFO] E1 removed checkpoint {ckpt_path}")

    sparsity_subset = _parse_sparsities_arg(args.sparsities)
    merge_overwrite = _parse_sparsities_arg(args.merge_overwrite_sparsities)
    print(
        f"[INFO] E1 oneshot sparsity curve "
        f"(smoke={args.smoke}, dry_run={args.dry_run}, resume={args.resume and use_checkpoint}, "
        f"fresh={args.fresh}, merge_only={merge_only}, sparsities={sparsity_subset}, "
        f"merge_overwrite={merge_overwrite})"
    )
    t0 = time.time()
    started_at = t0
    if torch.cuda.is_available() and not merge_only:
        torch.cuda.reset_peak_memory_stats()

    model_path = str(config["model"]["path"])
    full_grid = [float(x) for x in config.get("sparsity_grid", [0.1, 0.3, 0.5, 0.7])]
    if args.smoke:
        full_grid = [0.30]
    run_grid = filter_e1_sparsity_grid(full_grid, sparsity_subset)

    run_config = copy.deepcopy(config)
    if args.smoke:
        run_config.setdefault("evaluation", {}).setdefault("capability", {})["limit_override"] = 2
    skip_dims = _capability_skip_dimensions(run_config)

    dense_cap: Dict[str, Any] = {"vector": {d: None for d in DIM_ORDER}, "mode": "scan", "seed": 42}
    rows: List[Dict[str, Any]] = []
    partial: Optional[Dict[str, Any]] = None
    figure_paths: Dict[str, str] = {}
    cliff_out: Dict[str, Any] = {"capability_specific": False, "cliff_dimensions": []}

    local_ckpt: Optional[Dict[str, Any]] = None
    if use_checkpoint and args.resume and not args.fresh and not merge_only:
        local_ckpt = _load_checkpoint_flexible(ckpt_path, config, smoke=args.smoke)

    if local_ckpt is not None:
        dense_cap = dict(local_ckpt.get("dense_capability") or dense_cap)
        rows = list(local_ckpt.get("curve") or [])
        raw_partial = local_ckpt.get("partial")
        partial = dict(raw_partial) if isinstance(raw_partial, dict) else None
        started_at = float(local_ckpt.get("started_at", t0))
        print(
            f"[INFO] E1 resumed checkpoint: "
            f"dense_ok={dense_capability_complete(dense_cap, skip_dimensions=skip_dims)} "
            f"completed={[r['sparsity'] for r in rows]} "
            f"partial_sp={None if not partial else partial.get('sparsity')}"
        )
    elif args.seed_checkpoint:
        seed_path = Path(args.seed_checkpoint)
        dense_cap, rows = import_e1_checkpoint_seed(
            seed_path,
            alternate_digests=list(e1_checkpoint_digest_candidates(config, smoke=args.smoke)),
        )
        started_at = t0
        print(
            f"[INFO] E1 seeded from {seed_path}: dense_ok=True "
            f"imported={[r['sparsity'] for r in rows]}"
        )

    if merge_only:
        main_ckpt = _load_checkpoint_flexible(ckpt_path, config, smoke=args.smoke)
        if main_ckpt is not None:
            if not args.seed_checkpoint:
                dense_cap = dict(main_ckpt.get("dense_capability") or dense_cap)
                rows = list(main_ckpt.get("curve") or rows)
            started_at = float(main_ckpt.get("started_at", started_at))
        shard_dirs = [Path(p) for p in args.merge_from]
        rows = _merge_shard_checkpoints(
            rows,
            shard_dirs,
            config,
            smoke=args.smoke,
            overwrite_sparsities=merge_overwrite,
        )
        print(
            f"[INFO] E1 merge-only: {len(rows)} curve rows from main + {len(shard_dirs)} shard(s)"
        )
        cliff_out = detect_capability_cliffs(rows)
        figure_paths = _plot_curves(rows, figures)
        all_done = len(rows) == len(full_grid) and all(
            sparsity_in_completed(float(s), [float(r["sparsity"]) for r in rows]) for s in full_grid
        )
        if use_checkpoint:
            _write_checkpoint(
                ckpt_path,
                config_digest=config_digest,
                smoke=args.smoke,
                dense_cap=dense_cap,
                rows=rows,
                grid=full_grid,
                started_at=started_at,
                elapsed_sec=time.time() - started_at,
                status="done" if all_done else "in_progress",
            )
    elif args.dry_run:
        print("[INFO] E1 dry-run: skip GPU eval; writing aligned skeleton only")
    else:
        if not Path(model_path).joinpath("config.json").exists():
            raise FileNotFoundError(
                f"base model missing at {model_path}; download Qwen/Qwen2.5-3B before running E1"
            )
        model, tokenizer, device = load_model(config)
        calib_loader = prepare_wanda_calibration_loader(config, tokenizer, device)

        if not dense_capability_complete(dense_cap, skip_dimensions=skip_dims):
            print("[INFO] E1 dense base-3B capability scan")
            dense_cap = eval_capability_vector(
                model_path,
                run_config,
                model=model,
                tokenizer=tokenizer,
            )
            if use_checkpoint:
                _write_checkpoint(
                    ckpt_path,
                    config_digest=config_digest,
                    smoke=args.smoke,
                    dense_cap=dense_cap,
                    rows=rows,
                    grid=full_grid,
                    started_at=started_at,
                    elapsed_sec=time.time() - started_at,
                )
        else:
            print("[INFO] E1 skip dense scan (checkpoint)")

        for sp in run_grid:
            if sparsity_in_completed(sp, [float(r["sparsity"]) for r in rows]):
                print(f"[INFO] E1 skip sparsity={sp:.2f} (checkpoint)")
                if _partial_matches_sparsity(partial, sp):
                    partial = None
                continue
            print(f"[INFO] E1 sparsity={sp:.2f}")
            row, partial = _eval_sparsity_with_dim_checkpoints(
                sp=sp,
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_path=model_path,
                config=config,
                run_config=run_config,
                calib_loader=calib_loader,
                dense_cap=dense_cap,
                skip_dims=skip_dims,
                partial=partial,
                ckpt_path=ckpt_path,
                use_checkpoint=use_checkpoint,
                config_digest=config_digest,
                smoke=args.smoke,
                rows=rows,
                full_grid=full_grid,
                started_at=started_at,
            )
            rows.append(row)
            if use_checkpoint:
                _write_checkpoint(
                    ckpt_path,
                    config_digest=config_digest,
                    smoke=args.smoke,
                    dense_cap=dense_cap,
                    rows=rows,
                    grid=full_grid,
                    started_at=started_at,
                    elapsed_sec=time.time() - started_at,
                    partial=None,
                )

        cliff_out = detect_capability_cliffs(rows)
        figure_paths = _plot_curves(rows, figures)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_done = len(rows) == len(full_grid) and all(
            sparsity_in_completed(float(s), [float(r["sparsity"]) for r in rows]) for s in full_grid
        )
        if use_checkpoint and all_done:
            _write_checkpoint(
                ckpt_path,
                config_digest=config_digest,
                smoke=args.smoke,
                dense_cap=dense_cap,
                rows=rows,
                grid=full_grid,
                started_at=started_at,
                elapsed_sec=time.time() - started_at,
                status="done",
            )

    mem = gpu_mem_gb()
    elapsed = time.time() - started_at
    summary = {
        "experiment_id": "E1",
        "title": "One-shot Sparsity Curve",
        "purpose": "判断是否存在明显 performance cliff",
        "model": config["model"].get("spec"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": "wanda",
        "pruning": config.get("pruning"),
        "compression_or_sparsity": full_grid,
        "evaluation": "lm_eval six-dim scan; delta vs base-3B-dense (NOT E0 Instruct)",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device"),
        "priority": "P0",
        "success_criteria": "存在明显非线性 degradation / capability-specific degradation",
        "status": "smoke" if args.smoke else ("dry_run" if args.dry_run else "done"),
        "outputs": {"figures": figure_paths, "cliff_analysis": cliff_out},
        "records": {"dense_capability": dense_cap, "curve": rows},
        "notes": "Wanda MLP structured prune; calibration SST-2 train LM. Dense ref = base 3B unpruned.",
        "resources": {"gpu_memory_gb_peak": mem, "latency_sec": elapsed},
    }
    write_json(out / "e1_summary.json", summary)

    report = _build_report(config, dense_cap, rows, cliff_out, figure_paths, elapsed, smoke=args.smoke)
    write_report(out / "e1_report.md", report)
    if not args.smoke and not args.dry_run:
        mirror_docs_report("E1", "oneshot_sparsity_curve", report)
    print(f"[OK] E1 wrote {out}")


if __name__ == "__main__":
    main()
