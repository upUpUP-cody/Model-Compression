"""Build post-run compression reports (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compression_harness.paths import EXPERIMENTS_DIR

SCORE_DIMS = (
    "PPL",
    "Math",
    "Knowledge",
    "Reasoning",
    "Instruction",
    "Code",
    "primary",
)

# Phase B primary metric maps into Reasoning slot (hellaswag).
PRIMARY_DIM = "Reasoning"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def measure_dir_bytes(path: str | Path | None) -> int | None:
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    total = 0
    for f in root.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def measure_dense_weight_bytes(model_path: str | Path | None) -> int | None:
    """Sum weight shards (*.safetensors / *.bin / *.pt) under an HF model dir."""
    if not model_path:
        return None
    root = Path(model_path)
    if not root.exists():
        return None
    total = 0
    found = False
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        for f in root.glob(pat):
            if f.is_file():
                total += f.stat().st_size
                found = True
    if found:
        return total
    return measure_dir_bytes(root)


def _empty_score_vector() -> dict[str, float | None]:
    return {d: None for d in SCORE_DIMS}


def _recipe_weight_bits(artifact: Mapping[str, Any]) -> int | None:
    metrics = artifact.get("metrics") or {}
    exec_m = metrics.get("exec") or {}
    # Prefer bits recorded on artifact; fall back to common INT8 default if method says so
    for key in ("weight_bits",):
        if key in artifact and artifact[key] is not None:
            return int(artifact[key])
    size = metrics.get("size") or {}
    if size.get("weight_bits") is not None:
        return int(size["weight_bits"])
    method = str(exec_m.get("method") or "")
    if "int8" in method.lower():
        return 8
    if "int4" in method.lower() or "4bit" in method.lower():
        return 4
    return None


def _timing_from_artifact(artifact: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = artifact.get("metrics") or {}
    timing = metrics.get("timing") or artifact.get("timing") or {}
    return {
        "compress": _as_float_or_none(timing.get("compress_sec")),
        "eval": _as_float_or_none(timing.get("eval_sec")),
        "total": _as_float_or_none(timing.get("step_sec")),
    }


def _as_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _step_from_artifact(
    artifact: Mapping[str, Any],
    step_index: int,
    dense_size: int | None,
) -> dict[str, Any]:
    metrics = artifact.get("metrics") or {}
    exec_m = metrics.get("exec") or {}
    eval_m = metrics.get("eval") or {}
    size_m = metrics.get("size") or {}
    size_bytes = size_m.get("compressed_bytes")
    if size_bytes is None and artifact.get("compressed_dir"):
        size_bytes = measure_dir_bytes(artifact.get("compressed_dir"))
    size_bytes = int(size_bytes) if size_bytes is not None else None
    ratio = None
    if dense_size and size_bytes:
        ratio = float(dense_size) / float(size_bytes)

    recipe_id = (
        exec_m.get("recipe_id")
        or Path(str(artifact.get("recipe_ref") or "unknown")).stem
    )
    return {
        "step_index": step_index,
        "run_id": artifact.get("run_id"),
        "recipe_id": recipe_id,
        "backend": exec_m.get("backend"),
        "method": exec_m.get("method"),
        "weight_bits": _recipe_weight_bits(artifact),
        "duration_sec": _timing_from_artifact(artifact),
        "size_bytes": size_bytes,
        "size_ratio_vs_dense": ratio,
        "primary_score": _as_float_or_none(artifact.get("score") or eval_m.get("score")),
        "baseline_score": _as_float_or_none(
            artifact.get("baseline_score") or eval_m.get("baseline_score")
        ),
        "relative_drop": _as_float_or_none(
            artifact.get("relative_drop") or eval_m.get("relative_drop")
        ),
        "near_lossless_ok": artifact.get("near_lossless_ok"),
        "status": artifact.get("status"),
    }


def _scores_comparison(
    history: Sequence[Mapping[str, Any]],
    accepted: Mapping[str, Any] | None,
) -> dict[str, Any]:
    dense = _empty_score_vector()
    compressed = _empty_score_vector()
    ref = accepted or (history[-1] if history else None)
    primary_task = None
    if ref:
        eval_m = (ref.get("metrics") or {}).get("eval") or {}
        primary_task = eval_m.get("task")
        base = _as_float_or_none(ref.get("baseline_score") or eval_m.get("baseline_score"))
        score = _as_float_or_none(ref.get("score") or eval_m.get("score"))
        dense[PRIMARY_DIM] = base
        dense["primary"] = base
        compressed[PRIMARY_DIM] = score
        compressed["primary"] = score
        # Future: fill other dims from metrics.eval.vector if present
        vector = eval_m.get("vector") or {}
        for dim in SCORE_DIMS:
            if dim in vector and vector[dim] is not None:
                compressed[dim] = _as_float_or_none(vector[dim])
        base_vec = eval_m.get("baseline_vector") or {}
        for dim in SCORE_DIMS:
            if dim in base_vec and base_vec[dim] is not None:
                dense[dim] = _as_float_or_none(base_vec[dim])

    deltas: dict[str, float | None] = {}
    for dim in SCORE_DIMS:
        d = dense.get(dim)
        c = compressed.get(dim)
        if d is None or c is None:
            deltas[dim] = None
        else:
            deltas[dim] = float(c) - float(d)

    return {
        "dense": dense,
        "compressed": compressed,
        "deltas": deltas,
        "primary_task": primary_task,
        "primary_dim": PRIMARY_DIM,
    }


def build_report(
    goal_doc: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, Any]],
    *,
    model_ref: str,
    stop_reason: str | None,
    goal_ref: str = "",
    dense_size: int | None = None,
    baseline_sec: float | None = None,
    status: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if dense_size is None:
        dense_size = measure_dense_weight_bytes(model_ref)

    steps = [
        _step_from_artifact(art, i, dense_size) for i, art in enumerate(history, start=1)
    ]
    accepted = next((a for a in history if a.get("status") == "accepted"), None)
    if accepted is None and history:
        # Prefer near_lossless_ok
        accepted = next((a for a in history if a.get("near_lossless_ok")), None)

    final_size = None
    if accepted:
        final_size = (accepted.get("metrics") or {}).get("size", {}).get("compressed_bytes")
        if final_size is None:
            final_size = measure_dir_bytes(accepted.get("compressed_dir"))
    elif steps:
        final_size = steps[-1].get("size_bytes")

    compression_ratio = None
    reduction_pct = None
    if dense_size and final_size:
        compression_ratio = float(dense_size) / float(final_size)
        reduction_pct = 100.0 * (1.0 - float(final_size) / float(dense_size))

    if status is None:
        status = "accepted" if accepted else ("failed" if history else "partial")

    g = (goal_doc or {}).get("goal", goal_doc) or {}
    primary_metric = (g.get("evaluation") or {}).get("primary_metric", "acc_norm")

    missing_timing = any(
        (s.get("duration_sec") or {}).get("total") is None for s in steps
    )
    if missing_timing and not notes:
        notes = "Historical run(s) lack timing; duration_sec fields may be null."

    report_id = f"report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    scores = _scores_comparison(history, accepted)

    final_block = {
        "accepted_run_id": accepted.get("run_id") if accepted else None,
        "dense_size_bytes": dense_size,
        "compressed_size_bytes": int(final_size) if final_size is not None else None,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": reduction_pct,
        "primary_metric": primary_metric,
        "baseline_score": scores["dense"].get("primary"),
        "final_score": scores["compressed"].get("primary"),
        "relative_drop": _as_float_or_none(accepted.get("relative_drop")) if accepted else None,
    }

    return {
        "report_id": report_id,
        "created_at": _utc_now(),
        "goal_ref": goal_ref or str((accepted or {}).get("goal_ref") or ""),
        "model_ref": model_ref,
        "status": status,
        "stop_reason": stop_reason,
        "baseline_sec": baseline_sec,
        "notes": notes,
        "steps": steps,
        "final": final_block,
        "scores_comparison": scores,
    }


def render_report_markdown(report: Mapping[str, Any]) -> str:
    final = report.get("final") or {}
    scores = report.get("scores_comparison") or {}
    lines: list[str] = []
    lines.append(f"# Compression Harness Report (`{report.get('report_id')}`)")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- Created: `{report.get('created_at')}`")
    lines.append(f"- Goal: `{report.get('goal_ref')}`")
    lines.append(f"- Model: `{report.get('model_ref')}`")
    lines.append(f"- Status: **{report.get('status')}**")
    lines.append(f"- Stop reason: `{report.get('stop_reason')}`")
    if report.get("baseline_sec") is not None:
        lines.append(f"- Baseline eval time: `{report.get('baseline_sec'):.2f}` s")
    if report.get("notes"):
        lines.append(f"- Notes: {report.get('notes')}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append(
        "| # | round | plugin | layers | recipe | bits | spars_d | compress_s | eval_s | total_s | "
        "size_bytes | score | drop | ok |"
    )
    lines.append(
        "|---|-------|--------|--------|--------|------|---------|------------|--------|---------|"
        "------------|-------|------|----|"
    )
    for s in report.get("steps") or []:
        dur = s.get("duration_sec") or {}
        lines.append(
            "| {idx} | {rnd} | `{plug}` | `{layers}` | `{recipe}` | {bits} | {sd} | {c} | {e} | {t} | "
            "{sz} | {score} | {drop} | {ok} |".format(
                idx=s.get("step_index"),
                rnd=s.get("round") if s.get("round") is not None else "-",
                plug=s.get("plugin") or "-",
                layers=s.get("layer_range") if s.get("layer_range") is not None else "-",
                recipe=s.get("recipe_id"),
                bits=s.get("weight_bits") if s.get("weight_bits") is not None else "-",
                sd=_fmt_num(s.get("sparsity_delta")) if s.get("sparsity_delta") is not None else "-",
                c=_fmt_num(dur.get("compress") if dur.get("compress") is not None else dur.get("apply")),
                e=_fmt_num(dur.get("eval")),
                t=_fmt_num(dur.get("total")),
                sz=_fmt_int(s.get("size_bytes")),
                score=_fmt_num(s.get("primary_score")),
                drop=_fmt_num(s.get("relative_drop")),
                ok=s.get("near_lossless_ok"),
            )
        )
    lines.append("")
    lines.append("## Final compression")
    lines.append("")
    lines.append(f"- Accepted run: `{final.get('accepted_run_id')}`")
    lines.append(f"- Dense weight bytes: `{_fmt_int(final.get('dense_size_bytes'))}`")
    lines.append(f"- Compressed bytes: `{_fmt_int(final.get('compressed_size_bytes'))}`")
    lines.append(f"- Compression ratio (dense/compressed): **{_fmt_num(final.get('compression_ratio'))}**")
    lines.append(f"- Size reduction: **{_fmt_num(final.get('size_reduction_pct'))}%**")
    lines.append(
        f"- Primary `{final.get('primary_metric')}`: "
        f"baseline `{_fmt_num(final.get('baseline_score'))}` → "
        f"final `{_fmt_num(final.get('final_score'))}` "
        f"(drop `{_fmt_num(final.get('relative_drop'))}`)"
    )
    lines.append("")
    lines.append("## Score multi-dimension comparison")
    lines.append("")
    lines.append(
        f"Primary task: `{scores.get('primary_task')}` "
        f"(mapped to dim `{scores.get('primary_dim')}`). "
        "Unset dims are `null` placeholders for future full-vector eval."
    )
    lines.append("")
    lines.append("| dim | dense | compressed | delta (comp - dense) |")
    lines.append("|-----|-------|------------|----------------------|")
    dense = scores.get("dense") or {}
    comp = scores.get("compressed") or {}
    deltas = scores.get("deltas") or {}
    for dim in SCORE_DIMS:
        lines.append(
            f"| {dim} | {_fmt_num(dense.get(dim))} | {_fmt_num(comp.get(dim))} | "
            f"{_fmt_num(deltas.get(dim))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_num(v: Any) -> str:
    if v is None:
        return "null"
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_int(v: Any) -> str:
    if v is None:
        return "null"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def write_report(
    report: Mapping[str, Any],
    out_dir: str | Path | None = None,
    *,
    copy_to_run_dir: bool = True,
) -> dict[str, str]:
    """Write report.json + report.md; optionally copy into accepted run dir."""
    base = Path(out_dir) if out_dir else (EXPERIMENTS_DIR / "reports")
    base.mkdir(parents=True, exist_ok=True)
    rid = str(report.get("report_id") or "report")
    json_path = base / f"{rid}.json"
    md_path = base / f"{rid}.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dict(report), f, indent=2, ensure_ascii=False)
        f.write("\n")
    md = render_report_markdown(report)
    md_path.write_text(md, encoding="utf-8")

    paths = {"json_path": str(json_path), "md_path": str(md_path)}
    if copy_to_run_dir:
        run_id = (report.get("final") or {}).get("accepted_run_id")
        if run_id:
            run_dir = EXPERIMENTS_DIR / str(run_id)
            if run_dir.is_dir():
                run_json = run_dir / "report.json"
                run_md = run_dir / "report.md"
                run_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
                run_md.write_text(md, encoding="utf-8")
                paths["run_json_path"] = str(run_json)
                paths["run_md_path"] = str(run_md)
    print(f"[OK] report written json={json_path} md={md_path}")
    return paths


def build_report_from_runs(
    run_ids: Sequence[str] | None = None,
    *,
    experiments_dir: Path | None = None,
    goal_doc: Mapping[str, Any] | None = None,
    model_ref: str | None = None,
) -> dict[str, Any]:
    """Assemble a report from existing artifact.json files (no re-eval)."""
    root = Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
    if run_ids:
        dirs = [root / rid for rid in run_ids]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_"))
    history: list[dict[str, Any]] = []
    for d in dirs:
        art_path = d / "artifact.json"
        if not art_path.exists():
            continue
        with art_path.open(encoding="utf-8") as f:
            history.append(json.load(f))
    if not history:
        raise FileNotFoundError(f"no run artifacts under {root}")

    model = model_ref or ""
    if not model:
        exec0 = (history[0].get("metrics") or {}).get("exec") or {}
        model = str(exec0.get("model_ref") or "")
    goal_ref = str(history[0].get("goal_ref") or "")
    status = "accepted" if any(a.get("status") == "accepted" for a in history) else "failed"
    return build_report(
        goal_doc,
        history,
        model_ref=model,
        stop_reason="from_artifacts",
        goal_ref=goal_ref,
        status=status,
        notes="Built from existing artifacts; timing may be null for historical runs.",
    )
