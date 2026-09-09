"""Controller: auto near-lossless search loop."""

from __future__ import annotations

import time
from typing import Any

from compression_harness.diagnoser import Diagnoser
from compression_harness.evaluator import Evaluator
from compression_harness.executor import Executor
from compression_harness.memory import ExperimentMemory
from compression_harness.paths import DEFAULT_GOAL_PATH
from compression_harness.planner import Planner
from compression_harness.reporter import (
    build_report,
    measure_dense_weight_bytes,
    measure_dir_bytes,
    write_report,
)
from compression_harness.schema_util import load_yaml, validate_instance


def run_auto(
    goal_path: str | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run auto loop. dry_run=True keeps Phase A stub adapters/scores."""
    path = goal_path or str(DEFAULT_GOAL_PATH)
    goal_doc = load_yaml(path)
    errs = validate_instance(goal_doc, "goal.schema.json")
    if errs:
        return {"status": "failed", "errors": errs}

    goal = goal_doc.get("goal", goal_doc)
    model_ref = str(goal.get("model", {}).get("path", ""))
    max_trials = int(goal.get("search", {}).get("max_trials", 10))

    planner = Planner()
    executor = Executor()
    evaluator = Evaluator(use_real=not dry_run)
    diagnoser = Diagnoser()
    memory = ExperimentMemory()

    dense_size = None if dry_run else measure_dense_weight_bytes(model_ref)

    print(f"[INFO] harness auto dry_run={dry_run} model={model_ref}")
    t_base0 = time.perf_counter()
    baseline = evaluator.baseline(model_ref, goal_doc)
    baseline_sec = time.perf_counter() - t_base0

    tried: set[str] = set()
    history: list[dict[str, Any]] = []
    stop_reason: str | None = None
    accepted: dict[str, Any] | None = None

    for trial in range(1, max_trials + 1):
        recipe = planner.next_recipe(tried, last_ok=False)
        if recipe is None:
            stop_reason = "no_more_recipes"
            break

        rid = str(recipe.get("recipe_id"))
        tried.add(rid)
        run_id = f"run_{trial:03d}_{rid}"
        run_dir = memory.run_dir(run_id)
        compressed_dir = run_dir / "compressed"

        weight_bits = (recipe.get("layers") or {}).get("default", {}).get("weight_bits")

        t_step0 = time.perf_counter()
        t_c0 = time.perf_counter()
        exec_result = executor.run(
            model_ref,
            recipe,
            force_stub=dry_run,
            output_dir=None if dry_run else compressed_dir,
        )
        compress_sec = time.perf_counter() - t_c0
        # Drop heavy objects from persisted metrics.
        model_obj = exec_result.pop("model", None)
        tok_obj = exec_result.pop("tokenizer", None)

        t_e0 = time.perf_counter()
        eval_result = evaluator.evaluate(
            model_ref,
            recipe,
            goal_doc,
            baseline_score=baseline,
            model=model_obj,
            tokenizer=tok_obj,
            compressed_dir=None if dry_run else str(compressed_dir),
        )
        eval_sec = time.perf_counter() - t_e0
        step_sec = time.perf_counter() - t_step0

        # Free GPU memory before next trial if any.
        if model_obj is not None:
            del model_obj
        if tok_obj is not None:
            del tok_obj
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        compressed_bytes = None if dry_run else measure_dir_bytes(compressed_dir)
        size_ratio = None
        if dense_size and compressed_bytes:
            size_ratio = float(dense_size) / float(compressed_bytes)

        artifact: dict[str, Any] = {
            "run_id": run_id,
            "status": "accepted" if eval_result["near_lossless_ok"] else "rejected",
            "goal_ref": path,
            "recipe_ref": recipe.get("_recipe_path") or rid,
            "baseline_score": eval_result["baseline_score"],
            "score": eval_result["score"],
            "relative_drop": eval_result["relative_drop"],
            "near_lossless_ok": eval_result["near_lossless_ok"],
            "compressed_dir": None if dry_run else str(compressed_dir),
            "metrics": {
                "exec": exec_result,
                "eval": eval_result,
                "timing": {
                    "compress_sec": compress_sec,
                    "eval_sec": eval_sec,
                    "step_sec": step_sec,
                    "baseline_sec": baseline_sec,
                },
                "size": {
                    "dense_bytes": dense_size,
                    "compressed_bytes": compressed_bytes,
                    "compression_ratio": size_ratio,
                    "weight_bits": weight_bits,
                },
            },
            "notes": "dry_run_auto" if dry_run else "phase_b_auto",
        }

        if not eval_result["near_lossless_ok"]:
            diag = diagnoser.diagnose(artifact, goal_doc)
            artifact["hypothesis"] = diag["hypothesis"]
            artifact["next_recipe_suggestion"] = diag["next_recipe_suggestion"]
            artifact["status"] = "rejected"

        memory.write_artifact(run_id, artifact)
        history.append(artifact)

        stop_reason = planner.should_stop(
            bool(eval_result["near_lossless_ok"]), trial, max_trials
        )
        if stop_reason == "accepted_near_lossless":
            accepted = artifact
            artifact["status"] = "accepted"
            memory.write_artifact(run_id, artifact)
            break
        if stop_reason == "budget_exhausted":
            break

    status = "accepted" if accepted else "failed"
    report_paths: dict[str, str] | None = None
    try:
        report = build_report(
            goal_doc,
            history,
            model_ref=model_ref,
            stop_reason=stop_reason,
            goal_ref=path,
            dense_size=dense_size,
            baseline_sec=baseline_sec,
            status=status,
        )
        # Write under memory root / reports so tests using tmp_path stay isolated.
        reports_dir = memory.root / "reports"
        # Temporarily point EXPERIMENTS_DIR copies: write_report uses EXPERIMENTS_DIR
        # for run copies; pass accepted path explicitly via monkey-friendly helper.
        report_paths = _write_report_with_memory_root(report, memory.root, reports_dir)
    except Exception as exc:  # noqa: BLE001 — report must not kill auto result
        print(f"[WARNING] report generation failed: {exc}")

    return {
        "status": status,
        "stop_reason": stop_reason or "unknown",
        "accepted": accepted,
        "history": history,
        "model_ref": model_ref,
        "goal_path": path,
        "dry_run": dry_run,
        "baseline_sec": baseline_sec,
        "report_paths": report_paths,
    }


def _write_report_with_memory_root(
    report: dict[str, Any],
    experiments_root: Any,
    reports_dir: Any,
) -> dict[str, str]:
    from pathlib import Path

    from compression_harness import reporter as reporter_mod

    paths = write_report(report, reports_dir, copy_to_run_dir=False)
    run_id = (report.get("final") or {}).get("accepted_run_id")
    if run_id:
        run_dir = Path(experiments_root) / str(run_id)
        if run_dir.is_dir():
            run_json = run_dir / "report.json"
            run_md = run_dir / "report.md"
            run_json.write_text(Path(paths["json_path"]).read_text(encoding="utf-8"), encoding="utf-8")
            run_md.write_text(Path(paths["md_path"]).read_text(encoding="utf-8"), encoding="utf-8")
            paths["run_json_path"] = str(run_json)
            paths["run_md_path"] = str(run_md)
            print(f"[OK] report copied to run dir {run_dir}")
    _ = reporter_mod  # keep import for type checkers / future hooks
    return paths


def dry_run_auto(
    goal_path: str | None = None,
    *,
    force_stub: bool = True,
) -> dict[str, Any]:
    """Backward-compatible alias for Phase A tests."""
    return run_auto(goal_path, dry_run=bool(force_stub))
