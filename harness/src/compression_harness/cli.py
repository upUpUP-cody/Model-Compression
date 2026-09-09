"""CLI entrypoints for harness skills."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from compression_harness import actions
from compression_harness.controller import dry_run_auto, run_auto
from compression_harness.diagnoser import Diagnoser
from compression_harness.evaluator import Evaluator
from compression_harness.executor import Executor
from compression_harness.iterative_controller import run_iterate
from compression_harness.memory import ExperimentMemory
from compression_harness.paths import DEFAULT_GOAL_PATH, EXPERIMENTS_DIR, HARNESS_ROOT, RECIPES_DIR
from compression_harness.planner import Planner
from compression_harness.reporter import build_report_from_runs, write_report
from compression_harness.schema_util import load_yaml, validate_instance


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_validate(_: argparse.Namespace) -> int:
    script = HARNESS_ROOT / "scripts" / "validate_schemas.py"
    return subprocess.call([sys.executable, str(script)])


def cmd_compress(ns: argparse.Namespace) -> int:
    recipe = load_yaml(ns.recipe)
    errs = validate_instance(recipe, "recipe.schema.json")
    if errs:
        _print({"status": "failed", "errors": errs})
        return 1
    model = ns.model or load_yaml(ns.goal or DEFAULT_GOAL_PATH)["goal"]["model"]["path"]
    output_dir = ns.output_dir
    if not ns.stub and not output_dir:
        rid = recipe.get("recipe_id", "manual")
        output_dir = str(EXPERIMENTS_DIR / f"manual_{rid}" / "compressed")
    result = Executor().run(
        model,
        recipe,
        force_stub=bool(ns.stub),
        output_dir=None if ns.stub else output_dir,
    )
    result.pop("model", None)
    result.pop("tokenizer", None)
    _print(result)
    return 0 if result.get("status") in ("ok", "dry_run") else 1


def cmd_evaluate(ns: argparse.Namespace) -> int:
    goal_doc = load_yaml(ns.goal or DEFAULT_GOAL_PATH)
    recipe = load_yaml(ns.recipe) if ns.recipe else {"layers": {"default": {"weight_bits": 8}}}
    model = ns.model or goal_doc["goal"]["model"]["path"]
    evaluator = Evaluator(use_real=not ns.dry_run)
    result = evaluator.evaluate(
        model,
        recipe,
        goal_doc,
        compressed_dir=ns.compressed_dir,
    )
    _print(result)
    return 0


def cmd_diagnose(ns: argparse.Namespace) -> int:
    mem = ExperimentMemory()
    art = mem.read_artifact(ns.run_id)
    if art is None:
        _print({"status": "failed", "error": f"no artifact for {ns.run_id}"})
        return 1
    _print(Diagnoser().diagnose(art))
    return 0


def cmd_profile(ns: argparse.Namespace) -> int:
    _print(actions.profile_model(ns.model))
    return 0


def cmd_plan_next(ns: argparse.Namespace) -> int:
    tried = set(ns.tried or [])
    recipe = Planner().next_recipe(tried, last_ok=ns.last_ok)
    if recipe is None:
        _print({"status": "stop", "reason": "accepted_or_exhausted"})
        return 0
    _print({"status": "next", "recipe": recipe})
    return 0


def cmd_auto(ns: argparse.Namespace) -> int:
    if ns.dry_run:
        result = dry_run_auto(ns.goal, force_stub=True)
    else:
        result = run_auto(ns.goal, dry_run=False)
    _print(result)
    return 0 if result.get("status") == "accepted" else 1


def cmd_recover(_: argparse.Namespace) -> int:
    _print(
        {
            "status": "not_implemented",
            "message": (
                "[WARNING] recover skill is contract-only until Phase D "
                "(LoRA / recalibration / distill)."
            ),
        }
    )
    return 2


def cmd_report(ns: argparse.Namespace) -> int:
    """Build report from existing run artifacts (no re-eval)."""
    run_ids = [ns.run_id] if ns.run_id else None
    goal_doc = None
    if ns.goal:
        goal_doc = load_yaml(ns.goal)
    try:
        report = build_report_from_runs(
            run_ids,
            experiments_dir=EXPERIMENTS_DIR,
            goal_doc=goal_doc,
            model_ref=ns.model,
        )
    except FileNotFoundError as exc:
        _print({"status": "failed", "error": str(exc)})
        return 1
    out_dir = ns.out_dir or str(EXPERIMENTS_DIR / "reports")
    paths = write_report(report, out_dir, copy_to_run_dir=True)
    _print({"status": "ok", "report_id": report.get("report_id"), "paths": paths, "final": report.get("final")})
    return 0


def cmd_iterate(ns: argparse.Namespace) -> int:
    """Persistent short-step iterative compress (prune/quantize plugins)."""
    goal = ns.goal or str(RECIPES_DIR / "goal_lossless_iter.yaml")
    result = run_iterate(
        goal,
        resume=ns.resume,
        dry_run=not ns.real,
    )
    # Drop bulky history details in stdout if huge — keep summary fields
    summary = {
        "status": result.get("status"),
        "iter_id": result.get("iter_id"),
        "iter_dir": result.get("iter_dir"),
        "stop_reason": result.get("stop_reason"),
        "quality_mode": result.get("quality_mode"),
        "max_relative_drop": result.get("max_relative_drop"),
        "rounds_accepted": sum(1 for h in (result.get("history") or []) if h.get("status") == "accepted"),
        "rounds_total": len(result.get("history") or []),
        "last_good": result.get("last_good"),
        "report_paths": result.get("report_paths"),
        "errors": result.get("errors"),
    }
    _print(summary)
    if result.get("errors"):
        return 1
    return 0 if result.get("status") == "accepted" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compression_harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate schemas and example recipes")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("compress", help="Run compress (torchao INT8 by default)")
    c.add_argument("--recipe", required=True)
    c.add_argument("--goal", default=None)
    c.add_argument("--model", default=None)
    c.add_argument("--output-dir", default=None)
    c.add_argument(
        "--stub",
        action="store_true",
        help="Force stub adapter (no weight changes)",
    )
    c.set_defaults(func=cmd_compress)

    e = sub.add_parser("evaluate", help="Evaluate vs goal near-lossless band")
    e.add_argument("--recipe", default=None)
    e.add_argument("--goal", default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--compressed-dir", default=None)
    e.add_argument("--dry-run", action="store_true", help="Use stub scores")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("diagnose", help="Diagnose a run artifact")
    d.add_argument("--run-id", required=True)
    d.set_defaults(func=cmd_diagnose)

    pr = sub.add_parser("profile", help="Stub compressibility profile")
    pr.add_argument("--model", required=True)
    pr.set_defaults(func=cmd_profile)

    pn = sub.add_parser("plan-next", help="Suggest next milder_first recipe")
    pn.add_argument("--tried", nargs="*", default=[])
    pn.add_argument("--last-ok", action="store_true")
    pn.set_defaults(func=cmd_plan_next)

    a = sub.add_parser("auto", help="Auto near-lossless loop (real by default)")
    a.add_argument("--goal", default=str(DEFAULT_GOAL_PATH))
    a.add_argument(
        "--dry-run",
        action="store_true",
        help="Phase A stub adapters and synthetic scores",
    )
    a.set_defaults(func=cmd_auto)

    r = sub.add_parser("recover", help="Recover contract (Phase D)")
    r.set_defaults(func=cmd_recover)

    rp = sub.add_parser("report", help="Build post-run report from artifacts")
    rp.add_argument("--run-id", default=None, help="Single run_id; default=all run_*")
    rp.add_argument("--goal", default=None)
    rp.add_argument("--model", default=None)
    rp.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to EXPERIMENTS_DIR/reports (/mnt/data2/results/harness_experiments/reports)",
    )
    rp.set_defaults(func=cmd_report)

    it = sub.add_parser("iterate", help="Short-step iterative prune/quantize loop")
    it.add_argument(
        "--goal",
        default=str(RECIPES_DIR / "goal_lossless_iter.yaml"),
        help="lossless or lossy5 iterative goal YAML",
    )
    it.add_argument("--resume", default=None, help="Resume iter_* directory name")
    it.add_argument(
        "--real",
        action="store_true",
        help="Use Wanda prune + torchao layer INT8 + hellaswag@64 evaluate",
    )
    it.set_defaults(func=cmd_iterate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
