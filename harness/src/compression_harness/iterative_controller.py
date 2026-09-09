"""Persistent short-step iterative compression loop."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from compression_harness.paths import EXPERIMENTS_DIR
from compression_harness.plugins.base import ModelState, validate_step_spec
from compression_harness.plugins.evaluate import EvaluatePlugin
from compression_harness.plugins.registry import get_plugin, register_builtin_plugins
from compression_harness.reporter import build_report, write_report
from compression_harness.schema_util import load_yaml, validate_instance
from compression_harness.step_planner import StepPlanner


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_drop(goal: Mapping[str, Any]) -> float:
    g = goal.get("goal", goal)
    quality = g.get("quality") or {}
    mode = str(quality.get("mode") or "")
    if quality.get("max_relative_drop") is not None:
        return float(quality["max_relative_drop"])
    if mode == "lossless":
        return 0.01
    if mode == "lossy_bounded":
        return 0.05
    return float(quality.get("max_relative_drop") or 0.02)


def _search(goal: Mapping[str, Any]) -> dict[str, Any]:
    return dict((goal.get("goal", goal).get("search")) or {})


class IterativeController:
    def __init__(
        self,
        goal_doc: Mapping[str, Any],
        *,
        experiments_root: Path | None = None,
        score_fn: Callable[[ModelState], float] | None = None,
        dry_run: bool = True,
    ) -> None:
        self.goal_doc = goal_doc
        self.goal = goal_doc.get("goal", goal_doc)
        self.experiments_root = Path(experiments_root) if experiments_root else EXPERIMENTS_DIR
        self.dry_run = dry_run
        self.score_fn = score_fn
        self.max_drop = _max_drop(goal_doc)
        search = _search(goal_doc)
        self.max_rounds = int(search.get("max_rounds") or search.get("max_trials") or 20)
        self.stop_on_violation = str(search.get("stop_on_violation") or "revert_last_good")
        self.planner = StepPlanner(goal_doc, use_real=not dry_run)
        register_builtin_plugins()
        if dry_run:
            self.eval_plugin = EvaluatePlugin(score_fn=score_fn)
        else:
            self.eval_plugin = get_plugin("evaluate_real")

    def run(
        self,
        *,
        iter_id: str | None = None,
        resume: str | None = None,
        goal_ref: str = "",
    ) -> dict[str, Any]:
        if resume:
            iter_dir = self.experiments_root / resume
            state, history, meta = self._load_state(iter_dir)
            iter_id = resume
            if not self.dry_run:
                self._ensure_real_runtime(state, iter_dir, resume=True)
        else:
            iter_id = iter_id or f"iter_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            iter_dir = self.experiments_root / iter_id
            iter_dir.mkdir(parents=True, exist_ok=True)
            model_ref = str(self.goal.get("model", {}).get("path", ""))
            state = ModelState(model_ref=model_ref)
            state.meta["baseline_score"] = 1.0
            state.meta["max_relative_drop"] = self.max_drop
            state.meta["goal_doc"] = self.goal_doc
            if self.score_fn is not None:
                state.meta["baseline_score"] = float(self.score_fn(ModelState(model_ref=model_ref)))
            history = []
            meta = {
                "iter_id": iter_id,
                "goal_ref": goal_ref,
                "created_at": _utc_now(),
                "status": "running",
                "stop_reason": None,
                "last_good": None,
                "baseline_score": state.meta["baseline_score"],
                "max_relative_drop": self.max_drop,
                "quality_mode": (self.goal.get("quality") or {}).get("mode"),
                "dry_run": self.dry_run,
            }
            if not self.dry_run:
                self._ensure_real_runtime(state, iter_dir, resume=False)
                meta["baseline_score"] = state.meta["baseline_score"]
            self._save_state(iter_dir, state, history, meta)

        print(
            f"[INFO] iterate id={iter_id} mode={meta.get('quality_mode')} "
            f"max_drop={self.max_drop} dry_run={self.dry_run}"
        )

        start_round = int(state.round_index) + 1
        stop_reason = None
        last_good_state = ModelState.from_snapshot(state.snapshot())
        last_good_ckpt = None
        if meta.get("last_good"):
            try:
                last_good_state = ModelState.from_snapshot(meta["last_good"]["state"])
                last_good_ckpt = meta["last_good"].get("checkpoint_dir")
            except Exception:
                pass
        if not self.dry_run and state.checkpoint_dir and last_good_ckpt is None:
            last_good_ckpt = state.checkpoint_dir

        for round_index in range(start_round, self.max_rounds + 1):
            step = self.planner.next_step(state, round_index)
            if step is None:
                stop_reason = "search_space_exhausted"
                break

            search = _search(self.goal_doc)
            errs = validate_step_spec(
                step,
                max_layers_per_step=int(search.get("max_layers_per_step") or 2),
                max_sparsity_delta=float(search.get("max_sparsity_delta") or 0.05),
                min_weight_bits=int(search.get("min_weight_bits") or 8),
            )
            if errs:
                stop_reason = f"invalid_step:{errs}"
                break

            t0 = time.perf_counter()
            plugin = get_plugin(str(step["plugin"]))
            before_ckpt = state.checkpoint_dir
            state, apply_metrics = plugin.apply(state, step)
            apply_sec = time.perf_counter() - t0

            eval_step = {
                "step_id": f"r{round_index:03d}_eval",
                "plugin": self.eval_plugin.name,
                "kind": "evaluate",
                "baseline_score": state.meta.get("baseline_score", 1.0),
                "max_relative_drop": self.max_drop,
                "goal_doc": self.goal_doc,
            }
            t1 = time.perf_counter()
            state, eval_metrics = self.eval_plugin.apply(state, eval_step)
            eval_sec = time.perf_counter() - t1

            drop = float(eval_metrics["relative_drop"])
            within = bool(eval_metrics["within_threshold"])
            record = {
                "round": round_index,
                "run_id": f"{iter_id}_r{round_index:03d}",
                "status": "accepted" if within else "rejected",
                "step": step,
                "plugin": step.get("plugin"),
                "kind": step.get("kind"),
                "layer_range": step.get("layer_range"),
                "weight_bits": step.get("weight_bits"),
                "sparsity_delta": step.get("sparsity_delta"),
                "baseline_score": eval_metrics["baseline_score"],
                "score": eval_metrics["score"],
                "relative_drop": drop,
                "near_lossless_ok": within,
                "metrics": {
                    "apply": apply_metrics,
                    "eval": eval_metrics,
                    "timing": {
                        "apply_sec": apply_sec,
                        "eval_sec": eval_sec,
                        "step_sec": apply_sec + eval_sec,
                    },
                },
            }

            if within:
                state.round_index = round_index
                if not self.dry_run:
                    ckpt = self._accept_checkpoint(iter_dir, state, round_index)
                    last_good_ckpt = ckpt
                last_good_state = ModelState.from_snapshot(state.snapshot())
                meta["last_good"] = {
                    "round": round_index,
                    "state": last_good_state.snapshot(),
                    "score": eval_metrics["score"],
                    "relative_drop": drop,
                    "checkpoint_dir": last_good_ckpt or before_ckpt,
                }
                history.append(record)
                if drop > 0.5 * self.max_drop:
                    lr = step.get("layer_range") or []
                    if len(lr) == 2:
                        self.planner.mark_sensitive(state, list(range(int(lr[0]), int(lr[1]) + 1)))
                self._save_state(iter_dir, state, history, meta)
                self._write_round_artifact(iter_dir, record)
                print(
                    f"[OK] round={round_index} kind={step.get('kind')} "
                    f"layers={step.get('layer_range')} drop={drop:.4f} accepted"
                )
            else:
                print(
                    f"[WARNING] round={round_index} drop={drop:.4f} > max={self.max_drop}; "
                    f"revert_and_stop"
                )
                record["status"] = "rejected_reverted"
                history.append(record)
                if self.stop_on_violation == "revert_last_good":
                    if not self.dry_run:
                        state = self._revert_to_checkpoint(
                            state, last_good_state, last_good_ckpt, iter_dir
                        )
                    else:
                        state = ModelState.from_snapshot(last_good_state.snapshot())
                stop_reason = "threshold_exceeded_reverted"
                meta["status"] = "stopped"
                meta["stop_reason"] = stop_reason
                self._save_state(iter_dir, state, history, meta)
                self._write_round_artifact(iter_dir, record)
                break
        else:
            stop_reason = stop_reason or "max_rounds_reached"

        if meta.get("status") != "stopped":
            meta["status"] = "completed"
            meta["stop_reason"] = stop_reason or "completed"
            self._save_state(iter_dir, state, history, meta)

        report_paths = self._maybe_report(iter_dir, history, meta, goal_ref)
        return {
            "status": "accepted"
            if history and any(h.get("status") == "accepted" for h in history)
            else "failed",
            "iter_id": iter_id,
            "iter_dir": str(iter_dir),
            "stop_reason": meta.get("stop_reason"),
            "max_relative_drop": self.max_drop,
            "quality_mode": meta.get("quality_mode"),
            "history": history,
            "final_state": state.snapshot(),
            "last_good": meta.get("last_good"),
            "report_paths": report_paths,
            "dry_run": self.dry_run,
        }

    def _ensure_real_runtime(self, state: ModelState, iter_dir: Path, *, resume: bool) -> None:
        from compression_harness.model_io import (
            attach_runtime,
            load_checkpoint_into_state,
            load_hf_model,
            save_checkpoint,
        )
        from compression_harness.evaluator import Evaluator

        state.meta["max_relative_drop"] = self.max_drop
        state.meta["goal_doc"] = self.goal_doc

        if resume and state.checkpoint_dir and Path(state.checkpoint_dir).exists():
            load_checkpoint_into_state(state, state.checkpoint_dir)
        else:
            model, tokenizer = load_hf_model(state.model_ref)
            attach_runtime(state, model=model, tokenizer=tokenizer)
            ckpt0 = save_checkpoint(state, iter_dir / "ckpt" / "r000_init")
            state.checkpoint_dir = ckpt0

        evaluator = Evaluator(use_real=True)
        baseline = evaluator.baseline(state.model_ref, self.goal_doc)
        state.meta["baseline_score"] = float(baseline)
        print(f"[OK] iterate baseline hellaswag@64={baseline:.6f}")

    def _accept_checkpoint(self, iter_dir: Path, state: ModelState, round_index: int) -> str:
        from compression_harness.model_io import save_checkpoint

        path = save_checkpoint(state, iter_dir / "ckpt" / f"r{round_index:03d}")
        # Keep only init + latest accepted to limit disk (3B weights ~6GB each).
        ckpt_root = iter_dir / "ckpt"
        keep = {f"r{round_index:03d}", "r000_init"}
        if ckpt_root.is_dir():
            for child in ckpt_root.iterdir():
                if child.is_dir() and child.name not in keep:
                    import shutil

                    shutil.rmtree(child, ignore_errors=True)
        return path

    def _revert_to_checkpoint(
        self,
        state: ModelState,
        last_good_state: ModelState,
        last_good_ckpt: str | None,
        iter_dir: Path,
    ) -> ModelState:
        from compression_harness.model_io import clear_runtime, load_checkpoint_into_state

        clear_runtime(state)
        restored = ModelState.from_snapshot(last_good_state.snapshot())
        ckpt = last_good_ckpt or restored.checkpoint_dir
        if not ckpt:
            ckpt = str(iter_dir / "ckpt" / "r000_init")
        load_checkpoint_into_state(restored, ckpt)
        restored.meta["baseline_score"] = state.meta.get(
            "baseline_score", restored.meta.get("baseline_score")
        )
        restored.meta["max_relative_drop"] = self.max_drop
        restored.meta["goal_doc"] = self.goal_doc
        print(f"[OK] reverted to checkpoint {ckpt}")
        return restored

    def _maybe_report(
        self,
        iter_dir: Path,
        history: list[dict[str, Any]],
        meta: dict[str, Any],
        goal_ref: str,
    ) -> dict[str, str] | None:
        try:
            adapted = []
            for h in history:
                adapted.append(
                    {
                        "run_id": h.get("run_id"),
                        "status": "accepted" if h.get("status") == "accepted" else "rejected",
                        "goal_ref": goal_ref,
                        "recipe_ref": h.get("step", {}).get("step_id"),
                        "baseline_score": h.get("baseline_score"),
                        "score": h.get("score"),
                        "relative_drop": h.get("relative_drop"),
                        "near_lossless_ok": h.get("status") == "accepted",
                        "metrics": {
                            "exec": {
                                "backend": "plugin",
                                "method": h.get("kind"),
                                "recipe_id": h.get("step", {}).get("step_id"),
                                "plugin": h.get("plugin"),
                                "layer_range": h.get("layer_range"),
                                "weight_bits": h.get("weight_bits"),
                                "sparsity_delta": h.get("sparsity_delta"),
                                "round": h.get("round"),
                            },
                            "eval": {
                                "task": "hellaswag",
                                "score": h.get("score"),
                                "baseline_score": h.get("baseline_score"),
                            },
                            "timing": {
                                "compress_sec": ((h.get("metrics") or {}).get("timing") or {}).get(
                                    "apply_sec"
                                ),
                                "eval_sec": ((h.get("metrics") or {}).get("timing") or {}).get(
                                    "eval_sec"
                                ),
                                "step_sec": ((h.get("metrics") or {}).get("timing") or {}).get(
                                    "step_sec"
                                ),
                            },
                            "size": {
                                "weight_bits": h.get("weight_bits"),
                            },
                        },
                    }
                )
            model_ref = str(self.goal.get("model", {}).get("path", ""))
            report = build_report(
                self.goal_doc,
                adapted,
                model_ref=model_ref,
                stop_reason=meta.get("stop_reason"),
                goal_ref=goal_ref,
                status="accepted" if any(a["status"] == "accepted" for a in adapted) else "failed",
                notes="iterative short-step run (Phase II real plugins when --real)",
            )
            for i, h in enumerate(history):
                if i < len(report["steps"]):
                    report["steps"][i]["round"] = h.get("round")
                    report["steps"][i]["plugin"] = h.get("plugin")
                    report["steps"][i]["layer_range"] = h.get("layer_range")
                    report["steps"][i]["sparsity_delta"] = h.get("sparsity_delta")
            reports_dir = iter_dir / "reports"
            paths = write_report(report, reports_dir, copy_to_run_dir=False)
            (iter_dir / "report.json").write_text(
                Path(paths["json_path"]).read_text(encoding="utf-8"), encoding="utf-8"
            )
            (iter_dir / "report.md").write_text(
                Path(paths["md_path"]).read_text(encoding="utf-8"), encoding="utf-8"
            )
            paths["iter_report_json"] = str(iter_dir / "report.json")
            paths["iter_report_md"] = str(iter_dir / "report.md")
            return paths
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] iterate report failed: {exc}")
            return None

    def _write_round_artifact(self, iter_dir: Path, record: dict[str, Any]) -> None:
        rounds = iter_dir / "rounds"
        rounds.mkdir(parents=True, exist_ok=True)
        path = rounds / f"r{int(record['round']):03d}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def _save_state(
        self,
        iter_dir: Path,
        state: ModelState,
        history: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> None:
        payload = {
            "meta": meta,
            "state": state.snapshot(),
            "history": history,
            "updated_at": _utc_now(),
        }
        path = iter_dir / "state.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def _load_state(self, iter_dir: Path) -> tuple[ModelState, list[dict[str, Any]], dict[str, Any]]:
        path = iter_dir / "state.json"
        if not path.exists():
            raise FileNotFoundError(f"missing state.json in {iter_dir}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        state = ModelState.from_snapshot(data.get("state") or {})
        history = list(data.get("history") or [])
        meta = dict(data.get("meta") or {})
        return state, history, meta


def run_iterate(
    goal_path: str,
    *,
    resume: str | None = None,
    dry_run: bool = True,
    experiments_root: Path | None = None,
    score_fn: Callable[[ModelState], float] | None = None,
) -> dict[str, Any]:
    goal_doc = load_yaml(goal_path)
    errs = validate_instance(goal_doc, "goal.schema.json")
    if errs:
        return {"status": "failed", "errors": errs}
    ctrl = IterativeController(
        goal_doc,
        experiments_root=experiments_root,
        score_fn=score_fn,
        dry_run=dry_run,
    )
    return ctrl.run(resume=resume, goal_ref=goal_path)
