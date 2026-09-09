"""Harness Phase B / report unit tests (mocked; no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HARNESS_SRC = Path(__file__).resolve().parents[1] / "harness" / "src"
sys.path.insert(0, str(HARNESS_SRC))

from compression_harness.controller import dry_run_auto, run_auto
from compression_harness.evaluator import Evaluator, hellaswag_capability_config
from compression_harness.planner import Planner
from compression_harness.schema_util import load_yaml, validate_instance
from compression_harness.paths import DEFAULT_GOAL_PATH, DEFAULT_RECIPE_ORDER, RECIPES_DIR
from compression_harness.reporter import SCORE_DIMS, build_report


def test_goal_and_recipes_validate():
    goal = load_yaml(DEFAULT_GOAL_PATH)
    assert validate_instance(goal, "goal.schema.json") == []
    assert int(goal["goal"]["evaluation"]["limit"]) == 64
    for name in DEFAULT_RECIPE_ORDER:
        recipe = load_yaml(RECIPES_DIR / name)
        assert validate_instance(recipe, "recipe.schema.json") == []


def test_recipe_order_phase_b_only_int8():
    assert DEFAULT_RECIPE_ORDER == ("global_int8.yaml",)


def test_hellaswag_capability_config():
    goal = load_yaml(DEFAULT_GOAL_PATH)
    cfg = hellaswag_capability_config(goal)
    cap = cfg["evaluation"]["capability"]
    assert cap["only_dimensions"] == ["Reasoning"]
    assert cap["scan_limits"]["Reasoning"] == 64
    assert cap["dimensions"]["Reasoning"]["task"] == "hellaswag"


def test_near_lossless_gate():
    ev = Evaluator(use_real=False)
    goal = load_yaml(DEFAULT_GOAL_PATH)
    assert ev.near_lossless_ok(1.0, 0.99, goal) is True
    assert ev.near_lossless_ok(1.0, 0.90, goal) is False


def test_planner_stops_when_last_ok():
    assert Planner().next_recipe(set(), last_ok=True) is None
    r = Planner().next_recipe(set(), last_ok=False)
    assert r is not None
    assert r["recipe_id"] == "global_int8_torchao"


def test_dry_run_auto_accepts_mild_int8(tmp_path, monkeypatch):
    from compression_harness.memory import ExperimentMemory

    real_init = ExperimentMemory.__init__

    def _init(self, root=None):
        real_init(self, root=tmp_path)

    monkeypatch.setattr(ExperimentMemory, "__init__", _init)
    result = dry_run_auto(force_stub=True)
    assert result["status"] == "accepted"
    assert result["stop_reason"] == "accepted_near_lossless"
    assert result["accepted"]["near_lossless_ok"] is True
    assert result["accepted"]["relative_drop"] <= 0.02
    assert result.get("report_paths")
    assert Path(result["report_paths"]["json_path"]).exists()
    assert Path(result["report_paths"]["md_path"]).exists()
    # timing present on new autos
    timing = result["accepted"]["metrics"]["timing"]
    assert "compress_sec" in timing and "eval_sec" in timing and "step_sec" in timing


def test_build_report_framework():
    history = [
        {
            "run_id": "run_001_demo",
            "status": "accepted",
            "goal_ref": "goal.yaml",
            "recipe_ref": "global_int8.yaml",
            "baseline_score": 0.8,
            "score": 0.79,
            "relative_drop": 0.0125,
            "near_lossless_ok": True,
            "compressed_dir": None,
            "metrics": {
                "exec": {
                    "backend": "torchao",
                    "method": "int8_weight_only",
                    "recipe_id": "global_int8_torchao",
                },
                "eval": {"task": "hellaswag", "primary_metric": "acc_norm"},
                "timing": {"compress_sec": 1.5, "eval_sec": 2.0, "step_sec": 3.5},
                "size": {
                    "dense_bytes": 1000,
                    "compressed_bytes": 550,
                    "compression_ratio": 1000 / 550,
                    "weight_bits": 8,
                },
            },
        }
    ]
    report = build_report(
        {"goal": {"evaluation": {"primary_metric": "acc_norm"}}},
        history,
        model_ref="/fake/model",
        stop_reason="accepted_near_lossless",
        goal_ref="goal.yaml",
        dense_size=1000,
        baseline_sec=4.0,
        status="accepted",
    )
    assert report["steps"][0]["weight_bits"] == 8
    assert report["steps"][0]["duration_sec"]["total"] == 3.5
    assert report["final"]["compression_ratio"] == pytest.approx(1000 / 550)
    assert set(report["scores_comparison"]["dense"].keys()) >= set(SCORE_DIMS)
    assert report["scores_comparison"]["dense"]["Reasoning"] == 0.8
    assert report["scores_comparison"]["compressed"]["Math"] is None
    assert validate_instance(report, "report.schema.json") == []


def test_run_auto_real_wiring_mocked(tmp_path, monkeypatch):
    """Real path wiring with mocked quantize + eval (no GPU)."""
    from compression_harness.memory import ExperimentMemory
    from compression_harness import controller as controller_mod
    from compression_harness import evaluator as evaluator_mod

    real_init = ExperimentMemory.__init__

    def _init(self, root=None):
        real_init(self, root=tmp_path)

    monkeypatch.setattr(ExperimentMemory, "__init__", _init)

    def fake_baseline(self, model_ref, goal, force_refresh=False):
        return 0.80

    def fake_score_run(self, model_ref, recipe, goal, **kwargs):
        return 0.79  # 1.25% relative drop -> OK

    def fake_exec_run(self, model_ref, recipe, force_stub=False, output_dir=None):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "quantized_state_dict.pt").write_bytes(b"stub")
        return {
            "status": "ok",
            "backend": "torchao",
            "method": "int8_weight_only",
            "output_dir": str(output_dir) if output_dir else None,
            "message": "[OK] mocked",
        }

    monkeypatch.setattr(evaluator_mod.Evaluator, "baseline", fake_baseline)
    monkeypatch.setattr(evaluator_mod.Evaluator, "score_run", fake_score_run)
    monkeypatch.setattr(controller_mod.Executor, "run", fake_exec_run)
    monkeypatch.setattr(controller_mod, "measure_dense_weight_bytes", lambda p: 1000)

    result = run_auto(dry_run=False)
    assert result["status"] == "accepted"
    assert result["dry_run"] is False
    assert result["accepted"]["near_lossless_ok"] is True
    assert Path(result["accepted"]["compressed_dir"]).exists()
    assert result.get("report_paths")
    assert "size" in result["accepted"]["metrics"]
