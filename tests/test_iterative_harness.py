"""Iterative compression framework tests (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

HARNESS_SRC = Path(__file__).resolve().parents[1] / "harness" / "src"
sys.path.insert(0, str(HARNESS_SRC))

from compression_harness.iterative_controller import IterativeController, run_iterate
from compression_harness.plugins.base import ModelState, validate_step_spec
from compression_harness.plugins.registry import list_plugins, register_builtin_plugins
from compression_harness.schema_util import load_yaml, validate_instance
from compression_harness.paths import RECIPES_DIR


def test_iter_goals_validate():
    for name in ("goal_lossless_iter.yaml", "goal_lossy5_iter.yaml"):
        doc = load_yaml(RECIPES_DIR / name)
        assert validate_instance(doc, "goal.schema.json") == []
    lossless = load_yaml(RECIPES_DIR / "goal_lossless_iter.yaml")
    lossy = load_yaml(RECIPES_DIR / "goal_lossy5_iter.yaml")
    assert lossless["goal"]["quality"]["mode"] == "lossless"
    assert lossless["goal"]["quality"]["max_relative_drop"] == 0.01
    assert lossy["goal"]["quality"]["mode"] == "lossy_bounded"
    assert lossy["goal"]["quality"]["max_relative_drop"] == 0.05


def test_validate_step_rejects_giant_prune():
    errs = validate_step_spec(
        {
            "step_id": "bad",
            "plugin": "prune_stub",
            "kind": "prune",
            "sparsity": 0.6,
        },
        max_sparsity_delta=0.05,
    )
    assert errs


def test_validate_step_rejects_wide_layer_window():
    errs = validate_step_spec(
        {
            "step_id": "bad",
            "plugin": "prune_stub",
            "kind": "prune",
            "layer_range": [0, 10],
            "sparsity_delta": 0.05,
        },
        max_layers_per_step=2,
    )
    assert any("max_layers_per_step" in e for e in errs)


def test_builtin_plugins_registered():
    register_builtin_plugins()
    names = {p["name"] for p in list_plugins()}
    assert "prune_stub" in names
    assert "quantize_stub" in names
    assert "evaluate_stub" in names
    assert "prune_wanda" in names
    assert "quantize_torchao_layers" in names
    assert "evaluate_real" in names


def test_model_state_snapshot_strips_runtime():
    state = ModelState(model_ref="m")
    state.meta["_model"] = object()
    state.meta["_tokenizer"] = object()
    state.meta["prune_ops"] = [{"layers.0.mlp.intermediate": [0, 1]}]
    snap = state.snapshot()
    assert "_model" not in snap["meta"]
    assert "_tokenizer" not in snap["meta"]
    assert snap["meta"]["prune_ops"]


def test_step_planner_real_plugin_names():
    from compression_harness.step_planner import StepPlanner

    goal = load_yaml(RECIPES_DIR / "goal_lossy5_iter.yaml")
    goal["goal"]["model"]["num_layers"] = 8
    planner = StepPlanner(goal, use_real=True)
    state = ModelState(model_ref="m")
    step = planner.next_step(state, 1)
    assert step is not None
    assert step["plugin"] == "prune_wanda"
    step2 = planner.next_step(state, 2)
    assert step2 is not None
    assert step2["plugin"] == "quantize_torchao_layers"


def test_layer_utils_and_validate_goals_max_rounds():
    from compression_harness.plugins.layer_utils import layer_indices, mlp_intermediate_names

    assert layer_indices([2, 3]) == [2, 3]
    assert mlp_intermediate_names([0, 1]) == [
        "layers.0.mlp.intermediate",
        "layers.1.mlp.intermediate",
    ]
    lossy = load_yaml(RECIPES_DIR / "goal_lossy5_iter.yaml")
    assert lossy["goal"]["search"]["max_rounds"] == 6


def test_iterate_accepts_then_reverts_on_threshold(tmp_path):
    """Force scores: first rounds OK, then exceed 5% -> revert stop."""
    goal = load_yaml(RECIPES_DIR / "goal_lossy5_iter.yaml")
    goal["goal"]["search"]["max_rounds"] = 8
    goal["goal"]["model"]["num_layers"] = 8

    def score_fn(state: ModelState) -> float:
        if not state.applied_steps:
            return 1.0
        if len(state.applied_steps) >= 3:
            return 0.90  # 10% drop > 5%
        return 0.98  # 2% drop OK

    ctrl = IterativeController(goal, experiments_root=tmp_path, score_fn=score_fn, dry_run=True)
    result = ctrl.run(goal_ref="goal_lossy5_iter.yaml")
    assert result["stop_reason"] == "threshold_exceeded_reverted"
    assert (tmp_path / result["iter_id"] / "state.json").exists()
    accepted = [h for h in result["history"] if h["status"] == "accepted"]
    rejected = [h for h in result["history"] if "rejected" in str(h["status"])]
    assert len(accepted) >= 1
    assert len(rejected) >= 1
    assert result["last_good"] is not None


def test_iterate_lossless_threshold_tighter(tmp_path):
    goal = load_yaml(RECIPES_DIR / "goal_lossless_iter.yaml")
    goal["goal"]["search"]["max_rounds"] = 4
    goal["goal"]["model"]["num_layers"] = 6

    def score_fn(state: ModelState) -> float:
        if not state.applied_steps:
            return 1.0
        return 0.985  # 1.5% > 1% lossless band

    ctrl = IterativeController(goal, experiments_root=tmp_path, score_fn=score_fn, dry_run=True)
    result = ctrl.run(goal_ref="goal_lossless_iter.yaml")
    assert result["max_relative_drop"] == 0.01
    assert result["stop_reason"] == "threshold_exceeded_reverted"


def test_run_iterate_cli_path(tmp_path):
    result = run_iterate(
        str(RECIPES_DIR / "goal_lossless_iter.yaml"),
        dry_run=True,
        experiments_root=tmp_path,
    )
    assert "iter_id" in result
    assert result.get("errors") is None
    assert (tmp_path / result["iter_id"] / "state.json").exists()
