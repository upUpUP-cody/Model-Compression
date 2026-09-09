"""Soft-eval config alignment + checkpoint resume."""
from __future__ import annotations

import json
from pathlib import Path

from src.experiments.soft_eval_common import (
    SOFT_DIM_ORDER,
    SOFT_DIM_SPECS,
    SOFT_LITE_DIM_ORDER,
    apply_easy_capability_overrides,
    build_soft_checkpoint_payload,
    cell_is_completed,
    empty_scores_table,
    load_soft_checkpoint,
    resolve_soft_dim_specs,
    save_soft_checkpoint,
    soft_cell_key,
    soft_protocol_digest,
    sparsity_label,
)
from src.experiments.stage_a_common import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_soft_eval_yaml_model_is_base_not_instruct():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    path = str(cfg["model"]["path"])
    assert "Qwen2.5-3B" in path
    assert "Instruct" not in path
    assert cfg["model"]["pdf_model"] == "Qwen2.5-3B"


def test_soft_eval_grid_matches_e1_plus_dense():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    grid = [float(x) for x in cfg["sparsity_grid"]]
    assert grid[0] == 0.0
    assert grid[-1] == 0.70
    assert len(grid) == 8


def test_soft_dims_include_sentiment_and_easys():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    specs = resolve_soft_dim_specs(cfg, smoke=False)
    assert "Sentiment" in specs
    assert specs["Sentiment"]["kind"] == "glue_sst2"
    assert specs["Sentiment"]["metric"] == "classification_acc"
    for name in SOFT_DIM_ORDER:
        assert name in specs
    assert specs["Math_easy"]["base_dim"] == "Math"
    assert specs["Math_easy"]["kind"] == "capability"


def test_sentiment_is_not_calibration_lm_loader_config():
    """Sentiment must be classification; dataset.task sst2 is OK for glue splits,
    but soft dim must not be kind=lm_nll."""
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    sent = cfg["soft_dimensions"]["Sentiment"]
    assert sent["kind"] == "glue_sst2"
    assert "nll" not in str(sent).lower()


def test_easy_overrides_only_touch_base_dim():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    from src.experiments.soft_eval_common import resolve_soft_dim_specs

    specs = resolve_soft_dim_specs(cfg, smoke=False)
    spec = specs["Reasoning_easy"]
    out = apply_easy_capability_overrides(cfg, "Reasoning_easy", spec, smoke=False)
    cap = out["evaluation"]["capability"]
    assert cap["only_dimensions"] == ["Reasoning"]
    assert cap["scan_limits"]["Reasoning"] == 32
    assert cap["dimensions"]["Reasoning"]["task"] == "bbh_fewshot_boolean_expressions_easy"
    assert cap["dimensions"]["Reasoning"]["apply_chat_template"] is False
    gk = cap["dimensions"]["Reasoning"]["gen_kwargs"]
    assert gk["max_gen_toks"] == 32
    assert gk["do_sample"] is False
    assert "\n" in gk["until"]
    assert "<|im_end|>" in gk["until"]
    assert "exact_match,flexible-extract" in cap["dimensions"]["Reasoning"]["metric_candidates"]
    assert "lm_eval_tasks/easy" in str(cap.get("task_include_path") or "")


def test_reasoning_easy_soft_dim_spec_defaults():
    spec = SOFT_DIM_SPECS["Reasoning_easy"]
    assert spec["apply_chat_template"] is False
    assert spec["gen_kwargs"]["max_gen_toks"] == 32
    assert "<|im_end|>" in spec["gen_kwargs"]["until"]
    assert spec["task"] == "bbh_fewshot_boolean_expressions_easy"
def test_soft_checkpoint_resume_skips_completed(tmp_path: Path):
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    digest = soft_protocol_digest(cfg, smoke=False)
    grid = [0.0, 0.1]
    dims = ["Sentiment", "Math_easy"]
    scores = empty_scores_table(grid, dims)
    scores[sparsity_label(0.0)]["Sentiment"] = 0.9
    completed = [soft_cell_key(0.0, "Sentiment")]
    path = tmp_path / "soft_eval_checkpoint.json"
    payload = build_soft_checkpoint_payload(
        protocol_digest=digest,
        smoke=False,
        sparsity_grid=grid,
        soft_dims=dims,
        completed_cells=completed,
        scores=scores,
        started_at="t0",
        elapsed_sec=1.0,
        status="in_progress",
    )
    save_soft_checkpoint(path, payload)
    loaded = load_soft_checkpoint(path, digest)
    assert loaded is not None
    assert cell_is_completed(loaded["completed_cells"], 0.0, "Sentiment")
    assert not cell_is_completed(loaded["completed_cells"], 0.0, "Math_easy")
    assert loaded["scores"]["dense"]["Sentiment"] == 0.9


def test_soft_checkpoint_digest_mismatch_raises(tmp_path: Path):
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    digest = soft_protocol_digest(cfg, smoke=False)
    path = tmp_path / "soft_eval_checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "protocol_digest": "deadbeef",
                "completed_cells": [],
                "scores": {},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_soft_checkpoint(path, digest)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "digest mismatch" in str(exc)


def test_smoke_shrinks_limits():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    specs = resolve_soft_dim_specs(cfg, smoke=True)
    assert specs["Sentiment"]["max_samples"] <= 8
    assert specs["Math_easy"]["scan_limit"] <= 2


def test_lite_yaml_calib_stays_sst2_not_rte():
    """Wanda LM loaders need sst2; Instruction_lite forces rte only at eval."""
    cfg = load_yaml(ROOT / "configs/stage_a/e1_dim_lite_alts.yaml")
    assert cfg["dataset"]["task"] == "sst2"
    assert cfg["soft_dimensions"]["Instruction_lite"]["glue_task"] == "rte"
    assert cfg["soft_dimensions"]["Instruction_lite"]["kind"] == "glue_rte"


def test_lite_dims_local_tasks_and_include_path():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_dim_lite_alts.yaml")
    specs = resolve_soft_dim_specs(cfg, smoke=False)
    for name in SOFT_LITE_DIM_ORDER:
        assert name in specs
    assert specs["Math_lite"]["task"] == "arithmetic_2da_local"
    assert specs["Math_lite"]["task_include_path"] == "configs/lm_eval_tasks/lite"
    assert specs["Code_lite"]["task"] == "humaneval_single_line_infilling_local"
    assert specs["Code_lite"]["confirm_run_unsafe_code"] is True
    assert cfg["evaluation"]["capability"]["task_include_path"] == "configs/lm_eval_tasks/lite"


def test_lite_overrides_only_touch_math_base():
    cfg = load_yaml(ROOT / "configs/stage_a/e1_dim_lite_alts.yaml")
    spec = SOFT_DIM_SPECS["Math_lite"]
    out = apply_easy_capability_overrides(cfg, "Math_lite", spec, smoke=False)
    cap = out["evaluation"]["capability"]
    assert cap["only_dimensions"] == ["Math"]
    assert cap["dimensions"]["Math"]["task"] == "arithmetic_2da_local"
    assert cap["task_include_path"].endswith("configs/lm_eval_tasks/lite")
    assert Path(cap["task_include_path"]).is_absolute()
    assert cap["scan_limits"]["Math"] == 64


def test_lite_digest_differs_from_easy_and_soft():
    easy = load_yaml(ROOT / "configs/stage_a/e1_soft_eval_side.yaml")
    soft = load_yaml(ROOT / "configs/stage_a/e1_dim_soft_alts.yaml")
    lite = load_yaml(ROOT / "configs/stage_a/e1_dim_lite_alts.yaml")
    d_easy = soft_protocol_digest(easy, smoke=False)
    d_soft = soft_protocol_digest(soft, smoke=False)
    d_lite = soft_protocol_digest(lite, smoke=False)
    assert len({d_easy, d_soft, d_lite}) == 3
