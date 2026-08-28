"""Tests for capability config resolution (BBH gen_kwargs overrides)."""
from pathlib import Path

from src.evaluation.capability import resolve_capability_config
from src.experiments.stage_a_common import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_bbh_maxtok1024_calib_yaml_reasoning_gen_kwargs():
    cfg = load_yaml(ROOT / "configs/stage_a/bbh_maxtok1024_calib.yaml")
    resolved = resolve_capability_config(cfg)
    reasoning = resolved["dimensions"]["Reasoning"]
    gk = reasoning.get("gen_kwargs") or {}
    assert gk.get("max_gen_toks") == 1024
    assert gk.get("do_sample") is False
    assert reasoning["task"] == "bbh"
    assert resolved["scan_limits"]["Reasoning"] == 64
    assert resolved["only_dimensions"] == ("Reasoning",)
    assert resolved["batch_size"] == 2


def test_reasoning_yaml_gen_kwargs_merge():
    cfg = {
        "evaluation": {
            "capability": {
                "mode": "scan",
                "dimensions": {
                    "Reasoning": {
                        "gen_kwargs": {
                            "max_gen_toks": 512,
                            "do_sample": False,
                        },
                    },
                },
            },
        },
        "hardware": {"batch_size": 2},
    }
    resolved = resolve_capability_config(cfg)
    gk = resolved["dimensions"]["Reasoning"]["gen_kwargs"]
    assert gk["max_gen_toks"] == 512
    assert gk["do_sample"] is False


def test_default_reasoning_gen_kwargs_1024():
    resolved = resolve_capability_config({"hardware": {"batch_size": 4}})
    gk = resolved["dimensions"]["Reasoning"]["gen_kwargs"]
    assert gk["max_gen_toks"] == 1024
    assert gk["do_sample"] is False


def test_e0_e1_model_paths_isolated():
    e0 = load_yaml(ROOT / "configs/stage_a/e0_dense.yaml")
    e1 = load_yaml(ROOT / "configs/stage_a/e1_oneshot_curve.yaml")
    e2 = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    assert "Instruct" in str(e0["model"]["path"])
    assert "Instruct" not in str(e1["model"]["path"])
    assert "Instruct" not in str(e2["model"]["path"])
    r0 = resolve_capability_config(e0)
    r1 = resolve_capability_config(e1)
    assert r0["dimensions"]["Reasoning"]["gen_kwargs"]["max_gen_toks"] == 1024
    assert r1["dimensions"]["Reasoning"]["gen_kwargs"]["max_gen_toks"] == 1024
    assert r0["batch_size"] == 1
    assert r1["batch_size"] == 8
