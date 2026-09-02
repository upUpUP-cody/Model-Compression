"""E2 evaluation.capability must match E1 (formal protocol)."""
from pathlib import Path

from src.evaluation.capability import resolve_capability_config
from src.experiments.stage_a_common import load_yaml, pruning_eval_protocol_digest

ROOT = Path(__file__).resolve().parents[1]


def test_e1_e2_capability_config_aligned():
    e1 = load_yaml(ROOT / "configs/stage_a/e1_oneshot_curve.yaml")
    e2 = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    r1 = resolve_capability_config(e1)
    r2 = resolve_capability_config(e2)
    assert r1["mode"] == r2["mode"] == "scan"
    assert r1["seed"] == r2["seed"] == 42
    assert r1["batch_size"] == r2["batch_size"] == 16
    assert r1["bootstrap_iters"] == r2["bootstrap_iters"] == 0
    assert r1["scan_limits"] == r2["scan_limits"]
    assert r1["scan_limits"] == {
        "PPL": 4,
        "Math": 64,
        "Knowledge": 128,
        "Reasoning": 64,
        "Instruction": 64,
        "Code": 32,
    }
    # Code HumanEval completion protocol from defaults
    code = r2["dimensions"]["Code"]
    assert code["apply_chat_template"] is False
    assert code["gen_kwargs"]["do_sample"] is False
    assert code["gen_kwargs"]["max_gen_toks"] == 512
    assert "\ndef" in code["gen_kwargs"]["until"]
    # Reasoning BBH 1024 protocol
    reasoning = r2["dimensions"]["Reasoning"]
    assert reasoning["gen_kwargs"]["max_gen_toks"] == 1024
    assert reasoning["gen_kwargs"]["do_sample"] is False
    assert r1["dimensions"]["Reasoning"]["gen_kwargs"]["max_gen_toks"] == 1024


def test_e2_yaml_has_no_sst2_proxy_eval_fields():
    e2 = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    evaluation = e2.get("evaluation") or {}
    assert "max_seq_len" not in evaluation
    assert "validation_max_samples" not in evaluation
    assert "train_max_samples" not in evaluation
    assert "capability" in evaluation
    assert e2.get("incremental_step_sparsity") == 0.05
    assert "iterative_steps" not in e2
    assert e2.get("hardware", {}).get("batch_size") == 16


def test_pruning_eval_protocol_digest_stable():
    e1 = load_yaml(ROOT / "configs/stage_a/e1_oneshot_curve.yaml")
    e2 = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    # Same model+pruning+capability → same protocol digest
    assert pruning_eval_protocol_digest(e1) == pruning_eval_protocol_digest(e2)
