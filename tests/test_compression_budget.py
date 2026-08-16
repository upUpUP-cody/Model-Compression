"""Unit tests for dense-relative compression budget (no network)."""
from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.experiments.qwen_k5_comparison import (
    count_params,
    ratios_for_stage_target,
)
from src.pruning.pruning_backend import resolve_pruning_backend


def _tiny_qwen():
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_iterative_stages_do_not_compound_past_global_target():
    dense = _tiny_qwen()
    dense_count = count_params(dense)
    backend = resolve_pruning_backend(dense, "qwen")

    ratios_15 = ratios_for_stage_target(dense, 1.5, baseline_parameter_count=dense_count)
    model_15 = backend.create_pruned_model(ratios_15)
    comp_15 = dense_count / max(count_params(model_15), 1)
    assert 1.35 <= comp_15 <= 1.65, f"stage1 compression={comp_15}"

    ratios_20 = ratios_for_stage_target(
        model_15,
        2.0,
        baseline_parameter_count=dense_count,
    )
    model_20 = resolve_pruning_backend(model_15, "qwen").create_pruned_model(ratios_20)
    comp_20 = dense_count / max(count_params(model_20), 1)
    # Must not overshoot like KG.5 (~2.8x). Allow modest binary-search slack.
    assert 1.7 <= comp_20 <= 2.25, f"stage2 compression_vs_dense={comp_20}"
    assert comp_20 < 2.5, "compounding regression: compression too high"


def test_already_at_target_returns_zero_prune():
    dense = _tiny_qwen()
    dense_count = count_params(dense)
    backend = resolve_pruning_backend(dense, "qwen")
    ratios = ratios_for_stage_target(dense, 1.5, baseline_parameter_count=dense_count)
    pruned = backend.create_pruned_model(ratios)
    # 1.2x is already exceeded by the 1.5x model → no further prune.
    again = ratios_for_stage_target(pruned, 1.2, baseline_parameter_count=dense_count)
    assert all(abs(float(v)) < 1e-8 for v in again.values())
