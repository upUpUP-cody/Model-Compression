"""E1 Reasoning patch skip-if-1024 helpers."""
from __future__ import annotations

from experiments.stage_a.run_e1_reasoning_patch import (
    _dense_reasoning_is_1024,
    _row_reasoning_is_1024,
)


def test_row_reasoning_is_1024_top_level_details():
    row = {
        "sparsity": 0.1,
        "details": {
            "Reasoning": {
                "gen_kwargs": {"max_gen_toks": 1024, "do_sample": False},
            }
        },
    }
    assert _row_reasoning_is_1024(row) is True


def test_row_reasoning_is_1024_missing():
    row = {"sparsity": 0.3, "details": {"Reasoning": {"score": 0.1}}}
    assert _row_reasoning_is_1024(row) is False


def test_dense_reasoning_is_1024():
    dense = {
        "details": {
            "Reasoning": {
                "gen_kwargs": {"max_gen_toks": 1024, "do_sample": False},
            }
        }
    }
    assert _dense_reasoning_is_1024(dense) is True
