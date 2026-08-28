"""E2 iterative helpers (step math, winners)."""
from src.experiments.stage_a_common import (
    compare_iterative_vs_oneshot,
    incremental_step_targets,
    remaining_relative_prune_ratio,
)


def test_e2_step_counts_for_pdf_targets():
    assert len(incremental_step_targets(0.4, step=0.05)) == 8
    assert len(incremental_step_targets(0.5, step=0.05)) == 10
    assert len(incremental_step_targets(0.6, step=0.05)) == 12


def test_e2_remaining_relative_first_step_is_5pct():
    assert abs(remaining_relative_prune_ratio(0.0, 0.05) - 0.05) < 1e-12


def test_e2_winner_ppl_lower_is_better():
    cmp = compare_iterative_vs_oneshot(
        {"PPL": 100.0, "Math": 0.0, "Knowledge": 0.0, "Reasoning": 0.0, "Instruction": 0.0, "Code": 0.0},
        {"PPL": 50.0, "Math": 0.0, "Knowledge": 0.0, "Reasoning": 0.0, "Instruction": 0.0, "Code": 0.0},
    )
    assert cmp["winner_per_dim"]["PPL"] == "iterative"
