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


def test_e2_cell_winner_distinguishes_oneshot_and_tie():
    # iterative wins 3 main dims -> cell iterative
    win_i = compare_iterative_vs_oneshot(
        {"PPL": 10.0, "Math": 0.1, "Knowledge": 0.1, "Reasoning": 0.1, "Instruction": 0.9, "Code": 0.9},
        {"PPL": 5.0, "Math": 0.2, "Knowledge": 0.2, "Reasoning": 0.05, "Instruction": 0.1, "Code": 0.1},
    )
    assert win_i["cell_winner"] == "iterative"
    assert win_i["cell_iterative_win"] is True

    # oneshot wins 3 main dims -> cell oneshot (not "oneshot/tie")
    win_o = compare_iterative_vs_oneshot(
        {"PPL": 5.0, "Math": 0.2, "Knowledge": 0.2, "Reasoning": 0.2, "Instruction": 0.0, "Code": 0.0},
        {"PPL": 10.0, "Math": 0.1, "Knowledge": 0.1, "Reasoning": 0.3, "Instruction": 0.9, "Code": 0.9},
    )
    assert win_o["cell_winner"] == "oneshot"
    assert win_o["main_dim_oneshot_wins"] == 3
    assert win_o["cell_iterative_win"] is False

    # 2-2 split -> cell tie; dim-level equal scores stay "tie"
    split = compare_iterative_vs_oneshot(
        {"PPL": 5.0, "Math": 0.2, "Knowledge": 0.1, "Reasoning": 0.1, "Instruction": 0.5, "Code": 0.5},
        {"PPL": 10.0, "Math": 0.1, "Knowledge": 0.2, "Reasoning": 0.2, "Instruction": 0.5, "Code": 0.5},
    )
    assert split["cell_winner"] == "tie"
    assert split["winner_per_dim"]["Instruction"] == "tie"
    assert split["cell_iterative_win"] is False
