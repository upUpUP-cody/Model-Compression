"""E2 checkpoint resume and Gate A helpers."""
from pathlib import Path

import pytest

from src.experiments.stage_a_common import (
    build_e2_checkpoint_payload,
    cell_is_completed,
    compare_iterative_vs_oneshot,
    e2_cell_key,
    e2_config_digest,
    e2_partial_dims_complete,
    gate_a_iterative_advantage,
    incremental_step_targets,
    load_e2_checkpoint,
    load_yaml,
    merge_e2_capability_dim,
    merge_e2_records,
    new_e2_partial,
    remaining_relative_prune_ratio,
    save_e2_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_incremental_step_counts():
    assert incremental_step_targets(0.40) == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    assert len(incremental_step_targets(0.50)) == 10
    assert len(incremental_step_targets(0.60)) == 12
    with pytest.raises(ValueError):
        incremental_step_targets(0.42)


def test_remaining_relative_reaches_target_math():
    # Compound remaining-relative ratios should yield exact absolute sparsity product.
    prev = 0.0
    for cum in incremental_step_targets(0.40):
        r = remaining_relative_prune_ratio(prev, cum)
        # keep fraction of original = (1-prev)*(1-r) = 1-cum
        keep = (1.0 - prev) * (1.0 - r)
        assert abs(keep - (1.0 - cum)) < 1e-9
        prev = cum
    # Fixed 0.05 × 8 would undershoot — document contrast
    fixed_keep = (0.95) ** 8
    assert fixed_keep > 0.60  # ~0.66 remain → only ~34% sparse


def test_compare_and_gate():
    oneshot = {"PPL": 50.0, "Math": 0.1, "Knowledge": 0.2, "Reasoning": 0.1, "Instruction": 0.1, "Code": 0.1}
    iterative = {"PPL": 40.0, "Math": 0.2, "Knowledge": 0.25, "Reasoning": 0.05, "Instruction": 0.1, "Code": 0.0}
    cmp = compare_iterative_vs_oneshot(oneshot, iterative)
    # PPL, Math, Knowledge better; Reasoning worse → 3/4 → win
    assert cmp["main_dim_iterative_wins"] == 3
    assert cmp["cell_iterative_win"] is True
    assert cmp["winner_per_dim"]["PPL"] == "iterative"
    assert cmp["winner_per_dim"]["Reasoning"] == "oneshot"

    cells = [{"cell_iterative_win": True}] * 7 + [{"cell_iterative_win": False}] * 2
    gate = gate_a_iterative_advantage(cells, min_wins=7)
    assert gate["passed"] is True
    gate2 = gate_a_iterative_advantage(cells, min_wins=8)
    assert gate2["passed"] is False


def test_checkpoint_partial_resume(tmp_path: Path):
    cfg = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    digest = e2_config_digest(cfg, smoke=False, seeds=[42, 43])
    partial = new_e2_partial(43, 0.5, "oneshot")
    # Fake PPL slice
    cap_slice = {
        "vector": {"PPL": 12.0},
        "details": {"PPL": {"score": 12.0, "task": "wikitext"}},
        "raw": {},
    }
    partial = merge_e2_capability_dim(partial, "PPL", cap_slice)
    assert partial["completed_dimensions"] == ["PPL"]
    assert not e2_partial_dims_complete(partial)

    path = tmp_path / "e2_checkpoint.json"
    payload = build_e2_checkpoint_payload(
        config_digest=digest,
        smoke=False,
        dense_capability={"vector": {d: 0.0 for d in ("PPL", "Math", "Knowledge", "Reasoning", "Instruction", "Code")}},
        records=[],
        completed_cells=[e2_cell_key(42, 0.4, "oneshot")],
        started_at=1.0,
        elapsed_sec=1.0,
        partial=partial,
        imported_oneshot_seed42=True,
    )
    save_e2_checkpoint(path, payload)
    loaded = load_e2_checkpoint(path, digest)
    assert loaded is not None
    assert cell_is_completed(loaded["completed_cells"], 42, 0.4, "oneshot")
    assert not cell_is_completed(loaded["completed_cells"], 43, 0.5, "oneshot")
    assert loaded["partial"]["completed_dimensions"] == ["PPL"]

    with pytest.raises(ValueError, match="mismatch"):
        load_e2_checkpoint(path, "deadbeef")


def test_merge_e2_records_disjoint():
    a = [{"seed": 42, "target_sparsity": 0.4, "method": "iterative", "vector": {}}]
    b = [{"seed": 44, "target_sparsity": 0.4, "method": "iterative", "vector": {}}]
    merged = merge_e2_records(a, b)
    assert len(merged) == 2
    with pytest.raises(ValueError, match="duplicate"):
        merge_e2_records(a, a)
