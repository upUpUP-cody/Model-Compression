"""Unit tests for Stage C sampling / resume helpers (no GPU download)."""
from __future__ import annotations

from pathlib import Path

from src.experiments.stage_c_common import (
    cell_done,
    recovery_gain,
    save_cell,
    select_indices,
)


def _fake_scored(n: int = 20):
    rows = []
    for i in range(n):
        rows.append(
            {
                "index": float(i),
                "teacher_nll": float(i),
                "student_nll": float(n - i),
                "gap": float((n - i) - i),
                "length": 10.0,
            }
        )
    return rows


def test_select_high_gap_picks_largest_gap():
    scored = _fake_scored(10)
    # gap = (10-i)-i = 10-2i; largest at i=0
    idx = select_indices(scored, "high_gap", 3, seed=0)
    assert idx == [0, 1, 2]


def test_select_low_gap_picks_smallest_gap():
    scored = _fake_scored(10)
    idx = select_indices(scored, "low_gap", 2, seed=0)
    assert idx == [9, 8]


def test_select_random_deterministic():
    scored = _fake_scored(20)
    a = select_indices(scored, "random", 5, seed=42)
    b = select_indices(scored, "random", 5, seed=42)
    c = select_indices(scored, "random", 5, seed=43)
    assert a == b
    assert a != c


def test_cell_resume_roundtrip(tmp_path: Path):
    save_cell(tmp_path, "random_n256_seed42", {"recovery_gain": 0.1})
    assert cell_done(tmp_path, "random_n256_seed42")
    assert not cell_done(tmp_path, "missing")


def test_recovery_gain_definition():
    # lower loss -> higher perf; gain = (-rec) - (-comp) = comp - rec
    assert abs(recovery_gain({"loss": 2.0}, {"loss": 1.0}) - 1.0) < 1e-9
