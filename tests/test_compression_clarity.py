"""Lock dense baseline vs cell_target reporting so tables are not misread."""
from __future__ import annotations

from src.utils.compression_clarity import (
    SUMMARY_NOTE_DENSE_BASELINE,
    compression_clarity_fields,
    format_method_ok_line,
)


def test_dense_clarity_fields_are_unpruned_baseline() -> None:
    fields = compression_clarity_fields("dense", cell_target=1.5, actual_compression=1.0)
    assert fields["cell_target"] == 1.5
    assert fields["actual_compression"] == 1.0
    assert fields["pruning_applied"] is False


def test_pruned_methods_mark_pruning_applied() -> None:
    for method in ("oneshot", "iterative_level1", "autonomous_search"):
        fields = compression_clarity_fields(method, cell_target=2.0, actual_compression=2.0002)
        assert fields["cell_target"] == 2.0
        assert fields["actual_compression"] == 2.0002
        assert fields["pruning_applied"] is True


def test_ok_line_dense_says_baseline_no_prune() -> None:
    line = format_method_ok_line(
        method="dense",
        cell_target=1.5,
        actual_compression=1.0,
        metric_parts="f1=25.5",
    )
    assert "baseline, no prune" in line
    assert "actual=1.000x" in line
    assert "cell=1.5x" in line


def test_ok_line_oneshot_reports_actual_without_baseline_tag() -> None:
    line = format_method_ok_line(
        method="oneshot",
        cell_target=1.5,
        actual_compression=1.500153,
        metric_parts="f1=31.5",
    )
    assert "baseline" not in line
    assert "actual=1.500x" in line
    assert SUMMARY_NOTE_DENSE_BASELINE.startswith("dense is unpruned")
