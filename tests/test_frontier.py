import json

import pytest

from src.evaluation.frontier import FrontierPoint, ParetoFrontier


def point(accuracy, params, compression=1.0, **kwargs):
    return FrontierPoint(accuracy, params, compression, **kwargs)


def test_archive_keeps_only_non_dominated_points_and_sorts_by_size():
    archive = ParetoFrontier([
        point(90.0, 100, 2.0),
        point(91.0, 120, 1.8),
        point(89.0, 80, 2.5),
        point(90.5, 110, 1.9),
    ])

    assert [(item.parameter_count, item.validation_accuracy) for item in archive] == [
        (80, 89.0),
        (100, 90.0),
        (110, 90.5),
        (120, 91.0),
    ]
    assert archive.add(point(88.0, 130, 1.5)) is False


def test_equal_point_is_deduplicated_and_equal_size_lower_accuracy_is_rejected():
    archive = ParetoFrontier()
    first = point(90.0, 100, 2.0, run="a")
    assert archive.add(first) is True
    assert archive.add(first) is False
    assert archive.add(point(89.0, 100, 2.0, run="b")) is False
    assert archive.add(point(91.0, 100, 2.0, run="b")) is True
    assert len(archive) == 1
    assert archive.points[0].validation_accuracy == 91.0


def test_queries_return_none_for_empty_or_ineligible_archive():
    archive = ParetoFrontier()
    assert archive.best_under_accuracy_drop(90.0, 1.0) is None
    assert archive.best_under_parameter_budget(100) is None
    archive.add(point(88.0, 50, 3.0))
    assert archive.best_under_accuracy_drop(90.0, 1.0) is None
    assert archive.best_under_parameter_budget(49) is None


def test_queries_use_accuracy_drop_and_parameter_budget():
    archive = ParetoFrontier([
        point(90.0, 100, 2.0),
        point(89.5, 70, 2.8),
        point(91.0, 150, 1.3),
    ])
    assert archive.best_under_accuracy_drop(91.0, 1.0).parameter_count == 100
    assert archive.best_under_parameter_budget(100).validation_accuracy == 90.0


def test_archive_round_trips_as_json():
    archive = ParetoFrontier([
        point(90.0, 100, 2.0, candidate_spec={"layer": [1, 2]}, seed=7, run="r1", iteration=3)
    ])
    encoded = archive.to_json()
    restored = ParetoFrontier.from_dict(json.loads(encoded))
    assert restored.to_dict() == archive.to_dict()


def test_invalid_point_and_query_limits_are_rejected():
    with pytest.raises(ValueError):
        point(90.0, -1, 2.0)
    with pytest.raises(ValueError):
        point(90.0, 10, 0.0)
    archive = ParetoFrontier()
    with pytest.raises(ValueError):
        archive.best_under_accuracy_drop(90.0, -1)
    with pytest.raises(ValueError):
        archive.best_under_parameter_budget(-1)
