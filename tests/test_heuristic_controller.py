import json

import pytest

from src.controller.heuristic_controller import CandidateProfile, HeuristicController


def profile(**overrides):
    values = {
        "fingerprint": "candidate-a",
        "quality": 0.8,
        "accuracy": 94.0,
        "parent_accuracy": 96.0,
        "parameter_count": 80,
        "parent_parameter_count": 100,
    }
    values.update(overrides)
    return CandidateProfile(**values)


@pytest.mark.parametrize(
    ("candidate", "history", "expected_action", "expected_reason"),
    [
        (profile(), {"consecutive_failures": 0}, "accept", "constraints_satisfied"),
        (profile(parameter_count=100), {}, "reject", "no_parameter_reduction"),
        (profile(), {"consecutive_failures": 3}, "rollback", "failure_limit_reached"),
        (profile(accuracy=88.0), {}, "regrow", "capability_gap_exceeded"),
        (profile(quality=0.2), {}, "regrow", "quality_below_threshold"),
    ],
)
def test_decision_rules(candidate, history, expected_action, expected_reason):
    controller = HeuristicController(max_accuracy_drop_points=5.0, max_failures=3)

    decision = controller.decide_action(candidate, history=history)

    assert decision.action == expected_action
    assert decision.reason == expected_reason
    assert json.loads(json.dumps(decision.to_dict()))["action"] == expected_action


def test_duplicate_candidate_is_rejected_before_other_rules():
    controller = HeuristicController()
    first = controller.decide_action(profile())
    second = controller.decide_action(profile(quality=0.0), history={"consecutive_failures": 9})

    assert first.action == "accept"
    assert second.action == "reject"
    assert second.reason == "duplicate_candidate"


def test_dict_profile_and_candidate_overrides_are_supported():
    controller = HeuristicController()
    decision = controller.decide_action(
        {
            "fingerprint": "base",
            "quality": 0.9,
            "accuracy": 95.0,
            "parent_accuracy": 96.0,
            "parameter_count": 90,
            "parent_parameter_count": 100,
        },
        candidate={"fingerprint": "override", "parameter_count": 80},
    )

    assert decision.action == "accept"
    assert decision.fingerprint == "override"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_accuracy_drop_points": -1.0},
        {"max_failures": 0},
        {"min_quality_score": 1.1},
        {"regrow_ratio_multiplier": 1.0},
    ],
)
def test_invalid_controller_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HeuristicController(**kwargs)
