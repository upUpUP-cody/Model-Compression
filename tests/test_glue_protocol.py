"""Unit tests for GLUE train/val/test protocol (no network)."""
from __future__ import annotations

import pytest

from src.utils.glue_protocol import (
    assert_test_not_in_selection_path,
    glue_accuracy,
    label_to_verbalizer,
    normalize_glue_prediction,
    split_glue_train_validation_test,
)


class _FakeSplit(list):
    def select(self, indices):
        return _FakeSplit(self[i] for i in indices)


def _fake_sst2(n_train: int = 100, n_val: int = 20):
    train = _FakeSplit(
        {"idx": i, "sentence": f"good movie {i}", "label": int(i % 2)} for i in range(n_train)
    )
    validation = _FakeSplit(
        {"idx": i, "sentence": f"bad movie {i}", "label": int(i % 2)} for i in range(n_val)
    )
    return {"train": train, "validation": validation}


def test_split_reproducible_with_same_seed():
    raw = _fake_sst2()
    a = split_glue_train_validation_test(raw, task="sst2", validation_fraction=0.1, split_seed=42)
    b = split_glue_train_validation_test(raw, task="sst2", validation_fraction=0.1, split_seed=42)
    assert a.train_indices == b.train_indices
    assert a.validation_indices == b.validation_indices
    assert len(a.train) + len(a.validation) == 100
    assert len(a.test) == 20
    assert a.metadata()["test_source"] == "glue.sst2.official_validation"
    assert a.metadata()["selection_may_use_test"] is False


def test_split_changes_with_seed():
    raw = _fake_sst2()
    a = split_glue_train_validation_test(raw, validation_fraction=0.1, split_seed=1)
    b = split_glue_train_validation_test(raw, validation_fraction=0.1, split_seed=2)
    assert a.validation_indices != b.validation_indices


def test_selection_path_excludes_test():
    raw = _fake_sst2()
    bundle = split_glue_train_validation_test(raw, validation_fraction=0.1, split_seed=42)
    selection = bundle.selection_splits()
    assert set(selection.keys()) == {"train", "validation"}
    assert "test" not in selection
    assert_test_not_in_selection_path(list(selection.keys()))


def test_assert_blocks_glue_official_validation_name():
    with pytest.raises(ValueError, match="frozen test"):
        assert_test_not_in_selection_path(["train", "glue.sst2.official_validation"])


def test_verbalizer_and_accuracy():
    assert label_to_verbalizer(0) == "negative"
    assert label_to_verbalizer(1) == "positive"
    assert normalize_glue_prediction("Positive!") == "positive"
    assert normalize_glue_prediction("The answer is negative.") == "negative"
    metrics = glue_accuracy(
        {"a": "positive", "b": "negative"},
        {"a": 1, "b": 0},
    )
    assert metrics["accuracy"] == 100.0
    assert metrics["n_examples"] == 2.0


def test_rte_verbalizer_prefers_not_entailment():
    assert label_to_verbalizer(0, task="rte") == "entailment"
    assert label_to_verbalizer(1, task="rte") == "not_entailment"
    assert normalize_glue_prediction("not_entailment", task="rte") == "not_entailment"
    assert normalize_glue_prediction("The label is not_entailment.", task="rte") == "not_entailment"
    assert normalize_glue_prediction("entailment", task="rte") == "entailment"
    metrics = glue_accuracy(
        {"a": "entailment", "b": "not_entailment"},
        {"a": 0, "b": 1},
        task="rte",
    )
    assert metrics["accuracy"] == 100.0


def test_qnli_verbalizer_yes_no():
    assert label_to_verbalizer(0, task="qnli") == "yes"
    assert label_to_verbalizer(1, task="qnli") == "no"
    assert normalize_glue_prediction("Yes", task="qnli") == "yes"
    assert normalize_glue_prediction("answer: no", task="qnli") == "no"
    metrics = glue_accuracy(
        {"a": "yes", "b": "no"},
        {"a": 0, "b": 1},
        task="qnli",
    )
    assert metrics["accuracy"] == 100.0
