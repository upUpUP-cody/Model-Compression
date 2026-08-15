"""Unit tests for SQuAD train/val/test protocol (no network)."""
from __future__ import annotations

import pytest

from src.utils.squad_protocol import (
    assert_test_not_in_selection_path,
    split_squad_train_validation_test,
)


class _FakeSplit(list):
    def select(self, indices):
        return _FakeSplit(self[i] for i in indices)


def _fake_squad(n_train: int = 100, n_val: int = 20):
    train = _FakeSplit({"id": f"t{i}", "context": "c", "question": "q"} for i in range(n_train))
    validation = _FakeSplit(
        {"id": f"v{i}", "context": "c", "question": "q"} for i in range(n_val)
    )
    return {"train": train, "validation": validation}


def test_split_reproducible_with_same_seed():
    raw = _fake_squad()
    a = split_squad_train_validation_test(raw, validation_fraction=0.1, split_seed=42)
    b = split_squad_train_validation_test(raw, validation_fraction=0.1, split_seed=42)
    assert a.train_indices == b.train_indices
    assert a.validation_indices == b.validation_indices
    assert len(a.train) + len(a.validation) == 100
    assert len(a.test) == 20


def test_split_changes_with_seed():
    raw = _fake_squad()
    a = split_squad_train_validation_test(raw, validation_fraction=0.1, split_seed=1)
    b = split_squad_train_validation_test(raw, validation_fraction=0.1, split_seed=2)
    assert a.validation_indices != b.validation_indices


def test_selection_path_excludes_test():
    raw = _fake_squad()
    bundle = split_squad_train_validation_test(raw, validation_fraction=0.1, split_seed=42)
    selection = bundle.selection_splits()
    assert set(selection.keys()) == {"train", "validation"}
    assert "test" not in selection
    assert_test_not_in_selection_path(list(selection.keys()))


def test_assert_blocks_test_on_selection_path():
    with pytest.raises(ValueError, match="frozen test"):
        assert_test_not_in_selection_path(["train", "test"])
