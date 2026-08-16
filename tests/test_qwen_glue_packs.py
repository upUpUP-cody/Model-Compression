"""Unit tests for GLUE LM packs (SST-2 / RTE / QNLI; no network / no model download)."""
from __future__ import annotations

import pytest

from src.utils.qwen_glue_eval import apply_qwen_chat_prompt, build_glue_user_message, build_sst2_user_message
from src.utils.qwen_glue_train_data import GlueCausalLmDataset, build_glue_lm_loaders


class _FakeTok:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        suffix = "\nAssistant:" if add_generation_prompt else ""
        return f"<|im_start|>user\n{content}<|im_end|>{suffix}"

    def __call__(self, text, add_special_tokens=False):
        ids = [max(2, ord(ch) % 120) for ch in str(text)[:80]]
        return {"input_ids": ids or [2]}


class _FakeSplit(list):
    def select(self, indices):
        return _FakeSplit(self[i] for i in indices)


def _fake_sst2(n: int = 8):
    rows = []
    for i in range(n):
        rows.append({"idx": i, "sentence": f"A fine film number {i}.", "label": int(i % 2)})
    return _FakeSplit(rows)


def _fake_rte(n: int = 4):
    rows = []
    for i in range(n):
        rows.append(
            {
                "idx": i,
                "sentence1": f"Premise {i}.",
                "sentence2": f"Hypothesis {i}.",
                "label": int(i % 2),
            }
        )
    return _FakeSplit(rows)


def _fake_qnli(n: int = 4):
    rows = []
    for i in range(n):
        rows.append(
            {
                "idx": i,
                "question": f"What is {i}?",
                "sentence": f"Sentence about {i}.",
                "label": int(i % 2),
            }
        )
    return _FakeSplit(rows)


def test_chat_prompt_wraps_user_message():
    tok = _FakeTok()
    user = build_sst2_user_message("great movie")
    prompt = apply_qwen_chat_prompt(tok, user, add_generation_prompt=True)
    assert "great movie" in prompt
    assert "Assistant:" in prompt


def test_build_glue_user_message_all_tasks():
    assert "positive or negative" in build_glue_user_message("sst2", {"sentence": "ok"})
    rte = build_glue_user_message("rte", {"sentence1": "p", "sentence2": "h"})
    assert "Premise: p" in rte and "not_entailment" in rte
    qnli = build_glue_user_message("qnli", {"question": "q?", "sentence": "s"})
    assert "Question: q?" in qnli and "yes or no" in qnli


def test_glue_causal_dataset_masks_prompt():
    tok = _FakeTok()
    ds = GlueCausalLmDataset(_fake_sst2(3), tok, max_seq_len=128, max_samples=3, split_name="train")
    assert len(ds) == 3
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])
    assert any(label == -100 for label in item["labels"])
    assert any(label != -100 for label in item["labels"])


@pytest.mark.parametrize("task,factory", [("sst2", _fake_sst2), ("rte", _fake_rte), ("qnli", _fake_qnli)])
def test_glue_packs_all_tasks_have_supervised_tokens(task, factory):
    tok = _FakeTok()
    ds = GlueCausalLmDataset(factory(2), tok, task=task, max_seq_len=128, max_samples=2, split_name="train")
    assert len(ds) == 2
    assert any(label != -100 for label in ds[0]["labels"])


def test_glue_pack_keeps_supervised_tokens_under_short_limit():
    tok = _FakeTok()
    ds = GlueCausalLmDataset(_fake_sst2(1), tok, max_seq_len=16, max_samples=1, split_name="train")
    assert any(label != -100 for label in ds[0]["labels"])


def test_glue_loaders_reject_test_split():
    tok = _FakeTok()
    with pytest.raises(ValueError, match="frozen test"):
        build_glue_lm_loaders(
            {"train": _fake_sst2(4), "validation": _fake_sst2(2), "test": _fake_sst2(2)},
            tok,
            batch_size=2,
            max_seq_len=128,
        )


def test_glue_loaders_build_ok():
    tok = _FakeTok()
    train_loader, val_loader = build_glue_lm_loaders(
        {"train": _fake_sst2(6), "validation": _fake_sst2(4)},
        tok,
        batch_size=2,
        max_seq_len=128,
        train_max_samples=6,
        validation_max_samples=4,
    )
    batch = next(iter(train_loader))
    assert "input_ids" in batch and "labels" in batch and "attention_mask" in batch
    assert len(val_loader) >= 1
