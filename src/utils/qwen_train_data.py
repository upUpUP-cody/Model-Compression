"""SQuAD causal-LM training packs for Phase K Level-1 recovery."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.qwen_squad_eval import apply_qwen_chat_prompt, build_squad_user_message
from src.utils.squad_protocol import assert_test_not_in_selection_path


class SquadCausalLmDataset(Dataset):
    """Chat-templated prompt + answer packs; labels mask the prompt with -100."""

    def __init__(
        self,
        examples: Any,
        tokenizer,
        *,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        split_name: str = "train",
    ) -> None:
        assert_test_not_in_selection_path([split_name])
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.input_ids: List[List[int]] = []
        self.labels: List[List[int]] = []
        n = len(examples) if max_samples is None else min(len(examples), int(max_samples))
        for index in range(n):
            row = examples[index]
            user_text = build_squad_user_message(row["context"], row["question"])
            prompt = apply_qwen_chat_prompt(tokenizer, user_text, add_generation_prompt=True)
            answers = row.get("answers", {}) or {}
            texts = list(answers.get("text") or [])
            answer = texts[0] if texts else "unanswerable"
            prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
            # Leave room for answer tokens under max_seq_len.
            max_prompt = max(8, self.max_seq_len - len(answer_ids) - 1)
            if len(prompt_ids) > max_prompt:
                prompt_ids = prompt_ids[-max_prompt:]
            input_ids = prompt_ids + answer_ids
            if tokenizer.eos_token_id is not None:
                input_ids = input_ids + [int(tokenizer.eos_token_id)]
            if len(input_ids) > self.max_seq_len:
                overflow = len(input_ids) - self.max_seq_len
                keep_prompt = max(0, len(prompt_ids) - overflow)
                input_ids = prompt_ids[-keep_prompt:] + answer_ids
                if tokenizer.eos_token_id is not None:
                    input_ids = input_ids + [int(tokenizer.eos_token_id)]
                input_ids = input_ids[: self.max_seq_len]
                prompt_len = min(keep_prompt, len(input_ids))
            else:
                prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) != len(input_ids):
                labels = labels[: len(input_ids)]
            if not any(label != -100 for label in labels):
                raise ValueError("SQuAD pack has no supervised tokens; increase max_seq_len")
            self.input_ids.append([int(x) for x in input_ids])
            self.labels.append([int(x) for x in labels])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


def _pad_collate(rows: Sequence[Mapping[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    labels = []
    attention_mask = []
    for row in rows:
        ids = list(row["input_ids"])
        labs = list(row["labels"])
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(labs + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_squad_lm_loaders(
    splits: Mapping[str, Any],
    tokenizer,
    *,
    batch_size: int = 2,
    max_seq_len: int = 512,
    train_max_samples: Optional[int] = None,
    validation_max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation LM loaders from selection splits only."""
    assert_test_not_in_selection_path(list(splits.keys()))
    if "test" in splits:
        raise ValueError("frozen test must not be passed into LM loaders")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    train_ds = SquadCausalLmDataset(
        splits["train"],
        tokenizer,
        max_seq_len=max_seq_len,
        max_samples=train_max_samples,
        split_name="train",
    )
    val_ds = SquadCausalLmDataset(
        splits["validation"],
        tokenizer,
        max_seq_len=max_seq_len,
        max_samples=validation_max_samples,
        split_name="validation",
    )

    def collate(rows):
        return _pad_collate(rows, int(pad_id))

    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, num_workers=num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(batch_size), shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    return train_loader, val_loader
