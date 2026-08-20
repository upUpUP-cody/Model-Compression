"""Qwen extractive-QA helpers for SQuAD smoke evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.squad_protocol import (
    SquadSplitBundle,
    aggregate_squad_metrics,
    assert_test_not_in_selection_path,
    load_squad_v2_splits,
)


PROMPT_TEMPLATE = (
    "Answer the question using only the context. "
    "If the answer is not in the context, reply with \"unanswerable\".\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# User-message body for Instruct chat template (generation prompt appended separately).
SQUAD_USER_TEMPLATE = (
    "Answer the question using only the context. "
    "If the answer is not in the context, reply with \"unanswerable\".\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)


def build_squad_user_message(context: str, question: str) -> str:
    return SQUAD_USER_TEMPLATE.format(context=str(context), question=str(question))


def apply_qwen_chat_prompt(
    tokenizer,
    user_text: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    messages = [{"role": "user", "content": str(user_text)}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
    suffix = "\nAssistant:" if add_generation_prompt else ""
    return f"User: {user_text}{suffix}"


class SquadPromptDataset(Dataset):
    def __init__(
        self,
        examples: Any,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.examples = examples
        self.ids: List[str] = []
        self.prompts: List[str] = []
        self.answers: List[List[str]] = []
        n = len(examples) if max_samples is None else min(len(examples), int(max_samples))
        for index in range(n):
            row = examples[index]
            example_id = str(row["id"])
            answers = row.get("answers", {}) or {}
            texts = list(answers.get("text") or [])
            user_text = build_squad_user_message(row["context"], row["question"])
            if tokenizer is not None:
                prompt = apply_qwen_chat_prompt(tokenizer, user_text, add_generation_prompt=True)
            else:
                # Back-compat for unit tests without a tokenizer.
                prompt = PROMPT_TEMPLATE.format(context=row["context"], question=row["question"])
            self.ids.append(example_id)
            self.prompts.append(prompt)
            self.answers.append([str(text) for text in texts])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "id": self.ids[index],
            "prompt": self.prompts[index],
            "answers": self.answers[index],
        }


def load_qwen_for_eval(model_path: str, device: str, torch_dtype: str = "float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(str(torch_dtype), torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        # Prefer SDPA/default: attn_implementation="eager" yields garbage generations on this stack.
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> List[str]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seq_len),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    answers: List[str] = []
    for row in output_ids:
        text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
        answers.append(text.strip().split("\n")[0].strip())
    return answers


def evaluate_squad_split(
    model,
    tokenizer,
    examples: Any,
    *,
    device: str,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    max_samples: Optional[int] = None,
    split_name: str = "validation",
    allow_frozen_test: bool = False,
) -> Dict[str, float]:
    if not allow_frozen_test:
        assert_test_not_in_selection_path([split_name])
    dataset = SquadPromptDataset(examples, tokenizer=tokenizer, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=_collate)
    predictions: Dict[str, str] = {}
    references: Dict[str, List[str]] = {}
    for batch in loader:
        answers = generate_answers(
            model,
            tokenizer,
            batch["prompt"],
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
        )
        for example_id, prediction, truths in zip(batch["id"], answers, batch["answers"]):
            predictions[str(example_id)] = prediction
            references[str(example_id)] = list(truths)
    metrics = aggregate_squad_metrics(predictions, references)
    metrics["split"] = split_name  # type: ignore[assignment]
    return metrics


def _collate(rows: Sequence[Mapping[str, object]]) -> Dict[str, list]:
    return {
        "id": [row["id"] for row in rows],
        "prompt": [row["prompt"] for row in rows],
        "answers": [row["answers"] for row in rows],
    }


def prepare_squad_from_config(config: Mapping[str, Any]) -> SquadSplitBundle:
    dataset_cfg = config["dataset"]
    return load_squad_v2_splits(
        str(dataset_cfg["cache_dir"]),
        validation_fraction=float(dataset_cfg.get("validation_fraction", 0.1)),
        split_seed=int(dataset_cfg.get("split_seed", config.get("seed", 42))),
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
