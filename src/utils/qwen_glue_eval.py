"""Qwen prompt+verbalizer helpers for GLUE SST-2 / RTE / QNLI evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.glue_protocol import (
    SUPPORTED_GLUE_TASKS,
    GlueSplitBundle,
    assert_test_not_in_selection_path,
    glue_accuracy,
    label_to_verbalizer,
    load_glue_splits,
)
from src.utils.qwen_squad_eval import load_qwen_for_eval, write_json

# Locked KG eval strategy: prompt + verbalizer (not classification head).
SST2_USER_TEMPLATE = (
    "Classify the sentiment of the sentence as positive or negative.\n\n"
    "Sentence: {sentence}\n\n"
    "Reply with exactly one word: positive or negative."
)

RTE_USER_TEMPLATE = (
    "Decide whether the hypothesis is entailed by the premise.\n\n"
    "Premise: {sentence1}\n"
    "Hypothesis: {sentence2}\n\n"
    "Reply with exactly one label: entailment or not_entailment."
)

QNLI_USER_TEMPLATE = (
    "Decide whether the sentence contains the answer to the question.\n\n"
    "Question: {question}\n"
    "Sentence: {sentence}\n\n"
    "Reply with exactly one word: yes or no."
)

# Back-compat alias for docs/tests that referenced the old name.
SST2_PROMPT_TEMPLATE = SST2_USER_TEMPLATE


def build_sst2_user_message(sentence: str) -> str:
    return SST2_USER_TEMPLATE.format(sentence=str(sentence))


def build_glue_user_message(task: str, row: Mapping[str, Any]) -> str:
    key = str(task).strip().lower()
    if key == "sst2":
        return build_sst2_user_message(str(row["sentence"]))
    if key == "rte":
        return RTE_USER_TEMPLATE.format(
            sentence1=str(row["sentence1"]),
            sentence2=str(row["sentence2"]),
        )
    if key == "qnli":
        return QNLI_USER_TEMPLATE.format(
            question=str(row["question"]),
            sentence=str(row["sentence"]),
        )
    raise ValueError(f"unsupported glue task: {task}; expected one of {SUPPORTED_GLUE_TASKS}")


def apply_qwen_chat_prompt(
    tokenizer,
    user_text: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Wrap a user message with the tokenizer chat template (Instruct models)."""
    messages = [{"role": "user", "content": str(user_text)}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
    # Fallback for unit-test fakes without chat templates.
    suffix = "\nAssistant:" if add_generation_prompt else ""
    return f"User: {user_text}{suffix}"


@torch.no_grad()
def generate_glue_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: str,
    max_seq_len: int,
    max_new_tokens: int = 8,
) -> List[str]:
    """Greedy decode for GLUE verbalizers; keeps SQuAD generate_answers untouched."""
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


class GluePromptDataset(Dataset):
    """Yields chat-templated prompts; requires tokenizer at build time."""

    def __init__(
        self,
        examples: Any,
        tokenizer,
        *,
        task: str = "sst2",
        max_samples: Optional[int] = None,
    ) -> None:
        key = str(task).strip().lower()
        if key not in SUPPORTED_GLUE_TASKS:
            raise ValueError(f"unsupported glue task: {task}")
        self.task = key
        self.ids: List[str] = []
        self.prompts: List[str] = []
        self.labels: List[int] = []
        n = len(examples) if max_samples is None else min(len(examples), int(max_samples))
        for index in range(n):
            row = examples[index]
            example_id = str(row.get("idx", index))
            label = int(row["label"])
            user_text = build_glue_user_message(self.task, row)
            prompt = apply_qwen_chat_prompt(tokenizer, user_text, add_generation_prompt=True)
            self.ids.append(example_id)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "id": self.ids[index],
            "prompt": self.prompts[index],
            "label": self.labels[index],
        }


def _collate(rows: Sequence[Mapping[str, object]]) -> Dict[str, list]:
    return {
        "id": [row["id"] for row in rows],
        "prompt": [row["prompt"] for row in rows],
        "label": [row["label"] for row in rows],
    }


def evaluate_glue_split(
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
    task: str = "sst2",
) -> Dict[str, float]:
    assert_test_not_in_selection_path([split_name])
    dataset = GluePromptDataset(examples, tokenizer, task=task, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=_collate)
    predictions: Dict[str, str] = {}
    references: Dict[str, int] = {}
    gen_tokens = max(1, int(max_new_tokens))
    for batch in loader:
        answers = generate_glue_answers(
            model,
            tokenizer,
            batch["prompt"],
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=gen_tokens,
        )
        for example_id, prediction, label in zip(batch["id"], answers, batch["label"]):
            predictions[str(example_id)] = prediction
            references[str(example_id)] = int(label)
    metrics = glue_accuracy(predictions, references, task=task)
    metrics["split"] = split_name  # type: ignore[assignment]
    metrics["task"] = task  # type: ignore[assignment]
    return metrics


def prepare_glue_from_config(config: Mapping[str, Any]) -> GlueSplitBundle:
    dataset_cfg = config["dataset"]
    task = str(dataset_cfg.get("task", "sst2"))
    return load_glue_splits(
        str(dataset_cfg["cache_dir"]),
        task=task,
        validation_fraction=float(dataset_cfg.get("validation_fraction", 0.1)),
        split_seed=int(dataset_cfg.get("split_seed", config.get("seed", 42))),
    )


__all__ = [
    "SST2_PROMPT_TEMPLATE",
    "SST2_USER_TEMPLATE",
    "RTE_USER_TEMPLATE",
    "QNLI_USER_TEMPLATE",
    "GluePromptDataset",
    "apply_qwen_chat_prompt",
    "build_glue_user_message",
    "build_sst2_user_message",
    "evaluate_glue_split",
    "generate_glue_answers",
    "label_to_verbalizer",
    "load_qwen_for_eval",
    "prepare_glue_from_config",
    "write_json",
]
