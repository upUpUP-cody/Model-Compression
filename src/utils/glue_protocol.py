"""GLUE train / validation / frozen-test split protocol for Phase K KG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.utils.squad_protocol import assert_test_not_in_selection_path

SUPPORTED_GLUE_TASKS = ("sst2", "rte", "qnli")

SST2_LABEL_TO_TEXT = {0: "negative", 1: "positive"}
SST2_TEXT_TO_LABEL = {value: key for key, value in SST2_LABEL_TO_TEXT.items()}

RTE_LABEL_TO_TEXT = {0: "entailment", 1: "not_entailment"}
RTE_TEXT_TO_LABEL = {value: key for key, value in RTE_LABEL_TO_TEXT.items()}

# Locked Phase K verbalizer (not HF string names): 0=yes, 1=no.
QNLI_LABEL_TO_TEXT = {0: "yes", 1: "no"}
QNLI_TEXT_TO_LABEL = {value: key for key, value in QNLI_LABEL_TO_TEXT.items()}

_TASK_LABEL_TO_TEXT = {
    "sst2": SST2_LABEL_TO_TEXT,
    "rte": RTE_LABEL_TO_TEXT,
    "qnli": QNLI_LABEL_TO_TEXT,
}
_TASK_TEXT_TO_LABEL = {
    "sst2": SST2_TEXT_TO_LABEL,
    "rte": RTE_TEXT_TO_LABEL,
    "qnli": QNLI_TEXT_TO_LABEL,
}


@dataclass(frozen=True)
class GlueSplitBundle:
    """Official GLUE validation is frozen test; val is carved from train."""

    train: Any
    validation: Any
    test: Any
    task: str
    split_seed: int
    validation_fraction: float
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]

    def selection_splits(self) -> Dict[str, Any]:
        return {"train": self.train, "validation": self.validation}

    def metadata(self) -> Dict[str, object]:
        task = str(self.task)
        return {
            "task": task,
            "split_seed": int(self.split_seed),
            "validation_fraction": float(self.validation_fraction),
            "n_train": len(self.train_indices),
            "n_validation": len(self.validation_indices),
            "n_test": len(self.test),
            "test_source": f"glue.{task}.official_validation",
            "selection_may_use_test": False,
            "eval_strategy": "prompt_verbalizer",
        }


def _normalize_task(task: str) -> str:
    key = str(task).strip().lower()
    if key not in _TASK_LABEL_TO_TEXT:
        raise ValueError(f"unsupported glue task: {task}; expected one of {SUPPORTED_GLUE_TASKS}")
    return key


def _require_mapping(dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("GLUE dataset must contain 'train' and 'validation' splits")
    return dataset


def split_glue_train_validation_test(
    dataset: Mapping[str, Any],
    *,
    task: str = "sst2",
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> GlueSplitBundle:
    """
    Carve validation from official train; keep official validation as frozen test.

    Official GLUE `test` is often unlabeled and is not used.
    """
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    task = _normalize_task(task)
    dataset = _require_mapping(dataset)
    full_train = dataset["train"]
    frozen_test = dataset["validation"]
    total = len(full_train)
    validation_size = int(total * float(validation_fraction))
    if validation_size < 1 or validation_size >= total:
        raise ValueError("validation_fraction yields empty train or validation")

    rng = np.random.default_rng(int(split_seed))
    permutation = rng.permutation(total).tolist()
    validation_indices = tuple(int(i) for i in permutation[:validation_size])
    train_indices = tuple(int(i) for i in permutation[validation_size:])

    train_split = full_train.select(list(train_indices))
    validation_split = full_train.select(list(validation_indices))
    return GlueSplitBundle(
        train=train_split,
        validation=validation_split,
        test=frozen_test,
        task=task,
        split_seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def load_glue_splits(
    cache_dir: str,
    *,
    task: str = "sst2",
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> GlueSplitBundle:
    from datasets import load_dataset

    task = _normalize_task(task)
    raw = load_dataset("nyu-mll/glue", task, cache_dir=cache_dir)
    return split_glue_train_validation_test(
        raw,
        task=task,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )


def load_glue_sst2_splits(
    cache_dir: str,
    *,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> GlueSplitBundle:
    return load_glue_splits(
        cache_dir,
        task="sst2",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )


def label_to_verbalizer(label: int, task: str = "sst2") -> str:
    task = _normalize_task(task)
    mapping = _TASK_LABEL_TO_TEXT[task]
    if int(label) not in mapping:
        raise ValueError(f"unexpected {task} label: {label}")
    return mapping[int(label)]


def normalize_glue_prediction(text: str, task: str = "sst2") -> str:
    task = _normalize_task(task)
    raw = str(text).strip().lower()
    if not raw:
        return ""
    text_to_label = _TASK_TEXT_TO_LABEL[task]

    # Multi-token verbalizers first (avoid splitting "not_entailment").
    if task == "rte":
        if "not_entailment" in raw or "not entailment" in raw:
            return "not_entailment"
        if "entailment" in raw:
            return "entailment"

    tokens = raw.replace("_", " ").split()
    first = tokens[0].strip(".,;:!?\"'") if tokens else ""
    if first in text_to_label:
        return first
    # QNLI / SST-2 single-word scan.
    for verbalizer in text_to_label:
        if verbalizer in raw:
            # Prefer exclusive match when possible.
            others = [v for v in text_to_label if v != verbalizer]
            if not any(other in raw for other in others):
                return verbalizer
    for verbalizer in text_to_label:
        if verbalizer in raw:
            return verbalizer
    return first


def prediction_to_label(text: str, task: str = "sst2") -> Optional[int]:
    task = _normalize_task(task)
    normalized = normalize_glue_prediction(text, task=task)
    return _TASK_TEXT_TO_LABEL[task].get(normalized)


def glue_accuracy(
    predictions: Mapping[str, str],
    references: Mapping[str, int],
    *,
    task: str = "sst2",
) -> Dict[str, float]:
    task = _normalize_task(task)
    if set(predictions) != set(references):
        missing = set(references) - set(predictions)
        extra = set(predictions) - set(references)
        raise ValueError(f"prediction/reference id mismatch; missing={len(missing)} extra={len(extra)}")
    correct = 0
    n = 0
    for example_id, prediction in predictions.items():
        pred_label = prediction_to_label(prediction, task=task)
        truth = int(references[example_id])
        if pred_label is not None and pred_label == truth:
            correct += 1
        n += 1
    return {
        "accuracy": 100.0 * float(correct / n) if n else 0.0,
        "n_examples": float(n),
    }


__all__ = [
    "GlueSplitBundle",
    "QNLI_LABEL_TO_TEXT",
    "RTE_LABEL_TO_TEXT",
    "SST2_LABEL_TO_TEXT",
    "SUPPORTED_GLUE_TASKS",
    "assert_test_not_in_selection_path",
    "glue_accuracy",
    "label_to_verbalizer",
    "load_glue_splits",
    "load_glue_sst2_splits",
    "normalize_glue_prediction",
    "prediction_to_label",
    "split_glue_train_validation_test",
]
