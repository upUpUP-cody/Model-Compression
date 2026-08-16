"""SQuAD 2.0 train / validation / frozen-test split protocol for Phase K."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SquadSplitBundle:
    """Official SQuAD validation is frozen test; val is carved from train."""

    train: Any
    validation: Any
    test: Any
    split_seed: int
    validation_fraction: float
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]

    def selection_splits(self) -> Dict[str, Any]:
        """Splits allowed for training / search / model selection."""
        return {"train": self.train, "validation": self.validation}

    def metadata(self) -> Dict[str, object]:
        return {
            "split_seed": int(self.split_seed),
            "validation_fraction": float(self.validation_fraction),
            "n_train": len(self.train_indices),
            "n_validation": len(self.validation_indices),
            "n_test": len(self.test),
            "test_source": "squad_v2.official_validation",
            "selection_may_use_test": False,
        }


def _require_mapping(dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("SQuAD dataset must contain 'train' and 'validation' splits")
    return dataset


def split_squad_train_validation_test(
    dataset: Mapping[str, Any],
    *,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> SquadSplitBundle:
    """
    Carve validation from official train; keep official validation as frozen test.

    Selection / search / early-stop must use only train+validation. Test is for
    a single post-freeze report.
    """
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

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
    return SquadSplitBundle(
        train=train_split,
        validation=validation_split,
        test=frozen_test,
        split_seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def assert_test_not_in_selection_path(used_split_names: Sequence[str]) -> None:
    """Raise if a caller tries to put frozen test on the selection path."""
    forbidden = {
        "test",
        "official_validation",
        "squad_v2.official_validation",
        "glue.sst2.official_validation",
        "glue.rte.official_validation",
        "glue.qnli.official_validation",
    }
    overlap = forbidden.intersection({str(name).lower() for name in used_split_names})
    if overlap:
        raise ValueError(
            f"frozen test split must not be used for selection/search; got {sorted(overlap)}"
        )


def load_squad_v2_splits(
    cache_dir: str,
    *,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> SquadSplitBundle:
    from datasets import load_dataset

    raw = load_dataset("rajpurkar/squad_v2", cache_dir=cache_dir)
    return split_squad_train_validation_test(
        raw,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )


def normalize_squad_prediction(text: str) -> str:
    import re
    import string

    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def squad_em(prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        return float(normalize_squad_prediction(prediction) == "")
    norm_pred = normalize_squad_prediction(prediction)
    return float(any(norm_pred == normalize_squad_prediction(gt) for gt in ground_truths))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_squad_prediction(prediction).split()
    truth_tokens = normalize_squad_prediction(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in truth_tokens:
        count = common.get(token, 0)
        if count > 0:
            common[token] = count - 1
            num_same += 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        return float(normalize_squad_prediction(prediction) == "")
    return max(_f1_score(prediction, gt) for gt in ground_truths)


def aggregate_squad_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    if set(predictions) != set(references):
        missing = set(references) - set(predictions)
        extra = set(predictions) - set(references)
        raise ValueError(f"prediction/reference id mismatch; missing={len(missing)} extra={len(extra)}")
    em_scores = []
    f1_scores = []
    for example_id, prediction in predictions.items():
        truths = list(references[example_id])
        em_scores.append(squad_em(prediction, truths))
        f1_scores.append(squad_f1(prediction, truths))
    n = len(em_scores)
    return {
        "exact_match": 100.0 * float(sum(em_scores) / n) if n else 0.0,
        "f1": 100.0 * float(sum(f1_scores) / n) if n else 0.0,
        "n_examples": float(n),
    }
