"""Tests for Qwen MLP Wanda importance and Stage A Wanda prune."""
from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.experiments.qwen_k5_comparison import (
    count_params,
    magnitude_importance_mlp,
    mlp_only_layer_names,
    wanda_importance_mlp,
)
from src.experiments.stage_a_common import prune_mlp_wanda
from src.pruning.wanda import wanda_importance_mlp as wanda_export


def _tiny_qwen():
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _lm_loader(vocab_size: int = 128, seq_len: int = 8, batches: int = 2, batch_size: int = 1):
    rows = []
    for _ in range(batches * batch_size):
        ids = torch.randint(2, vocab_size, (seq_len,), dtype=torch.long)
        mask = torch.ones_like(ids)
        rows.append((ids, mask))
    input_ids = torch.stack([r[0] for r in rows])
    attention_mask = torch.stack([r[1] for r in rows])
    ds = TensorDataset(input_ids, attention_mask)
    return DataLoader(ds, batch_size=batch_size, collate_fn=lambda batch: {
        "input_ids": torch.stack([b[0] for b in batch]),
        "attention_mask": torch.stack([b[1] for b in batch]),
        "labels": torch.stack([b[0] for b in batch]),
    })


def test_wanda_export_matches_impl():
    assert wanda_export is wanda_importance_mlp


def test_wanda_importance_mlp_shapes_all_layers():
    model = _tiny_qwen()
    loader = _lm_loader()
    names = mlp_only_layer_names(model)
    scores = wanda_importance_mlp(model, loader, "cpu", num_batches=2)
    assert set(scores.keys()) == set(names)
    for name in names:
        assert scores[name].numel() == 128
        assert torch.all(scores[name] >= 0)


def test_wanda_scores_change_with_activation():
    model = _tiny_qwen()
    loader = _lm_loader()
    mag_only = magnitude_importance_mlp(model, max_layers=None)
    wanda = wanda_importance_mlp(model, loader, "cpu", num_batches=2)
    # Wanda includes activation factor; should not be identical to pure magnitude.
    diffs = [torch.max(torch.abs(wanda[n] - mag_only[n])).item() for n in mag_only]
    assert any(d > 0 for d in diffs)


def test_prune_mlp_wanda_reduces_params():
    model = _tiny_qwen()
    before = count_params(model)
    loader = _lm_loader()
    pruned = prune_mlp_wanda(model, 0.5, loader, "cpu", num_batches=2)
    after = count_params(pruned)
    assert after < before
    pruner_names = mlp_only_layer_names(pruned)
    assert len(pruner_names) == 2


def test_prune_mlp_wanda_keep_count():
    model = _tiny_qwen()
    loader = _lm_loader()
    pruned = prune_mlp_wanda(copy.deepcopy(model), 0.5, loader, "cpu", num_batches=2)
    layer0 = pruned.model.layers[0].mlp
    kept = int(layer0.intermediate_size)
    assert kept == max(1, int(round(128 * 0.5)))


def test_wanda_rejects_empty_loader():
    model = _tiny_qwen()

    class _EmptyLoader:
        def __iter__(self):
            return iter([])

    with pytest.raises(ValueError, match="at least one batch"):
        wanda_importance_mlp(model, _EmptyLoader(), "cpu", num_batches=1)
