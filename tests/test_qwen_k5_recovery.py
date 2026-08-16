"""Unit tests for Qwen LM recovery / K5 wiring (tiny random model, no download)."""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.experiments.qwen_k5_comparison import magnitude_importance_mlp, mlp_only_layer_names
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.qwen_lm_recovery import evaluate_lm_loss, quick_lm_recovery, run_configured_recovery
from src.recovery.qwen_lora_recovery import quick_lora_recovery
from src.utils.qwen_train_data import SquadCausalLmDataset, build_squad_lm_loaders
from src.utils.squad_protocol import assert_test_not_in_selection_path


class _FakeTok:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        suffix = "\nAssistant:" if add_generation_prompt else ""
        return f"<|im_start|>user\n{content}<|im_end|>{suffix}"

    def __call__(self, text, add_special_tokens=False):
        # Deterministic toy tokenization by char codes mod vocab.
        ids = [max(2, ord(ch) % 120) for ch in str(text)[:40]]
        return {"input_ids": ids or [2]}


class _FakeSplit(list):
    def select(self, indices):
        return _FakeSplit(self[i] for i in indices)


def _fake_examples(n: int = 8):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"ex{i}",
                "context": f"Paris is the capital of France. Fact {i}.",
                "question": "What is the capital of France?",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            }
        )
    return _FakeSplit(rows)


def _tiny_qwen():
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_assert_blocks_test_split_in_loaders():
    with pytest.raises(ValueError, match="frozen test"):
        assert_test_not_in_selection_path(["train", "test"])


def test_squad_causal_dataset_masks_prompt():
    tok = _FakeTok()
    ds = SquadCausalLmDataset(_fake_examples(3), tok, max_seq_len=64, max_samples=3, split_name="train")
    assert len(ds) == 3
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])
    assert any(label == -100 for label in item["labels"])
    assert any(label != -100 for label in item["labels"])


def test_lm_recovery_one_step_and_mlp_prune():
    model = _tiny_qwen()
    tok = _FakeTok()
    splits = {"train": _fake_examples(6), "validation": _fake_examples(4)}
    train_loader, val_loader = build_squad_lm_loaders(
        splits, tok, batch_size=2, max_seq_len=64, train_max_samples=6, validation_max_samples=4
    )
    before = sum(p.numel() for p in model.parameters())
    backend = resolve_pruning_backend(model, "qwen")
    names = mlp_only_layer_names(model)
    assert names
    pruned = backend.create_pruned_model({names[0]: 0.5})
    after = sum(p.numel() for p in pruned.parameters())
    assert after < before

    recovered, history = quick_lm_recovery(
        pruned,
        train_loader,
        val_loader,
        epochs=1,
        learning_rate=1e-3,
        device="cpu",
        precision="fp32",
        verbose=False,
    )
    assert history["best_validation_loss"] is not None
    metrics = evaluate_lm_loss(recovered, val_loader, device="cpu")
    assert "accuracy" in metrics and metrics["loss"] >= 0.0

    importance = magnitude_importance_mlp(pruned)
    assert names[0] in importance or any(k.endswith(".mlp.intermediate") for k in importance)


def test_lora_recovery_one_step_merges():
    peft = pytest.importorskip("peft")
    del peft
    model = _tiny_qwen()
    tok = _FakeTok()
    splits = {"train": _fake_examples(6), "validation": _fake_examples(4)}
    train_loader, val_loader = build_squad_lm_loaders(
        splits, tok, batch_size=2, max_seq_len=64, train_max_samples=6, validation_max_samples=4
    )
    backend = resolve_pruning_backend(model, "qwen")
    names = mlp_only_layer_names(model)
    pruned = backend.create_pruned_model({names[0]: 0.5})
    recovered, history = quick_lora_recovery(
        pruned,
        train_loader,
        val_loader,
        epochs=1,
        learning_rate=1e-3,
        device="cpu",
        verbose=False,
        lora_r=4,
        lora_alpha=8,
    )
    assert history["recovery_kind"] == "qwen_lora_ce"
    assert history["merged"] is True
    assert history["best_validation_loss"] is not None
    assert not hasattr(recovered, "peft_config")
    metrics = evaluate_lm_loss(recovered, val_loader, device="cpu")
    assert metrics["loss"] >= 0.0


def test_run_configured_recovery_dispatches_lora():
    pytest.importorskip("peft")
    model = _tiny_qwen()
    tok = _FakeTok()
    splits = {"train": _fake_examples(4), "validation": _fake_examples(2)}
    train_loader, val_loader = build_squad_lm_loaders(
        splits, tok, batch_size=2, max_seq_len=64, train_max_samples=4, validation_max_samples=2
    )
    config = {
        "hardware": {"device": "cpu", "precision": "fp32"},
        "recovery": {
            "backend": "lora",
            "epochs": 1,
            "learning_rate": 1e-3,
            "lora_r": 4,
            "lora_alpha": 8,
        },
    }
    recovered, history = run_configured_recovery(
        model, train_loader, val_loader, config, copy_model=True, verbose=False
    )
    assert history["recovery_kind"] == "qwen_lora_ce"
    assert history["merged"] is True
    assert sum(p.numel() for p in recovered.parameters()) > 0


def test_build_loaders_rejects_test_key():
    tok = _FakeTok()
    with pytest.raises(ValueError, match="frozen test"):
        build_squad_lm_loaders(
            {"train": _fake_examples(2), "validation": _fake_examples(2), "test": _fake_examples(1)},
            tok,
            batch_size=1,
            max_seq_len=32,
        )
