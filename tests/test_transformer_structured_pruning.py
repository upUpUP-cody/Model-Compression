"""Smoke tests for Qwen2-style physical head/FFN pruning (tiny random model)."""
from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.pruning.pruning_backend import TransformerBackend, resolve_pruning_backend
from src.pruning.transformer_structured_pruning import TransformerStructuredPruning


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


def test_transformer_backend_prunes_mlp_and_forward():
    model = _tiny_qwen()
    before = sum(p.numel() for p in model.parameters())
    backend = resolve_pruning_backend(model, "qwen")
    assert isinstance(backend, TransformerBackend)
    names = backend.prunable_layer_names()
    assert "layers.0.mlp.intermediate" in names
    assert "layers.0.self_attn.heads" in names

    pruned = backend.create_pruned_model({"layers.0.mlp.intermediate": 0.5})
    after = sum(p.numel() for p in pruned.parameters())
    assert after < before

    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out = pruned(input_ids=input_ids)
    assert out.logits.shape[0] == 1


def test_transformer_head_group_prune_forward():
    model = _tiny_qwen()
    pruner = TransformerStructuredPruning(model)
    before = pruner.get_structural_parameter_count()
    # Keep one full KV-group (2 query heads when n_q=4, n_kv=2).
    pruner.prune_heads("layers.0.self_attn.heads", [0, 1])
    after = pruner.get_structural_parameter_count()
    assert after < before
    assert model.model.layers[0].self_attn.num_key_value_groups == 2

    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.logits.shape[-1] == 128
