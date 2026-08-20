"""Shared helpers for Stage A E0–E3 smoke (PDF order; proxy_1.5B allowed)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml

from src.experiments.qwen_k5_comparison import count_params, magnitude_importance_mlp, mlp_uniform_ratios
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.qwen_lm_recovery import evaluate_lm_loss
from src.utils.device import resolve_device
from src.utils.glue_protocol import assert_test_not_in_selection_path
from src.utils.qwen_glue_eval import load_qwen_for_eval, prepare_glue_from_config
from src.utils.qwen_glue_train_data import build_glue_lm_loaders
from src.utils.qwen_squad_eval import write_json


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mirror_docs_report(experiment_id: str, slug: str, report_text: str) -> Path:
    docs = Path("docs/results")
    ensure_dir(docs)
    out = docs / f"{experiment_id}_{slug}.md"
    write_report(out, report_text)
    return out


def gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def prepare_sst2_loaders(config: Mapping[str, Any], tokenizer, device: str):
    """Build tiny SST-2 LM loaders for Stage A smoke metrics."""
    cfg = dict(config)
    ds = dict(cfg.get("dataset") or {})
    ds.setdefault("name", "nyu-mll/glue")
    ds.setdefault("task", "sst2")
    ds.setdefault("cache_dir", "/mnt/data/datasets/glue")
    ds.setdefault("validation_fraction", 0.1)
    ds.setdefault("split_seed", int(cfg.get("seed", 42)))
    ds.setdefault("max_seq_len", int(cfg.get("evaluation", {}).get("max_seq_len", 256)))
    ds.setdefault("train_max_samples", int(cfg.get("evaluation", {}).get("train_max_samples", 128)))
    ds.setdefault("validation_max_samples", int(cfg.get("evaluation", {}).get("validation_max_samples", 64)))
    cfg["dataset"] = ds
    splits = prepare_glue_from_config(cfg)
    selection = splits.selection_splits()
    assert_test_not_in_selection_path(list(selection.keys()))
    train_loader, val_loader = build_glue_lm_loaders(
        selection,
        tokenizer,
        task="sst2",
        batch_size=int(cfg.get("hardware", {}).get("batch_size", 1)),
        max_seq_len=int(ds["max_seq_len"]),
        train_max_samples=ds.get("train_max_samples"),
        validation_max_samples=ds.get("validation_max_samples"),
    )
    return train_loader, val_loader, splits.metadata()


def load_model(config: Mapping[str, Any]):
    device = resolve_device(str(config.get("hardware", {}).get("device", "cuda")))
    model, tokenizer = load_qwen_for_eval(
        str(config["model"]["path"]),
        device=device,
        torch_dtype=str(config["model"].get("torch_dtype", "float16")),
    )
    return model, tokenizer, device


def prune_mlp_ratio(model: nn.Module, prune_ratio: float) -> nn.Module:
    """One-shot uniform MLP intermediate prune (magnitude path; Wanda TODO)."""
    ratios = mlp_uniform_ratios(model, float(prune_ratio))
    cpu = model.cpu()
    pruned = resolve_pruning_backend(cpu, "qwen").create_pruned_model(ratios)
    return pruned


def prune_mlp_by_importance(
    model: nn.Module,
    prune_ratio: float,
    *,
    max_layers: int = 4,
) -> nn.Module:
    importance = magnitude_importance_mlp(model, max_layers=max_layers)
    keep_indices: Dict[str, List[int]] = {}
    for name, scores in importance.items():
        n = int(scores.numel())
        keep = max(1, int(round(n * (1.0 - float(prune_ratio)))))
        ranked = sorted(range(n), key=lambda i: (-float(scores[i]), i))
        keep_indices[name] = sorted(ranked[:keep])
    # Fill remaining MLP layers with full keep (backend may require all or subset)
    cpu = model.cpu()
    backend = resolve_pruning_backend(cpu, "qwen")
    return backend.create_pruned_model_by_indices(keep_indices)


def eval_proxy(model: nn.Module, loader, device: str) -> Dict[str, float]:
    out = evaluate_lm_loss(model, loader, device=device)
    loss = float(out.get("loss", float("nan")))
    ppl = float(torch.exp(torch.tensor(loss)).item()) if loss == loss else float("nan")
    return {
        "loss": loss,
        "ppl": ppl,
        "proxy_accuracy": float(out.get("accuracy", float("nan"))),
    }
