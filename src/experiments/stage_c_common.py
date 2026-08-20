"""Shared helpers for Stage C E8/E9 (resumable cell runs)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.experiments.stage_a_common import (
    ensure_dir,
    eval_proxy,
    load_model,
    load_yaml,
    mirror_docs_report,
    prepare_sst2_loaders,
    prune_mlp_ratio,
    write_report,
)
from src.recovery.qwen_lora_recovery import quick_lora_recovery
from src.utils.qwen_squad_eval import write_json

STRATEGIES = (
    "random",
    "teacher_hard",
    "student_hard",
    "low_gap",
    "high_gap",
    "high_gap_random",
)


def cell_path(out_root: Path, cell_id: str) -> Path:
    return ensure_dir(out_root / "cells") / f"{cell_id}.json"


def cell_done(out_root: Path, cell_id: str) -> bool:
    path = cell_path(out_root, cell_id)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("status") == "done")


def save_cell(out_root: Path, cell_id: str, payload: Mapping[str, Any]) -> Path:
    path = cell_path(out_root, cell_id)
    body = dict(payload)
    body["cell_id"] = cell_id
    body["status"] = "done"
    body["saved_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, body)
    return path


def load_all_cells(out_root: Path) -> List[Dict[str, Any]]:
    cells_dir = out_root / "cells"
    if not cells_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(cells_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def write_progress(out_root: Path, experiment_id: str, lines: Sequence[str]) -> Path:
    path = out_root / "RUN_PROGRESS.md"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = [
        f"# {experiment_id} run progress",
        "",
        f"Updated: {stamp}",
        "",
        "Resume: re-run the same command; completed `cells/*.json` are skipped.",
        "",
        *lines,
        "",
    ]
    write_report(path, "\n".join(body))
    # Human mirror in repo
    mirror = Path("docs/process") / f"{experiment_id}_RUN_PROGRESS.md"
    write_report(mirror, "\n".join(body))
    return path


@torch.no_grad()
def per_example_nll(model: nn.Module, batch: Mapping[str, Any], device: str) -> List[float]:
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch.get("labels", input_ids).to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = loss_flat.view(shift_labels.size(0), -1)
    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    nll = (token_loss * mask).sum(dim=1) / denom
    return [float(x) for x in nll.detach().cpu()]


def score_pool(
    parent: nn.Module,
    child: nn.Module,
    loader: DataLoader,
    device: str,
) -> List[Dict[str, float]]:
    """Score each example in loader (expects batch_size=1 for clean indexing)."""
    rows: List[Dict[str, float]] = []
    idx = 0
    for batch in loader:
        t_nll = per_example_nll(parent, batch, device)
        s_nll = per_example_nll(child, batch, device)
        labels = batch.get("labels", batch["input_ids"])
        lengths = [
            int((labels[i] != -100).sum().item()) if (labels[i] == -100).any() else int(labels[i].numel())
            for i in range(labels.size(0))
        ]
        for j, (tn, sn) in enumerate(zip(t_nll, s_nll)):
            rows.append(
                {
                    "index": float(idx),
                    "teacher_nll": float(tn),
                    "student_nll": float(sn),
                    "gap": float(sn - tn),
                    "length": float(lengths[j]),
                }
            )
            idx += 1
    return rows


def select_indices(
    scored: Sequence[Mapping[str, float]],
    strategy: str,
    n: int,
    seed: int,
) -> List[int]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy={strategy}; expected one of {STRATEGIES}")
    n = min(int(n), len(scored))
    if n <= 0:
        return []
    indices = [int(row["index"]) for row in scored]
    g = torch.Generator()
    g.manual_seed(int(seed))

    if strategy == "random":
        perm = torch.randperm(len(indices), generator=g).tolist()
        return [indices[i] for i in perm[:n]]

    if strategy == "teacher_hard":
        ranked = sorted(scored, key=lambda r: (-float(r["teacher_nll"]), int(r["index"])))
        return [int(r["index"]) for r in ranked[:n]]

    if strategy == "student_hard":
        ranked = sorted(scored, key=lambda r: (-float(r["student_nll"]), int(r["index"])))
        return [int(r["index"]) for r in ranked[:n]]

    if strategy == "low_gap":
        ranked = sorted(scored, key=lambda r: (float(r["gap"]), int(r["index"])))
        return [int(r["index"]) for r in ranked[:n]]

    if strategy == "high_gap":
        ranked = sorted(scored, key=lambda r: (-float(r["gap"]), int(r["index"])))
        return [int(r["index"]) for r in ranked[:n]]

    # high_gap_random: 80/20
    n_high = int(round(n * 0.8))
    n_rand = n - n_high
    high = select_indices(scored, "high_gap", n_high, seed)
    remaining = [i for i in indices if i not in set(high)]
    if not remaining or n_rand <= 0:
        return high[:n]
    perm = torch.randperm(len(remaining), generator=g).tolist()
    extra = [remaining[i] for i in perm[:n_rand]]
    return (high + extra)[:n]


def subset_loader(loader: DataLoader, indices: Sequence[int], batch_size: int = 1) -> DataLoader:
    dataset = loader.dataset
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=getattr(loader, "collate_fn", None),
        num_workers=0,
    )


def recovery_gain(compressed: Mapping[str, float], recovered: Mapping[str, float]) -> float:
    """Perf = -loss (lower CE is better). Gain = Perf_rec - Perf_comp."""
    return float(-recovered["loss"] - (-compressed["loss"]))


def run_lora_on_subset(
    child: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    recovery_cfg: Mapping[str, Any],
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    return quick_lora_recovery(
        child,
        train_loader,
        val_loader,
        epochs=int(recovery_cfg.get("epochs", 1)),
        learning_rate=float(recovery_cfg.get("learning_rate", 1e-4)),
        device=device,
        verbose=True,
        weight_decay=float(recovery_cfg.get("weight_decay", 0.0)),
        copy_model=True,
        lora_r=int(recovery_cfg.get("lora_r", 8)),
        lora_alpha=int(recovery_cfg.get("lora_alpha", 16)),
        lora_dropout=float(recovery_cfg.get("lora_dropout", 0.05)),
        max_steps=int(recovery_cfg["max_steps"]) if recovery_cfg.get("max_steps") is not None else None,
    )


def build_pool_loaders(config: Mapping[str, Any], tokenizer, device: str):
    """Larger train pool for mining + small val for eval."""
    cfg = dict(config)
    ev = dict(cfg.get("evaluation") or {})
    pool_n = int(cfg.get("pool_max_samples", ev.get("train_max_samples", 1024)))
    val_n = int(ev.get("validation_max_samples", 64))
    ev["train_max_samples"] = pool_n
    ev["validation_max_samples"] = val_n
    cfg["evaluation"] = ev
    return prepare_sst2_loaders(cfg, tokenizer, device)


__all__ = [
    "STRATEGIES",
    "build_pool_loaders",
    "cell_done",
    "cell_path",
    "eval_proxy",
    "ensure_dir",
    "load_all_cells",
    "load_model",
    "load_yaml",
    "mirror_docs_report",
    "prune_mlp_ratio",
    "recovery_gain",
    "run_lora_on_subset",
    "save_cell",
    "score_pool",
    "select_indices",
    "subset_loader",
    "write_progress",
    "write_report",
    "write_json",
]
