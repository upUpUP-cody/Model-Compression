#!/usr/bin/env python3
"""Phase K smoke: dense (+ optional oneshot) Qwen SQuAD eval -> /mnt/data/results."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pruning.pruning_backend import resolve_pruning_backend
from src.utils.qwen_squad_eval import (
    evaluate_squad_split,
    load_qwen_for_eval,
    prepare_squad_from_config,
    write_json,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_device(config: Dict[str, Any]) -> str:
    requested = str(config.get("hardware", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable; falling back to CPU")
        return "cpu"
    return requested


def _count_params(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _oneshot_prune(model, config: Dict[str, Any]):
    backend = resolve_pruning_backend(model, "qwen")
    mlp_ratio = float(config.get("pruning", {}).get("oneshot_mlp_ratio", 0.0))
    head_ratio = float(config.get("pruning", {}).get("oneshot_head_ratio", 0.0))
    pruning_config = {}
    for name in backend.prunable_layer_names():
        if name.endswith(".mlp.intermediate") and mlp_ratio > 0.0:
            pruning_config[name] = mlp_ratio
        if name.endswith(".self_attn.heads") and head_ratio > 0.0:
            pruning_config[name] = head_ratio
    if not pruning_config:
        return model, {"applied": False, "param_before": _count_params(model)}
    before = _count_params(model)
    pruned = backend.create_pruned_model(pruning_config)
    after = _count_params(pruned)
    return pruned, {
        "applied": True,
        "param_before": before,
        "param_after": after,
        "compression_ratio": float(before) / float(after) if after else None,
        "pruning_config_keys": sorted(pruning_config.keys())[:8],
        "n_pruned_layers": len(pruning_config),
        "oneshot_mlp_ratio": mlp_ratio,
        "oneshot_head_ratio": head_ratio,
        "recovery": "none_documented",
    }


def run_mode(mode: str, config: Dict[str, Any], device: str) -> Dict[str, Any]:
    model_path = str(config["model"]["path"])
    splits = prepare_squad_from_config(config)
    print(f"[INFO] split metadata: {json.dumps(splits.metadata())}")

    model, tokenizer = load_qwen_for_eval(
        model_path,
        device=device,
        torch_dtype=str(config["model"].get("torch_dtype", "float16")),
    )
    prune_info = {"applied": False, "param_before": _count_params(model)}
    if mode == "oneshot":
        model, prune_info = _oneshot_prune(model, config)
        model.to(device)
        model.eval()

    t0 = time.time()
    metrics = evaluate_squad_split(
        model,
        tokenizer,
        splits.validation,
        device=device,
        batch_size=int(config["hardware"].get("batch_size", 1)),
        max_seq_len=int(config["dataset"].get("max_seq_len", 512)),
        max_new_tokens=int(config["model"].get("max_new_tokens", 64)),
        max_samples=config["dataset"].get("eval_max_samples"),
        split_name="validation",
    )
    elapsed = time.time() - t0
    payload = {
        "mode": mode,
        "run_label": config.get("run_label"),
        "model_path": model_path,
        "device": device,
        "metrics": metrics,
        "split_metadata": splits.metadata(),
        "pruning": prune_info,
        "elapsed_sec": elapsed,
        "note": "selection uses carved validation only; official validation is frozen test",
        "claim": "smoke only; do not claim search superiority on LLM",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen SQuAD smoke eval")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "qwen_squad_smoke.yaml"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "oneshot", "both"],
        default="both",
    )
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    device = _resolve_device(config)
    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    modes = ["dense", "oneshot"] if args.mode == "both" else [args.mode]
    summary = {"modes": {}}
    for mode in modes:
        print(f"[INFO] running mode={mode}")
        payload = run_mode(mode, config, device)
        out_path = output_root / f"{mode}_metrics.json"
        write_json(out_path, payload)
        print(
            f"[OK] {mode}: F1={payload['metrics']['f1']:.2f} "
            f"EM={payload['metrics']['exact_match']:.2f} -> {out_path}"
        )
        summary["modes"][mode] = {
            "f1": payload["metrics"]["f1"],
            "exact_match": payload["metrics"]["exact_match"],
            "path": str(out_path),
            "pruning": payload.get("pruning"),
        }

    summary_path = output_root / "smoke_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
