#!/usr/bin/env python3
"""Phase KG smoke: dense / oneshot / iterative / search on Qwen+GLUE SST-2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.qwen_glue_comparison import METHOD_NAMES, count_params, results_to_records, run_method
from src.utils.glue_protocol import assert_test_not_in_selection_path
from src.utils.qwen_glue_eval import load_qwen_for_eval, prepare_glue_from_config, write_json
from src.utils.qwen_glue_train_data import build_glue_lm_loaders


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_device(config: Dict[str, Any]) -> str:
    requested = str(config.get("hardware", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable; falling back to CPU")
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen GLUE SST-2 smoke comparison")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "qwen_glue_smoke.yaml"),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated subset of methods; default = config.comparison.methods",
    )
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    config["hardware"]["device"] = _resolve_device(config)
    device = config["hardware"]["device"]

    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    splits = prepare_glue_from_config(config)
    selection = splits.selection_splits()
    assert_test_not_in_selection_path(list(selection.keys()))
    print(f"[INFO] split metadata: {json.dumps(splits.metadata())}")

    model, tokenizer = load_qwen_for_eval(
        str(config["model"]["path"]),
        device=device,
        torch_dtype=str(config["model"].get("torch_dtype", "float16")),
    )
    train_loader, validation_loader = build_glue_lm_loaders(
        selection,
        tokenizer,
        task=str(config["dataset"].get("task", "sst2")),
        batch_size=int(config["hardware"].get("batch_size", 1)),
        max_seq_len=int(config["dataset"].get("max_seq_len", 128)),
        train_max_samples=config["dataset"].get("train_max_samples"),
        validation_max_samples=config["dataset"].get("validation_max_samples"),
    )

    if args.methods.strip():
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = list(config.get("comparison", {}).get("methods") or METHOD_NAMES)
    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(f"unsupported method: {method}")

    baseline_params = count_params(model)
    results = []
    for method in methods:
        print(f"[INFO] running method={method}")
        if method != "dense":
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, tokenizer = load_qwen_for_eval(
                str(config["model"]["path"]),
                device=device,
                torch_dtype=str(config["model"].get("torch_dtype", "float16")),
            )
        result, model = run_method(
            method,
            model,
            tokenizer,
            train_loader,
            validation_loader,
            splits.validation,
            config,
            baseline_parameter_count=baseline_params,
        )
        out_path = output_root / f"{method}_metrics.json"
        payload = {
            "run_label": config.get("run_label"),
            "claim": "KG GLUE SST-2 smoke only; do not claim search superiority on LLM",
            "eval_strategy": "prompt_verbalizer",
            "split_metadata": splits.metadata(),
            "result": results_to_records([result])[0],
        }
        write_json(out_path, payload)
        print(
            f"[OK] {method}: compression={result.compression_ratio:.3f}x "
            f"proxy={result.val_proxy_accuracy:.2f} "
            f"acc={result.accuracy} -> {out_path}"
        )
        results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "run_label": config.get("run_label"),
        "output_root": str(output_root),
        "task": config["dataset"].get("task", "sst2"),
        "eval_strategy": "prompt_verbalizer",
        "methods": results_to_records(results),
        "note": "selection uses carved validation only; official validation is frozen test",
    }
    summary_path = output_root / "glue_smoke_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
