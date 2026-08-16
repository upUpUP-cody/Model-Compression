#!/usr/bin/env python3
"""Phase KG.5: SST-2 + RTE + QNLI x 1.5x/2.0x four-method small scan."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.qwen_glue_comparison import METHOD_NAMES, count_params, results_to_records, run_method
from src.utils.glue_protocol import SUPPORTED_GLUE_TASKS, assert_test_not_in_selection_path
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


def _stage_targets_for(target: float) -> List[float]:
    target = float(target)
    if abs(target - 1.5) < 1e-6:
        return [1.25, 1.5]
    if abs(target - 2.0) < 1e-6:
        return [1.5, 2.0]
    # Generic two-stage ramp for other targets.
    mid = 1.0 + 0.5 * (target - 1.0)
    return [round(mid, 4), target]


def _cell_dir_name(task: str, target: float) -> str:
    # e.g. sst2_1.5x
    text = f"{float(target):g}"
    return f"{task}_{text}x"


def _apply_cell(config: Dict[str, Any], task: str, target: float) -> Dict[str, Any]:
    cell = copy.deepcopy(config)
    cell["dataset"]["task"] = str(task)
    comparison = cell.setdefault("comparison", {})
    comparison["target_compression_ratio"] = float(target)
    comparison["iterative_stage_targets"] = _stage_targets_for(target)
    search = cell.setdefault("search", {})
    search["max_step_compression"] = float(target)
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen GLUE KG.5 three-task small scan")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "qwen_glue_kg5.yaml"),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated subset of methods; default = config.comparison.methods",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated task subset; default = config.tasks",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated compression targets; default = config.comparison.compression_targets",
    )
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    config["hardware"]["device"] = _resolve_device(config)
    device = config["hardware"]["device"]

    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if args.tasks.strip():
        tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    else:
        tasks = list(config.get("tasks") or ["sst2", "rte", "qnli"])
    for task in tasks:
        if task not in SUPPORTED_GLUE_TASKS:
            raise ValueError(f"unsupported task: {task}")

    if args.targets.strip():
        targets = [float(item.strip()) for item in args.targets.split(",") if item.strip()]
    else:
        targets = [float(x) for x in (config.get("comparison", {}).get("compression_targets") or [1.5, 2.0])]

    if args.methods.strip():
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = list(config.get("comparison", {}).get("methods") or METHOD_NAMES)
    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(f"unsupported method: {method}")

    summary_rows: List[Dict[str, Any]] = []

    for task in tasks:
        for target in targets:
            cell_config = _apply_cell(config, task, target)
            cell_dir = output_root / _cell_dir_name(task, target)
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] cell task={task} target={target}x -> {cell_dir}")

            splits = prepare_glue_from_config(cell_config)
            selection = splits.selection_splits()
            assert_test_not_in_selection_path(list(selection.keys()))
            print(f"[INFO] split metadata: {json.dumps(splits.metadata())}")

            model, tokenizer = load_qwen_for_eval(
                str(cell_config["model"]["path"]),
                device=device,
                torch_dtype=str(cell_config["model"].get("torch_dtype", "float16")),
            )
            train_loader, validation_loader = build_glue_lm_loaders(
                selection,
                tokenizer,
                task=task,
                batch_size=int(cell_config["hardware"].get("batch_size", 1)),
                max_seq_len=int(cell_config["dataset"].get("max_seq_len", 256)),
                train_max_samples=cell_config["dataset"].get("train_max_samples"),
                validation_max_samples=cell_config["dataset"].get("validation_max_samples"),
            )
            baseline_params = count_params(model)

            for method in methods:
                print(f"[INFO] running task={task} target={target}x method={method}")
                if method != "dense":
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model, tokenizer = load_qwen_for_eval(
                        str(cell_config["model"]["path"]),
                        device=device,
                        torch_dtype=str(cell_config["model"].get("torch_dtype", "float16")),
                    )
                result, model = run_method(
                    method,
                    model,
                    tokenizer,
                    train_loader,
                    validation_loader,
                    splits.validation,
                    cell_config,
                    baseline_parameter_count=baseline_params,
                )
                out_path = cell_dir / f"{method}_metrics.json"
                payload = {
                    "run_label": cell_config.get("run_label"),
                    "claim": "KG.5 GLUE three-task small scan; informal gate only; not paper main table",
                    "eval_strategy": "prompt_verbalizer",
                    "task": task,
                    "target_compression_ratio": float(target),
                    "split_metadata": splits.metadata(),
                    "result": results_to_records([result])[0],
                }
                write_json(out_path, payload)
                print(
                    f"[OK] {task} {target}x {method}: compression={result.compression_ratio:.3f}x "
                    f"ce={result.val_loss:.4f} acc={result.accuracy} -> {out_path}"
                )
                summary_rows.append(
                    {
                        "task": task,
                        "target_compression_ratio": float(target),
                        "method": method,
                        "compression_ratio": result.compression_ratio,
                        "accuracy": result.accuracy,
                        "val_loss": result.val_loss,
                        "elapsed_sec": result.elapsed_sec,
                        "metrics_path": str(out_path),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Free model between cells.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "run_label": config.get("run_label"),
        "output_root": str(output_root),
        "tasks": tasks,
        "compression_targets": targets,
        "methods": methods,
        "eval_strategy": "prompt_verbalizer",
        "rows": summary_rows,
        "note": (
            "selection uses carved validation only; official validation is frozen test; "
            "informal KG.5 gate — do not claim search superiority"
        ),
    }
    summary_path = output_root / "kg5_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
