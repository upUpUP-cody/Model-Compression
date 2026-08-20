#!/usr/bin/env python3
"""Phase K6: SQuAD 1.5x/2x/4x four-method small matrix (dense-relative budgets)."""
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

from src.experiments.qwen_k5_comparison import METHOD_NAMES, count_params, results_to_records, run_method
from src.utils.compression_clarity import (
    SUMMARY_NOTE_DENSE_BASELINE,
    compression_clarity_fields,
    format_method_ok_line,
)
from src.utils.qwen_squad_eval import (
    evaluate_squad_split,
    load_qwen_for_eval,
    prepare_squad_from_config,
    write_json,
)
from src.utils.qwen_train_data import build_squad_lm_loaders
from src.utils.squad_protocol import assert_test_not_in_selection_path


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
    if abs(target - 4.0) < 1e-6:
        return [1.5, 2.0, 4.0]
    mid = 1.0 + 0.5 * (target - 1.0)
    return [round(mid, 4), target]


def _cell_dir_name(target: float) -> str:
    return f"target_{float(target):g}x"


def _apply_cell(config: Dict[str, Any], target: float) -> Dict[str, Any]:
    cell = copy.deepcopy(config)
    comparison = cell.setdefault("comparison", {})
    comparison["target_compression_ratio"] = float(target)
    comparison["iterative_stage_targets"] = _stage_targets_for(target)
    search = cell.setdefault("search", {})
    search["max_step_compression"] = float(target)
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen K6 SQuAD small matrix")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "qwen_k6.yaml"))
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--targets", type=str, default="")
    parser.add_argument(
        "--eval-only-dense",
        action="store_true",
        help="Smoke: run dense generative eval only (chat/SDPA check)",
    )
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    config["hardware"]["device"] = _resolve_device(config)
    device = config["hardware"]["device"]

    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if args.targets.strip():
        targets = [float(item.strip()) for item in args.targets.split(",") if item.strip()]
    else:
        targets = [float(x) for x in (config.get("comparison", {}).get("compression_targets") or [1.5, 2.0, 4.0])]

    if args.eval_only_dense:
        methods = ["dense"]
        targets = targets[:1]
    elif args.methods.strip():
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = list(config.get("comparison", {}).get("methods") or METHOD_NAMES)
    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(f"unsupported method: {method}")

    splits = prepare_squad_from_config(config)
    selection = splits.selection_splits()
    assert_test_not_in_selection_path(list(selection.keys()))
    print(f"[INFO] split metadata: {json.dumps(splits.metadata())}")

    summary_rows: List[Dict[str, Any]] = []

    for target in targets:
        cell_config = _apply_cell(config, target)
        cell_dir = output_root / _cell_dir_name(target)
        cell_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] cell target={target}x -> {cell_dir}")

        model, tokenizer = load_qwen_for_eval(
            str(cell_config["model"]["path"]),
            device=device,
            torch_dtype=str(cell_config["model"].get("torch_dtype", "float16")),
        )
        train_loader, validation_loader = build_squad_lm_loaders(
            selection,
            tokenizer,
            batch_size=int(cell_config["hardware"].get("batch_size", 1)),
            max_seq_len=int(cell_config["dataset"].get("max_seq_len", 384)),
            train_max_samples=cell_config["dataset"].get("train_max_samples"),
            validation_max_samples=cell_config["dataset"].get("validation_max_samples"),
        )
        baseline_params = count_params(model)

        for method in methods:
            print(f"[INFO] running target={target}x method={method}")
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
            frozen_metrics = None
            if bool(cell_config.get("comparison", {}).get("run_frozen_test", False)):
                frozen_cap = cell_config["dataset"].get(
                    "frozen_test_max_samples",
                    cell_config["dataset"].get("eval_max_samples"),
                )
                frozen_metrics = evaluate_squad_split(
                    model,
                    tokenizer,
                    splits.test,
                    device=device,
                    batch_size=int(cell_config["hardware"].get("batch_size", 1)),
                    max_seq_len=int(cell_config["dataset"].get("max_seq_len", 384)),
                    max_new_tokens=int(cell_config["model"].get("max_new_tokens", 32)),
                    max_samples=frozen_cap,
                    split_name="frozen_report",
                    allow_frozen_test=True,
                )
                frozen_path = cell_dir / f"{method}_frozen_test.json"
                clarity = compression_clarity_fields(
                    method, float(target), float(result.compression_ratio)
                )
                write_json(
                    frozen_path,
                    {
                        "run_label": cell_config.get("run_label"),
                        "task": "squad_v2",
                        "target_compression_ratio": float(target),
                        **clarity,
                        "method": method,
                        "split": "official_validation_frozen_once",
                        "metrics": frozen_metrics,
                    },
                )
                print(
                    f"[OK] frozen test {target}x {method}: "
                    f"f1={frozen_metrics.get('f1')} em={frozen_metrics.get('exact_match')} "
                    f"n={frozen_metrics.get('n_examples')} -> {frozen_path}"
                )
            out_path = cell_dir / f"{method}_metrics.json"
            clarity = compression_clarity_fields(
                method, float(target), float(result.compression_ratio)
            )
            payload = {
                "run_label": cell_config.get("run_label"),
                "claim": str(
                    cell_config.get(
                        "claim",
                        "K6 SQuAD small matrix; informal until frozen-test once; "
                        "do not claim search superiority on LLM",
                    )
                ),
                "task": "squad_v2",
                "target_compression_ratio": float(target),
                **clarity,
                "split_metadata": splits.metadata(),
                "result": results_to_records([result])[0],
                "frozen_test": frozen_metrics,
            }
            write_json(out_path, payload)
            print(
                format_method_ok_line(
                    method=method,
                    cell_target=float(target),
                    actual_compression=float(result.compression_ratio),
                    metric_parts=(
                        f"ce={result.val_loss:.4f} f1={result.f1} "
                        f"em={result.exact_match} -> {out_path}"
                    ),
                )
            )
            summary_rows.append(
                {
                    "target_compression_ratio": float(target),
                    **clarity,
                    "method": method,
                    "compression_ratio": result.compression_ratio,
                    "f1": result.f1,
                    "exact_match": result.exact_match,
                    "val_loss": result.val_loss,
                    "elapsed_sec": result.elapsed_sec,
                    "metrics_path": str(out_path),
                    "frozen_test_f1": None if frozen_metrics is None else frozen_metrics.get("f1"),
                    "frozen_test_em": None
                    if frozen_metrics is None
                    else frozen_metrics.get("exact_match"),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "run_label": config.get("run_label"),
        "output_root": str(output_root),
        "compression_targets": targets,
        "methods": methods,
        "rows": summary_rows,
        "note": (
            "selection uses carved validation only; official validation is frozen test; "
            "iterative budgets are cumulative vs dense; search is single-candidate; "
            + SUMMARY_NOTE_DENSE_BASELINE
        ),
        "run_frozen_test": bool(config.get("comparison", {}).get("run_frozen_test", False)),
    }
    summary_name = str(config.get("logging", {}).get("summary_filename") or "k6_summary.json")
    summary_path = output_root / summary_name
    write_json(summary_path, summary)
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
