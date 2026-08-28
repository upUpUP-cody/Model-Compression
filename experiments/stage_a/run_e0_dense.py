#!/usr/bin/env python3
"""E0 Dense Model Baseline (PDF Stage A)."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector
from src.experiments.stage_a_common import (
    ensure_dir,
    gpu_mem_gb,
    load_yaml,
    mirror_docs_report,
    write_report,
)
from src.experiments.qwen_k5_comparison import count_params
from src.utils.qwen_squad_eval import load_qwen_for_eval, write_json


def _fmt_cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_report(config, cap_out, resources, elapsed, smoke: bool) -> str:
    vector = cap_out["vector"]
    mode = cap_out.get("mode", "scan")

    detail = cap_out.get("details", {})
    limit_table_lines = ["| 维 | 任务 | limit | 指标 |", "|----|------|-------|------|"]
    for dim in DIM_ORDER:
        d = detail.get(dim, {})
        limit_table_lines.append(
            f"| {dim} | {d.get('task', 'n/a')} | {d.get('limit', 'n/a')} | {d.get('metric_key', 'n/a')} |"
        )
    limit_table = "\n".join(limit_table_lines)

    vec_row = " | ".join(_fmt_cell(vector[d]) for d in DIM_ORDER)
    return f"""# E0. Dense Model Baseline

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E0 |
| 目的 | 建立所有后续实验的 dense baseline |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | None |
| Evaluation | 标准小样本六维（lm_eval, mode={mode}, seed={cap_out.get("seed", 42)}） |
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| status | {"smoke" if smoke else "done"} |

## 六维协议（冻结）

{limit_table}

## 输出

Dense performance vector P(M0) 与资源行（PDF §E0）。

## 记录表

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| {vec_row} |

| GPU memory (GB peak) | latency_sec_eval | parameter_count | model path |
|----------------------|------------------|-----------------|------------|
| {resources["gpu_memory_gb_peak"] if resources["gpu_memory_gb_peak"] is not None else "n/a"} | {elapsed:.1f} | {resources["parameter_count"]} | `{resources["model_size_path"]}` |

## 结论（对照成功条件）

- **{"冒烟" if smoke else "满足"}**：六维 lm_eval pipeline {"通路验证" if smoke else "已落盘"}；小样本协议主看后续相对 Delta 与 capability-specific cliff。
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E。
- 旧 SST-2 部分向量产物作废；以本报告为准。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e0_dense.yaml"))
    parser.add_argument("--smoke", action="store_true", help="limit_override=2 per dim; write to *_smoke/")
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out_root = Path(config["logging"]["output_root"])
    if args.smoke:
        out_root = Path(str(out_root) + "_smoke")
    out = ensure_dir(out_root)

    print(f"[INFO] E0 dense baseline starting (smoke={args.smoke})")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_config = copy.deepcopy(config)
    if args.smoke:
        run_config.setdefault("evaluation", {}).setdefault("capability", {})["limit_override"] = 2

    model_path = str(config["model"]["path"])
    # Load once for param count; capability eval uses model_path via lm_eval HFLM.
    model, _tokenizer = load_qwen_for_eval(
        model_path,
        device=str(config.get("hardware", {}).get("device", "cuda:0")),
        torch_dtype=str(config["model"].get("torch_dtype", "float16")),
    )
    params = count_params(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cap_out = eval_capability_vector(model_path, run_config)
    mem = gpu_mem_gb()
    elapsed = time.time() - t0

    vector = cap_out["vector"]
    resources = {
        "parameter_count": params,
        "gpu_memory_gb_peak": mem,
        "latency_sec_eval": elapsed,
        "model_size_path": model_path,
    }
    summary = {
        "experiment_id": "E0",
        "title": "Dense Model Baseline",
        "purpose": "建立所有后续实验的 dense baseline",
        "model": config["model"].get("spec", "unknown"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": "none",
        "compression_or_sparsity": "None",
        "evaluation": "lm_eval six-dim scan (wikitext/gsm8k/mmlu/bbh/ifeval/humaneval)",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device", "cuda"),
        "priority": "P0",
        "success_criteria": "所有 benchmark pipeline 可稳定复现",
        "status": "smoke" if args.smoke else "done",
        "outputs": {"P_M0": vector, "resources": resources},
        "capability": cap_out,
    }
    write_json(out / "e0_summary.json", summary)

    report = _build_report(config, cap_out, resources, elapsed, smoke=args.smoke)
    write_report(out / "e0_report.md", report)
    if not args.smoke:
        mirror_docs_report("E0", "dense_baseline", report)
    print(f"[OK] E0 wrote {out}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
