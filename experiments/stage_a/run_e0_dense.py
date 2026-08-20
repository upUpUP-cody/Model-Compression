#!/usr/bin/env python3
"""E0 Dense Model Baseline (PDF Stage A)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_a_common import (
    ensure_dir,
    eval_proxy,
    gpu_mem_gb,
    load_model,
    load_yaml,
    mirror_docs_report,
    prepare_sst2_loaders,
    write_report,
)
from src.experiments.qwen_k5_comparison import count_params
from src.utils.qwen_squad_eval import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e0_dense.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    figures = ensure_dir(out / "figures")

    print("[INFO] E0 dense baseline starting (proxy_1.5B if marked)")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer, device = load_model(config)
    _, val_loader, meta = prepare_sst2_loaders(config, tokenizer, device)
    metrics = eval_proxy(model, val_loader, device)
    params = count_params(model)
    mem = gpu_mem_gb()
    elapsed = time.time() - t0

    # PDF vector slots not fully instrumented yet -> mark n/a
    vector = {
        "PPL": metrics["ppl"],
        "Math": None,
        "Knowledge": None,
        "Reasoning": None,
        "Instruction": metrics["proxy_accuracy"],
        "Code": None,
    }
    resources = {
        "parameter_count": params,
        "gpu_memory_gb_peak": mem,
        "latency_sec_eval": elapsed,
        "model_size_path": config["model"]["path"],
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
        "evaluation": "SST-2 LM proxy PPL + proxy_acc (Math/Code TBD)",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device", "cuda"),
        "priority": "P0",
        "success_criteria": "所有 benchmark pipeline 可稳定复现",
        "status": "done_proxy",
        "outputs": {"P_M0": vector, "resources": resources},
        "records": {"sst2_val": metrics, "split_meta": meta},
        "notes_proxy": "PDF asks Qwen2.5-3B + full capability vector; this run is 1.5B SST-2 proxy.",
    }
    write_json(out / "e0_summary.json", summary)

    report = f"""# E0. Dense Model Baseline

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E0 |
| 目的 | 建立所有后续实验的 dense baseline |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | None |
| Evaluation | SST-2 LM proxy PPL + Instruction proxy_acc；Math/Knowledge/Reasoning/Code = n/a（待扩） |
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| status | done_proxy |

## 输出

Dense performance vector P(M0) 与资源行（PDF §E0）。

## 记录表

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| {vector["PPL"]:.4f} | n/a | n/a | n/a | {vector["Instruction"]:.2f} | n/a |

| GPU memory (GB peak) | latency_sec_eval | parameter_count | model path |
|----------------------|------------------|-----------------|------------|
| {mem if mem is not None else "n/a"} | {elapsed:.1f} | {params} | `{config["model"]["path"]}` |

## 结论（对照成功条件）

- **部分满足**：pipeline 可复现（load + SST-2 LM eval）；完整六维能力向量未齐。
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E。
- 规格：`spec=proxy_1.5B`（PDF 要 3B）。
"""
    write_report(out / "e0_report.md", report)
    mirror_docs_report("E0", "dense_baseline", report)
    print(f"[OK] E0 wrote {out}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
