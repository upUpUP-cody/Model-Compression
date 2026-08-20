#!/usr/bin/env python3
"""E1 One-shot sparsity curve (PDF Stage A)."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.qwen_k5_comparison import count_params
from src.experiments.stage_a_common import (
    ensure_dir,
    eval_proxy,
    load_model,
    load_yaml,
    mirror_docs_report,
    prepare_sst2_loaders,
    prune_mlp_ratio,
    write_json,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_oneshot_curve.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    figures = ensure_dir(out / "figures")

    print("[INFO] E1 oneshot sparsity curve starting")
    model, tokenizer, device = load_model(config)
    _, val_loader, _ = prepare_sst2_loaders(config, tokenizer, device)
    dense = eval_proxy(model, val_loader, device)
    dense_ppl = dense["ppl"]

    rows = []
    grid = [float(x) for x in config.get("sparsity_grid", [0.1, 0.3, 0.5, 0.7])]
    for sp in grid:
        print(f"[INFO] E1 sparsity={sp:.2f}")
        child = prune_mlp_ratio(copy.deepcopy(model), sp).to(device)
        m = eval_proxy(child, val_loader, device)
        rows.append(
            {
                "sparsity": sp,
                "ppl": m["ppl"],
                "delta_ppl": m["ppl"] - dense_ppl,
                "proxy_acc": m["proxy_accuracy"],
                "params": count_params(child),
            }
        )
        del child
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    xs = [r["sparsity"] * 100 for r in rows]
    ys = [r["delta_ppl"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Sparsity % (MLP intermediate prune)")
    plt.ylabel("Delta PPL vs dense")
    plt.title("E1 performance vs sparsity (proxy)")
    plt.grid(True, alpha=0.3)
    fig_path = figures / "performance_vs_sparsity.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    summary = {
        "experiment_id": "E1",
        "title": "One-shot Sparsity Curve",
        "purpose": "判断是否存在明显 performance cliff",
        "model": config["model"].get("spec"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": "oneshot_mlp_magnitude_proxy (PDF: Wanda)",
        "compression_or_sparsity": grid,
        "evaluation": "SST-2 LM PPL delta; Recovery None",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device"),
        "priority": "P0",
        "success_criteria": "存在明显非线性 degradation / capability-specific degradation",
        "status": "done_proxy",
        "outputs": {"figure": str(fig_path)},
        "records": {"dense": dense, "curve": rows},
        "notes_proxy": "Wanda not wired; magnitude MLP prune. Single capability (SST-2 PPL).",
    }
    write_json(out / "e1_summary.json", summary)

    table_lines = ["| Sparsity | Delta PPL | PPL | proxy_acc |", "|----------|-----------|-----|-----------|"]
    for r in rows:
        table_lines.append(
            f"| {r['sparsity']*100:.0f}% | {r['delta_ppl']:.3f} | {r['ppl']:.3f} | {r['proxy_acc']:.2f} |"
        )
    table = "\n".join(table_lines)

    # crude cliff check: max consecutive second difference
    deltas = [r["delta_ppl"] for r in rows]
    cliff = False
    if len(deltas) >= 3:
        for i in range(2, len(deltas)):
            if deltas[i] - deltas[i - 1] > 1.5 * max(1e-6, abs(deltas[i - 1] - deltas[i - 2]) + 0.5):
                cliff = True
                break
        if deltas[-1] > deltas[0] + 2.0:
            cliff = True

    report = f"""# E1. One-shot Sparsity Curve

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E1 |
| 目的 | 判断是否存在明显 performance cliff |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | oneshot MLP magnitude（PDF 写 Wanda；本跑为 proxy） |
| Evaluation | SST-2 LM PPL；Recovery **None** |
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | 存在明显非线性 degradation / capability-specific degradation |
| status | done_proxy |

## 输出

核心图：`figures/performance_vs_sparsity.png`

## 记录表

Dense PPL = {dense_ppl:.4f}

{table}

## 结论（对照成功条件）

- {"**部分满足**：PPL 随 sparsity 上升，见曲线；" if cliff or deltas[-1] > deltas[0] else "**不满足/信号弱**："}单任务 proxy，非 PDF 多能力 cliff。
- **Gate A 输入**：本跑提供 E1 侧证据（需与 E2 一并判断是否做 Agent）。
- 规格：`proxy_1.5B`；方法非 Wanda。
"""
    write_report(out / "e1_report.md", report)
    mirror_docs_report("E1", "oneshot_sparsity_curve", report)
    print(f"[OK] E1 wrote {out}")


if __name__ == "__main__":
    main()
