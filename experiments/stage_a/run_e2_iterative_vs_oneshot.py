#!/usr/bin/env python3
"""E2 Iterative vs One-shot (PDF Stage A)."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

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


def iterative_prune(model, target_sparsity: float, steps: int, device: str):
    """Equal-ratio incremental prune toward target (no recovery)."""
    current = copy.deepcopy(model)
    # compound: each step prune fraction f of remaining so total ~ target
    # approx: per-step prune_ratio = 1 - (1-target)^(1/steps)
    remain = (1.0 - float(target_sparsity)) ** (1.0 / max(int(steps), 1))
    step_prune = 1.0 - remain
    for _ in range(int(steps)):
        current = prune_mlp_ratio(current, step_prune).to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    ensure_dir(out / "figures")

    print("[INFO] E2 iterative vs oneshot starting")
    model, tokenizer, device = load_model(config)
    _, val_loader, _ = prepare_sst2_loaders(config, tokenizer, device)
    dense = eval_proxy(model, val_loader, device)
    steps = int(config.get("iterative_steps", 4))
    targets = [float(x) for x in config.get("target_sparsities", [0.4, 0.5])]

    rows = []
    for sp in targets:
        print(f"[INFO] E2 target_sparsity={sp:.2f}")
        one = prune_mlp_ratio(copy.deepcopy(model), sp).to(device)
        m_one = eval_proxy(one, val_loader, device)
        del one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        it = iterative_prune(model, sp, steps, device)
        m_it = eval_proxy(it, val_loader, device)
        del it
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows.append(
            {
                "target_sparsity": sp,
                "oneshot_ppl": m_one["ppl"],
                "iterative_ppl": m_it["ppl"],
                "oneshot_better": m_one["ppl"] < m_it["ppl"],
                "iterative_better": m_it["ppl"] < m_one["ppl"],
            }
        )

    iter_wins = sum(1 for r in rows if r["iterative_better"])
    success = iter_wins == len(rows) and len(rows) > 0

    summary = {
        "experiment_id": "E2",
        "title": "Iterative vs One-shot Compression",
        "purpose": "验证逐步压缩是否优于一次性压缩",
        "model": config["model"].get("spec"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": "Baseline A oneshot vs Baseline B incremental; Recovery None",
        "compression_or_sparsity": targets,
        "evaluation": "SST-2 LM PPL (Final performance proxy)",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device"),
        "priority": "P0",
        "success_criteria": "iterative 稳定优于 one-shot",
        "status": "done_proxy",
        "outputs": {"rows": rows, "dense_ppl": dense["ppl"]},
        "records": {"dense": dense, "comparison": rows},
        "notes_proxy": "Smoke targets 40/50% only; magnitude MLP prune; no Wanda.",
    }
    write_json(out / "e2_summary.json", summary)

    table = ["| Target sparsity | One-shot PPL | Iterative PPL | Winner |",
             "|-----------------|--------------|---------------|--------|"]
    for r in rows:
        w = "iterative" if r["iterative_better"] else ("oneshot" if r["oneshot_better"] else "tie")
        table.append(
            f"| {r['target_sparsity']*100:.0f}% | {r['oneshot_ppl']:.3f} | {r['iterative_ppl']:.3f} | {w} |"
        )

    report = f"""# E2. Iterative vs One-shot Compression

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E2 |
| 目的 | 验证逐步压缩是否优于一次性压缩 |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | Baseline A One-shot；Baseline B {steps}-step incremental；Recovery **None** |
| Evaluation | SST-2 LM PPL（Final performance proxy） |
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | iterative 稳定优于 one-shot |
| status | done_proxy |

## 输出

核心指标：Final performance（PPL，越低越好）。Dense PPL = {dense["ppl"]:.4f}

## 记录表

{chr(10).join(table)}

## 结论（对照成功条件）

- {"**满足（本 proxy）**：各档 iterative PPL 更低。" if success else f"**不满足/未稳定**：iterative 胜 {iter_wins}/{len(rows)} 档。"}
- **Gate A**：与 E1 一并判断是否存在 iterative advantage；本跑 {"支持" if success else "不支持"}「必须做 Agent」的前提之一。
- 规格：`proxy_1.5B`；PDF 目标 40/50/60%×3 seed 未全开。
"""
    write_report(out / "e2_report.md", report)
    mirror_docs_report("E2", "iterative_vs_oneshot", report)
    print(f"[OK] E2 wrote {out}")


if __name__ == "__main__":
    main()
