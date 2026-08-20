#!/usr/bin/env python3
"""E3 Compression Gap correlation (PDF Stage A)."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_a_common import (
    ensure_dir,
    load_model,
    load_yaml,
    mirror_docs_report,
    prepare_sst2_loaders,
    prune_mlp_ratio,
    write_json,
    write_report,
)


@torch.no_grad()
def per_example_nll(model, batch, device: str) -> List[float]:
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch.get("labels", input_ids)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100 if (shift_labels == -100).any() else -100,
    )
    # mean over tokens per example
    token_loss = loss_flat.view(shift_labels.size(0), -1)
    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    nll = (token_loss * mask).sum(dim=1) / denom
    return [float(x) for x in nll.detach().cpu()]


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx < 1e-12 or deny < 1e-12:
        return float("nan")
    return float(num / (denx * deny))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e3_compression_gap.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    ensure_dir(out / "figures")

    print("[INFO] E3 compression gap starting")
    parent, tokenizer, device = load_model(config)
    _, val_loader, _ = prepare_sst2_loaders(config, tokenizer, device)
    sp = float(config.get("child_sparsity", 0.5))
    child = prune_mlp_ratio(copy.deepcopy(parent), sp).to(device)

    gaps: List[float] = []
    parent_nlls: List[float] = []
    child_nlls: List[float] = []
    failures: List[float] = []
    lengths: List[float] = []

    max_samples = int(config.get("pool_max_samples", 64))
    seen = 0
    for batch in val_loader:
        if seen >= max_samples:
            break
        pn = per_example_nll(parent, batch, device)
        cn = per_example_nll(child, batch, device)
        ids = batch["input_ids"]
        for i in range(len(pn)):
            if seen >= max_samples:
                break
            g = cn[i] - pn[i]
            gaps.append(g)
            parent_nlls.append(pn[i])
            child_nlls.append(cn[i])
            failures.append(1.0 if cn[i] > pn[i] + 0.05 else 0.0)
            lengths.append(float(ids.size(1)))
            seen += 1

    corr_gap_fail = pearson(gaps, failures)
    corr_parent = pearson(parent_nlls, failures)
    corr_child = pearson(child_nlls, failures)
    corr_len = pearson(lengths, failures)
    # random score baseline
    import random

    rnd = [random.random() for _ in failures]
    corr_rand = pearson(rnd, failures)

    go = corr_gap_fail == corr_gap_fail and corr_gap_fail > max(
        corr_rand if corr_rand == corr_rand else -1.0,
        0.05,
    )

    summary = {
        "experiment_id": "E3",
        "title": "Compression-Induced Capability Gap",
        "purpose": "验证 Compression Gap 是否能描述真实能力损失",
        "model": config["model"].get("spec"),
        "pdf_model": config["model"].get("pdf_model"),
        "spec": config["model"].get("spec"),
        "method": f"Parent dense vs Child sparsity={sp}; G=L_child-L_parent",
        "compression_or_sparsity": sp,
        "evaluation": f"SST-2 pool n={seen}; corr(G, Failure)",
        "seeds": config.get("seed", 42),
        "gpu": config.get("hardware", {}).get("device"),
        "priority": "P0",
        "success_criteria": "Gap 显著预测 downstream degradation（优于 teacher/student NLL / length / random）",
        "status": "done_proxy",
        "outputs": {
            "corr_gap_failure": corr_gap_fail,
            "corr_parent_nll_failure": corr_parent,
            "corr_child_nll_failure": corr_child,
            "corr_length_failure": corr_len,
            "corr_random_failure": corr_rand,
            "n": seen,
        },
        "records": {"mean_gap": sum(gaps) / max(len(gaps), 1)},
        "notes_proxy": "Failure defined as child NLL > parent NLL + 0.05 on SST-2 LM; not full AUROC suite.",
    }
    write_json(out / "e3_summary.json", summary)

    report = f"""# E3. Compression-Induced Capability Gap

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E3 |
| 目的 | 验证 Compression Gap 是否能描述真实能力损失 |
| Model | {config["model"].get("spec")} (PDF: {config["model"].get("pdf_model")}) |
| Method / Compression | Parent Dense；Child oneshot MLP sparsity={sp*100:.0f}%；Recovery **None** |
| Evaluation | pool n={seen}；G=NLL_child−NLL_parent；Failure=G>0.05 |
| Seeds | {config.get("seed", 42)} |
| GPU | {config.get("hardware", {}).get("device")} |
| 优先级 | P0 |
| 成功条件 | Gap 显著预测 degradation（相对 teacher/student NLL、length、random） |
| status | done_proxy |

## 输出

核心指标：correlation /（本跑未算完整 AUROC）。

## 记录表

| Score | corr with Failure |
|-------|-------------------|
| Compression Gap G | {corr_gap_fail:.4f} |
| Parent NLL | {corr_parent:.4f} |
| Child NLL | {corr_child:.4f} |
| Prompt length | {corr_len:.4f} |
| Random | {corr_rand:.4f} |

## 结论（对照成功条件）

- {"**部分满足 / Go 倾向**：Gap 相关高于 random 阈值。" if go else "**No-Go / 信号弱（本 proxy）**：Gap 未明显高于对照。"}
- **Gate B 输入**：需与后续 **E9** 一并判断是否放弃 Frontier 数据主线；仅 E3 不足以下最终结论。
- 规格：`proxy_1.5B`；PDF 要 2k–5k pool + 多 child sparsity。
"""
    write_report(out / "e3_report.md", report)
    mirror_docs_report("E3", "compression_gap", report)
    print(f"[OK] E3 wrote {out}")


if __name__ == "__main__":
    main()
