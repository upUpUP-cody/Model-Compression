# 工作日志（精简）

> 详细版：[WORK_LOG.md](WORK_LOG.md) · 证据包：[EVIDENCE_PACK.md](EVIDENCE_PACK.md)
> **论文成果提纲**：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) · Phase J：[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) · Phase K：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 日期：2026-08-15
> 图例：`[√]` 已完成 · `[×]` 有问题/证据不支持 · `[ ]` 未做

## 主主张（定稿）

CIFAR 上 search vs iterative 是 **regime-dependent**（≤4x iterative 略稳；≥8x search 更高）。
[×] 「系统全面更优」— 证据不支持。

## 已做

1. [√] formal100 + 4x/8x/10x 机制链；叙事定稿。
2. [√] 预算对齐；一步到位复验与 4x 诊断（策略收窄 `target<=2`）。
3. [√] Phase J 规划；**Phase K 冒烟已跑通**（Qwen2.5-1.5B-Instruct + SQuAD；`/mnt/data`）。
4. [√] 论文成果提纲：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md)。

## 现在处于哪一步

Phase K dense/oneshot 冒烟完成（产物 `/mnt/data/results/qwen_squad_smoke/`）。
不预设 LLM 上 search 全面更优。系统盘已清理；视觉 results 在 `/mnt/data/results/vision`。

## 下一步

接线 iterative / autonomous_search 小矩阵（1.5x–4x）。建议 `/mnt/data` 扩到合计约 **100G**。不重跑 formal100。
