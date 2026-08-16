# 工作日志（精简）

> 详细版：[WORK_LOG.md](WORK_LOG.md) · 证据包：[EVIDENCE_PACK.md](EVIDENCE_PACK.md)
> **论文成果提纲**：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) · Phase J：[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) · Phase K：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 日期：2026-08-16
> 图例：`[√]` 已完成 · `[×]` 有问题/证据不支持 · `[ ]` 未做

## 主主张（定稿）

CIFAR 上 search vs iterative 是 **regime-dependent**（≤4x iterative 略稳；≥8x search 更高）。
[×] 「系统全面更优」— 证据不支持。

## LLM / SQuAD 结论（更新）

| 项 | 结论 |
|----|------|
| 短/中 SGD 恢复 | 负对照：剪枝后 F1≈0（保留） |
| LoRA @1.5x（8192×2） | F1 可读；预算 1.50x 对齐 |
| Informal 排序（n=64） | **iterative 38.9 ≥ oneshot 30.6 ≈ search 30.0**（dense 25.5） |
| 论文口径 | 附录/讨论 Informal；**非** LLM 主表；**不**写 search 更优 |
| 2x / frozen test | 暂不扩 / 不开 |

## 已做

1. [√] formal100 + 机制链；叙事定稿。
2. [√] KG.5 GLUE 过渡门禁；K6 预算对齐 + SQuAD 小扫。
3. [√] SGD 加深失败对照 + LoRA R1/方法扩展。
4. [√] K6-lit 短表。

## 下一步

1. 同步 PAPER_RESULTS_OUTLINE / EVIDENCE_PACK LLM 节为本口径
2. Related Work 挂钩 LLM-Pruner（LoRA 恢复）
3. 不默认扩 2x / 不重跑 formal100
