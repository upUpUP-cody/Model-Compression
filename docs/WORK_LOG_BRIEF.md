# 工作日志（精简）

> 详细版：[WORK_LOG.md](WORK_LOG.md) · 证据包：[EVIDENCE_PACK.md](EVIDENCE_PACK.md)
> **论文成果提纲**：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) · Phase J：[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) · Phase K：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 日期：2026-08-16
> 图例：`[√]` 已完成 · `[×]` 有问题/证据不支持 · `[ ]` 未做

## 主主张（定稿）

CIFAR 上 search vs iterative 是 **regime-dependent**（≤4x iterative 略稳；≥8x search 更高）。
[×] 「系统全面更优」— 证据不支持。

## 已做

1. [√] formal100 + 机制链；叙事定稿。
2. [√] KG.5 GLUE 过渡门禁；K6 预算对齐 + SQuAD 小扫。
3. [√] **加深恢复 1.5x**（4 epoch / 512 train / 64 eval）— 剪枝后 F1 仍塌。
4. [√] K6-lit 短表：[K6_LIT_BASELINE_SHORTLIST.md](K6_LIT_BASELINE_SHORTLIST.md)

## 现在处于哪一步

加深恢复 1.5x（`/mnt/data2/results/qwen_k6_recover_1p5x/`，n=64）：

| 方法 | 压缩 | F1 | CE |
|------|------|-----|-----|
| dense | 1.00x | 25.5 | 1.90 |
| oneshot | 1.50x | 0.24 | 14.8 |
| iterative_level1 | 1.50x | 0.0 | 7.33 |
| autonomous_search | 1.50x | 0.0 | 5.59 |

**验收未过** → **不扩 2x**。论文 LLM 节按 **局限/负结果** 收口（主贡献仍 CIFAR + GLUE 过渡）。

## 下一步

1. 写论文 Limitations（SQuAD 短/中恢复不足）
2. 可选：更强恢复或换设定后再试（非必须挡收口）
3. frozen test / 方法 crossover 叙事延后
