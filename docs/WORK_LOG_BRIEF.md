# 工作日志（精简）

> **导师交付（结果表）**：[MENTOR_DELIVERY.md](MENTOR_DELIVERY.md) ← 打开就能念
> 详细版：[WORK_LOG.md](WORK_LOG.md) · 证据包：[EVIDENCE_PACK.md](EVIDENCE_PACK.md)
> **论文成果提纲**：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) · Phase J：[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) · Phase K：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 日期：2026-08-20
> 图例：`[√]` 已完成 · `[~]` 暂缓 · `[×]` 有问题/证据不支持 · `[ ]` 未做

## 主主张（定稿）

CIFAR 上 search vs iterative 是 **regime-dependent**（≤4x iterative 略稳；≥8x search 更高）。
[×] 「系统全面更优」— 证据不支持。

## LLM / RQ 口径（导师）

| 项 | 结论 |
|----|------|
| 顺序 | **先 GLUE、后 SQuAD** — 符合导师 |
| KG.6 GLUE | [√] 过渡信号可读 |
| SQuAD | 难压属预期；formal 附录；非主表 |
| RQ4 Self-Governance | **[ ] 必做 · 下一档**（External vs Self；优先 GLUE 冒烟） |
| RQ3 High-Gap | **[ ] RQ4 之后做**（不插队；非取消） |
| 论文 | 主文 CIFAR；LLM 过渡 + 难压局限（主文范围仍待拍板） |

## 已做

1. [√] formal100 + 机制链；叙事定稿。
2. [√] KG.5/KG.6 GLUE；K6 SQuAD + LoRA formal。
3. [√] PDF 对齐审计写入 PROJECT_PLAN / PHASE_K（导师口径）。

## 下一步

1. RQ4 最小矩阵规划/冒烟（必做）
2. RQ3 / E9 — **仅 RQ4 有结论后再排**
3. 论文口径收口（CIFAR 主文范围仍待拍板）；不重跑 formal100
