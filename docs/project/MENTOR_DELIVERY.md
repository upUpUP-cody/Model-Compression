# 导师交付页（打开就能念）

> **唯一标准**：experiment_plan.pdf §31 E0–E14。进度：[EXPERIMENT_E_MAP.md](EXPERIMENT_E_MAP.md) · 要求：[PDF_E_REQUIREMENTS_31_34.md](PDF_E_REQUIREMENTS_31_34.md) · 日志：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md) / [WORK_LOG.md](WORK_LOG.md)
> 更新：2026-08-20

---

## 纲领 E 进度

| E | 状态 | 报告 |
|---|------|------|
| E0 Dense baseline | done_proxy (1.5B) | [../results/E0](../results/E0_dense_baseline.md) · `/mnt/data2/results/E0_dense_baseline/` |
| E1 One-shot curve | done_proxy | [../results/E1](../results/E1_oneshot_sparsity_curve.md) |
| E2 Iterative vs one-shot | done_proxy | [../results/E2](../results/E2_iterative_vs_oneshot.md) |
| E3 Compression Gap | done_proxy | [../results/E3](../results/E3_compression_gap.md) |
| E4–E7 | pending | Stage B（Gate A 后） |
| E8–E9 | partial_smoke / pending | [process/NEXT_E8_E9](../process/NEXT_E8_E9.md) · [results/E8](../results/E8_random_recovery.md) |
| E10–E14 | pending | E13 **不插队** |

第一批（§34）：E0–E3 已跑 proxy；E8 冒烟 cell 已落盘；续跑见 [`../process/NEXT_E8_E9.md`](../process/NEXT_E8_E9.md)。Gate A 本 proxy 暂不可判。

---

## Non-E · CIFAR formal100（视觉附录 · 必问）

**设定**：ResNet-18 / CIFAR-10；dense test **90.36%**；seeds 42/43/44；`results/cifar_p12_comparison_gpu_formal100_full/`。
**口述**：crossover → **regime-dependent**；非「search 全面更优」。oneshot ≥4x 崩溃保留。

| 目标 | iterative test | search test | 读法 |
|------|----------------|-------------|------|
| 1.5x | 90.52±0.30 | 90.64±0.94 | 接近 |
| 2.0x | **90.66±0.12** | 90.31±0.52 | iterative 略优 |
| 4.0x | **89.85±0.16** | 89.37±0.16 | iterative 略优 |
| 6.0x | 88.22±0.38 | 88.25±0.56 | 持平 |
| 8.0x | 86.67±0.16 | **87.40±0.97** | search 略优 |
| 10.0x | 83.76±0.43 | **86.03±0.33** | search 更优（约 +2.3） |

历史详表：`archive/docs/EVIDENCE_PACK.md`（已归档，不作主编号）。

---

## Non-E · LLM 过渡附录

### GLUE（SST-2 / RTE / QNLI）

产物：`/mnt/data2/results/qwen_glue_kg6/kg6_summary.json`（路径名历史保留）。dense ≫ pruned；无 search 全面更优。

| Task | Target | Dense Acc | Oneshot | Iterative | Search |
|------|--------|-----------|---------|-----------|--------|
| SST-2 | 1.5x | 85.94 | 45.31 | 54.69 | 54.69 |
| SST-2 | 2.0x | 85.94 | 52.34 | 45.31 | 42.19 |
| RTE | 1.5x | 85.94 | 47.66 | 48.44 | 48.44 |
| RTE | 2.0x | 85.94 | 50.00 | 50.00 | 50.00 |
| QNLI | 1.5x | 81.25 | 53.12 | 46.88 | 53.12 |
| QNLI | 2.0x | 81.25 | 53.12 | 53.12 | 53.91 |

### SQuAD LoRA formal（难压局限）

产物：`/mnt/data2/results/qwen_k6_lora_formal/k6_lora_formal_summary.json`。dense **无** LoRA → pruned>dense **不可**当优势。

| Target | Method | Actual | Val F1 | Frozen F1 |
|--------|--------|--------|--------|-----------|
| — | dense | 1.00x | 23.98 | 15.17 |
| 1.5x | oneshot | 1.50x | 31.48 | 25.87 |
| 1.5x | iterative | 1.50x | 32.15 | 27.74 |
| 1.5x | search | 1.50x | 32.60 | 24.26 |
| 2.0x | oneshot | 2.00x | 26.73 | 19.58 |
| 2.0x | iterative | 2.00x | 28.58 | 24.49 |
| 2.0x | search | 2.00x | 21.34 | 16.56 |

---

## 路线口径（仅 E ID）

| 项 | 状态 |
|----|------|
| 执行顺序 | **E0→E1→E2→E3→E8→E9→…** |
| Non-E 视觉 | CIFAR regime-dependent |
| E9 | 第一批后半（E3/E8 后） |
| E13 | Stage D；当前**不**开 |
| 硬件 | 默认 **1×4090** |

---

## 已知问题（如实）

| 项 | 说明 |
|----|------|
| E0–E3 proxy | 1.5B + magnitude 代理 Wanda；Gate A/B 暂不可判 |
| CIFAR oneshot ≥4x | 正式表仍崩溃 |
| SQuAD 恢复不对称 | dense 无 LoRA；不可写 pruned>dense 优势 |
| LLM 单 seed / 小样本 | 附录级 |

---

## 问题分级

| 级别 | 动作 |
|------|------|
| **P0** | 当轮明确说 + 写入待拍板 |
| **P1** | WORK_LOG + 本节 |
| **P2** | 日志即可 |

---

## 需导师拍板

**已覆盖（以 PDF 为准）**：下一档按 **E8→E9**，不以口头 RQ 打乱 E 序。

**仍待拍板**

1. 本期论文是否确认 **主文只押 CIFAR（Non-E）**，LLM 仅过渡附录？

---

## 产物路径速查

| 内容 | 路径 |
|------|------|
| E0–E3 | `/mnt/data2/results/E{0-3}_*/` · `docs/results/` |
| CIFAR formal100 | `results/cifar_p12_comparison_gpu_formal100_full/` |
| GLUE | `/mnt/data2/results/qwen_glue_kg6/` |
| SQuAD LoRA formal | `/mnt/data2/results/qwen_k6_lora_formal/` |
| 历史证据包 | `archive/docs/EVIDENCE_PACK.md` |
