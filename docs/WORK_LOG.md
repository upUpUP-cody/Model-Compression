# 工作日志（按实验 ID）

> **唯一标准**：[refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf](refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf) §31 整体实验清单（E0–E14）。
> 要求摘录：[PDF_E_REQUIREMENTS_31_34.md](PDF_E_REQUIREMENTS_31_34.md) · 状态总表：[EXPERIMENT_E_MAP.md](EXPERIMENT_E_MAP.md) · 精简：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md) · 交付：[MENTOR_DELIVERY.md](MENTOR_DELIVERY.md)
> 更新：2026-08-20 · 主机：1×RTX 4090 · E 产物：`/mnt/data2/results/E{n}_*/`

本日志**按 E ID 分节**。不再使用 Phase H/I/J/K、P1.2、KG.*、RQ* 作为主编号。历史视觉/过渡工作归入文末 **Non-E**，不计入 E 完成。

---

## 执行顺序与 Gates（§32–34）

| 批次 | ID | 说明 |
|------|-----|------|
| 第一批前半 | E0→E1→E2→E3 | Stage A；Compressibility Frontier |
| 第一批后半 | E8→E9 | Recovery / High-Gap；须 E3 后 |
| Stage B | E4–E7 | 过 Gate A 后再开 |
| Stage C 余 | E10–E11 | 过 Gate B 相关信号后再开 |
| Stage D | E12–E14 | 禁止插队 |

| Gate | 依赖 | 失败动作 |
|------|------|----------|
| A | E1/E2 | 暂缓 Agent |
| B | E3/E9 | 放弃 Frontier 数据主线 |
| C | E5–E7 | 可不做复杂 controller |
| D | E11 | synthetic 仅 optional |
| E | E13 | 不声称完全 self-compressing |

---

## E0 — Dense baseline（P0）

| 项 | 内容 |
|----|------|
| 目的 | 建立评价基准 |
| 状态 | **done_proxy**（`proxy_1.5B`；PDF 要 Qwen2.5-3B） |
| 产物 | `/mnt/data2/results/E0_dense_baseline/` · [e_reports/E0_dense_baseline.md](e_reports/E0_dense_baseline.md) |
| 要点 | SST-2 LM PPL≈1.99；Instruction proxy_acc≈59.3；Math/Knowledge/Reasoning/Code = n/a |
| 成功条件（PDF） | 全 benchmark pipeline 可复现 → **部分满足** |

---

## E1 — One-shot sparsity curve（P0）

| 项 | 内容 |
|----|------|
| 目的 | Frontier 是否存在（performance cliff / capability-specific） |
| 状态 | **done_proxy** |
| 产物 | `/mnt/data2/results/E1_oneshot_sparsity_curve/`（含 `figures/performance_vs_sparsity.png`） · [报告](e_reports/E1_oneshot_sparsity_curve.md) |
| 要点 | sparsity 10%–70%；方法为 magnitude MLP **代理** Wanda；剪后 PPL 爆炸 |
| Gate A | **暂不可判**（方法过糙，非可信 cliff） |

---

## E2 — One-shot vs iterative（P0）

| 项 | 内容 |
|----|------|
| 目的 | 逐步搜索是否必要 |
| 状态 | **done_proxy** |
| 产物 | `/mnt/data2/results/E2_iterative_vs_oneshot/` · [报告](e_reports/E2_iterative_vs_oneshot.md) |
| 要点 | 仅 40%/50%、单 seed；无稳定 iterative 优势 |
| Gate A | **暂不可判** |

---

## E3 — Compression Gap（P0）

| 项 | 内容 |
|----|------|
| 目的 | Gap 是否预测损伤 |
| 状态 | **done_proxy** |
| 产物 | `/mnt/data2/results/E3_compression_gap/` · [报告](e_reports/E3_compression_gap.md) |
| 要点 | pool n=64（PDF 要 2k–5k）；corr=nan；信号弱 |
| Gate B | 须与 **E9** 一并判断；仅 E3 不足下最终结论 |

---

## E4 — Fixed iterative（P0）

| 状态 | pending（Stage B；须 Gate A 后再开） |
|------|--------------------------------------|
| 核心结论 | Autonomous baseline |

---

## E5 — Adaptive ratio（P1）

| 状态 | pending |
|------|---------|
| 核心结论 | 自适应步长是否有效 |

---

## E6 — Adaptive region（P1）

| 状态 | pending |
|------|---------|
| 核心结论 | Capability feedback 是否有效 |

---

## E7 — Rewind + regrow（P1）

| 状态 | pending |
|------|---------|
| 核心结论 | 可逆搜索是否有效 |

---

## E8 — Random recovery（P0）

| 状态 | **pending**（第一批后半；E3 审阅后启动） |
|------|------------------------------------------|
| PDF 规格摘要 | Child 50% sparse；LoRA 200–500 steps；Random 数据 256/512/1k；seeds 3 |
| 产物约定 | `/mnt/data2/results/E8_random_recovery/` |

---

## E9 — High-gap recovery（P0）

| 状态 | **pending**（紧接 E8） |
|------|------------------------|
| 核心结论 | Compression-specific data |
| PDF 期望 | HighGap > StudentHard > Random > LowGap |
| 产物约定 | `/mnt/data2/results/E9_high_gap_recovery/` |

---

## E10 — Cross-compression（P1）

| 状态 | pending |
|------|---------|
| 核心结论 | Policy-specific frontier |

---

## E11 — Synthetic frontier（P1）

| 状态 | pending |
|------|---------|
| 核心结论 | BigBang-style generator · Gate D |

---

## E12 — Controller comparison（P1）

| 状态 | pending（Stage D） |
|------|-------------------|
| 核心结论 | LLM controller 是否必要 |

---

## E13 — Self vs external（P1）

| 状态 | pending（Stage D；**禁止插队**） |
|------|--------------------------------|
| 核心结论 | Self-governance frontier · Gate E |

---

## E14 — Full autonomous run（P2）

| 状态 | pending |
|------|---------|
| 核心结论 | Autonomous Lottery Ticket |

---

## Non-E — 视觉 / 过渡附录（不计 E 完成）

> 历史协议与数字；**不得**改贴为已完成 En。详表曾在 `archive/docs/EVIDENCE_PACK.md`。

### Non-E · CIFAR formal100（视觉主表候选）

- 设定：ResNet-18 / CIFAR-10；基线 test **90.36%**；seeds 42/43/44
- 产物：`results/cifar_p12_comparison_gpu_formal100_full/`
- 结论：**regime-dependent**（低–中压缩 iterative 略稳；高压缩 ≥8x search 同压缩更强）；**不得**写 search 全面更优；oneshot ≥4x 崩溃保留

| 目标 | iterative test | search test |
|------|----------------|-------------|
| 1.5x | 90.52±0.30 | 90.64±0.94 |
| 2.0x | 90.66±0.12 | 90.31±0.52 |
| 4.0x | 89.85±0.16 | 89.37±0.16 |
| 6.0x | 88.22±0.38 | 88.25±0.56 |
| 8.0x | 86.67±0.16 | 87.40±0.97 |
| 10.0x | 83.76±0.43 | 86.03±0.33 |

### Non-E · GLUE 过渡

- 产物：`/mnt/data2/results/qwen_glue_kg6/`（路径名历史保留）
- 设定：Qwen2.5-1.5B；SST-2 / RTE / QNLI；1.5x/2.0x；预算对齐
- 口述：dense ≫ pruned；三剪枝互有胜负，无 search 全面更优

### Non-E · SQuAD LoRA formal

- 产物：`/mnt/data2/results/qwen_k6_lora_formal/`
- 难压局限附录；dense 无 LoRA → pruned F1>dense **不可**当优势；单 seed

### Non-E · MNIST

- 产物：`results/p12_comparison_gpu_sweep/` 等
- 迭代/搜索+恢复在高压缩下约 97–98%；one-shot 无恢复过约 2x 崩溃

---

## 下一步（锁定）

1. 审阅 E0–E3 与 Gate A（当前 proxy **不足以**判 Yes）
2. 规划并跑 **E8→E9**
3. 不启动 E13；仍单卡即可
4. 主文是否只押 CIFAR（Non-E）仍待导师拍板
