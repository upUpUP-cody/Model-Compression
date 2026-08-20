# 导师交付页（打开就能念）

> **维护约定**：交付数字**只改本文件**；改数须对照下方产物 JSON / [EVIDENCE_PACK.md](EVIDENCE_PACK.md)。
> 过程细节：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md) · 全文日志：[WORK_LOG.md](WORK_LOG.md)
> 更新：2026-08-20 · **不新跑实验**，只用已落盘结果

---

## 0. 一句话主张

CIFAR-10 ResNet-18 正式基线下，iterative vs search 是 **regime-dependent**（低–中压缩 iterative 略稳，高压缩 ≥8x search 更强）——**不是**「search 全面更优」。LLM 侧：GLUE 作过渡、SQuAD 作难压局限附录；**不写**「剪枝优于 dense」。

---

## 1. 主表（CIFAR formal100）— 必问

**设定**：ResNet-18 / CIFAR-10；基线 `checkpoints/cifar_resnet18_baseline_formal100.pth`（dense test **90.36%**）；seeds 42/43/44；产物 `results/cifar_p12_comparison_gpu_formal100_full/`。证据索引：[EVIDENCE_PACK.md](EVIDENCE_PACK.md) §4.5 · [PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) §3。

| 目标 | iterative test | search test | 读法 |
|------|----------------|-------------|------|
| 1.5x | 90.52±0.30 | 90.64±0.94 | 接近 |
| 2.0x | **90.66±0.12** | 90.31±0.52 | iterative 略优 |
| 4.0x | **89.85±0.16** | 89.37±0.16 | iterative 略优 |
| 6.0x | 88.22±0.38 | 88.25±0.56 | 持平 |
| 8.0x | 86.67±0.16 | **87.40±0.97** | search 略优 |
| 10.0x | 83.76±0.43 | **86.03±0.33** | search 更优（约 +2.3） |

**口述**：存在压缩率 crossover → **regime-dependent**。oneshot ≥4x 仍崩溃（负结果保留）。

**支撑消融（可一句带过）**：10x path/gate → `path_and_gate`；8x 稳健性 → 无翻转。详见 EVIDENCE_PACK §4.6–4.7。

---

## 2. LLM 附录（导师已确认：先 GLUE 再 SQuAD）

### 2.1 GLUE KG.6（过渡）

**产物**：`/mnt/data2/results/qwen_glue_kg6/kg6_summary.json`
**设定**：Qwen2.5-1.5B；SST-2 / RTE / QNLI；1.5x、2.0x；budget-aligned；**oneshot 已开恢复**；小样本帽；单 seed。
**表读法**：dense 固定 **1.00x（baseline, no prune）**；剪枝三方法按 `cell_target` 对齐压缩。**禁止**写「四方法均压缩到 Xx」。

| Task | Target | Dense Acc (1.00x) | Oneshot | Iterative | Search |
|------|--------|-------------------|---------|-----------|--------|
| SST-2 | 1.5x | 85.94 | 45.31 | 54.69 | 54.69 |
| SST-2 | 2.0x | 85.94 | 52.34 | 45.31 | 42.19 |
| RTE | 1.5x | 85.94 | 47.66 | 48.44 | 48.44 |
| RTE | 2.0x | 85.94 | 50.00 | 50.00 | 50.00 |
| QNLI | 1.5x | 81.25 | 53.12 | 46.88 | 53.12 |
| QNLI | 2.0x | 81.25 | 53.12 | 53.12 | 53.91 |

**口述**：dense 明显好于剪枝（分类任务上可保住精度）；剪枝三方法互有胜负，**无** search 全面更优。

### 2.2 SQuAD LoRA formal（难压局限）

**产物**：`/mnt/data2/results/qwen_k6_lora_formal/k6_lora_formal_summary.json`
**设定**：1.5x / 2.0x × dense + 三剪枝；剪枝后 8192×2 LoRA；评测 n=256 + frozen test；单 seed。
**预期**：生成式 QA **难压**；数字是局限证据，不是主文胜利表。

| Target | Method | Actual | Val F1 | Frozen F1 | 备注 |
|--------|--------|--------|--------|-----------|------|
| — | dense | **1.00x** | 23.98 | 15.17 | baseline，**无 LoRA** |
| 1.5x | oneshot | 1.50x | 31.48 | 25.87 | |
| 1.5x | iterative | 1.50x | 32.15 | 27.74 | |
| 1.5x | search | 1.50x | 32.60 | 24.26 | |
| 2.0x | oneshot | 2.00x | 26.73 | 19.58 | |
| 2.0x | iterative | 2.00x | 28.58 | 24.49 | |
| 2.0x | search | 2.00x | 21.34 | 16.56 | |

**口述注意**：

- 1.5x 三方法接近；2.0x 上 iterative ≥ oneshot > search。
- **pruned Val F1 > dense** 来自 **恢复不对称**（dense 无 LoRA；剪枝臂有 LoRA），**不是**「剪枝优于 dense」。公平对照只比剪枝三方法；分类侧见 GLUE dense ≫ pruned。
- 早期弱恢复曾出现 F1≈0（协议问题，已修）；oneshot 无恢复会崩——见「已知问题」。

---

## 3. RQ / 路线口径（与导师口头一致）

| 项 | 状态 |
|----|------|
| 主文 | CIFAR regime-dependent |
| LLM | GLUE 过渡 → SQuAD 难压附录；非 LLM 主表 |
| **RQ4**（External vs Self controller） | **必做 · 下一档**（未跑；优先 GLUE smoke） |
| **RQ3**（High-Gap / E9） | **RQ4 之后做**（不插队；非取消） |
| 硬件 | 默认 **1×4090**；仅可拆 ≥2 cell 且墙钟很长时再加第 2 卡 |

---

## 4. 已知问题 / 负结果（如实，P1）

| 项 | 说明 |
|----|------|
| CIFAR oneshot ≥4x | 正式表下仍崩溃（负结果保留） |
| SQuAD 弱恢复 | 早期 F1≈0；属恢复协议不足，非方法定论；formal 已换 8192×2 |
| SQuAD dense vs pruned | dense **无** LoRA；pruned>dense **不可**当优势主张 |
| formal summary 覆盖 | 双卡拆跑时后写 job 曾覆盖 summary；已合并为完整 8 行 JSON |
| LLM 单 seed / 小样本帽 | 附录级证据；不可写成大规模稳健结论 |
| iterative 超剪 | 曾出现相对目标超剪；预算对齐后 KG.6 / formal 已按实际压缩对齐 |

---

## 5. 问题分级与上报（交付 + 日常）

| 级别 | 例子 | 动作 |
|------|------|------|
| **P0** 立刻告诉你 | 主表数字对不上产物；实验无法复现；口径与导师口头冲突；要改 RQ 优先级 / 加卡 / 换模型 | **当轮聊天明确说** + 写入下方「待拍板」 |
| **P1** 如实记录 | 负结果、协议不对称、单 seed、summary 覆盖、超剪已修 | WORK_LOG + 本节「已知问题」 |
| **P2** 内部债 | 测试/文档小修、emoji、路径整理 | 日志即可 |

---

## 6. 需导师拍板

**已拍板（2026-08-20）**

| # | 问题 | 结论 |
|---|------|------|
| 2 | RQ4 是否下一实验必做？ | **做** — 下一档必做（优先 GLUE smoke） |
| 3 | RQ3 / E9 是否投稿前都不做？ | **否** — 改为 **RQ4 做完后再做 RQ3**（顺序锁定，非永久砍掉） |

**仍待拍板**

1. 本期论文是否确认 **主文只押 CIFAR**，LLM 仅过渡 + 难压局限附录？

（新拍板同步本节 + WORK_LOG_BRIEF。）

---

## 7. 产物路径速查

| 内容 | 路径 |
|------|------|
| CIFAR formal100 | `results/cifar_p12_comparison_gpu_formal100_full/` |
| GLUE KG.6 | `/mnt/data2/results/qwen_glue_kg6/kg6_summary.json` |
| SQuAD LoRA formal | `/mnt/data2/results/qwen_k6_lora_formal/k6_lora_formal_summary.json` |
| 证据总索引 | `docs/EVIDENCE_PACK.md` |
