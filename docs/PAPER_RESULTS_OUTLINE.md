# 论文成果提纲（视觉域收成）

> 用途：投稿前方法/实验/讨论的**成果收成稿**（提纲级，非全文）。
> 证据索引：[EVIDENCE_PACK.md](EVIDENCE_PACK.md) · 过程日志：[WORK_LOG.md](WORK_LOG.md) · 精简：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md)
> 主表协议：CIFAR-10 + ResNet-18，**100 epoch 正式基线**；seeds 42/43/44；train/val/test 隔离。
> 日期：2026-08-15 · **不**把 20 epoch smoke 与正式表混写

---

## 1. 一句话贡献

在严格数据隔离下，实现 **Wanda 驱动的物理结构化剪枝 + 可审计自主搜索**，并在 CIFAR 上相对迭代剪枝给出 **regime-dependent** 结论：低–中压缩 iterative 略稳，高压缩（≥8x）同预算下 search 更强——**不是**「search 系统全面优于 iterative」。

---

## 2. 方法要点

1. **物理结构化剪枝**：按 Wanda 索引真正改 Linear / 块内 `conv1` 形状，参数量可测；非幅值掩码稀疏。
2. **三路数据协议**：选择与门禁只看 validation；manifest 冻结后 test **评估一次**。
3. **可审计搜索**：候选 fingerprint、Cheap Critic、淘汰原因、rollback；Pareto 用 val accuracy vs 真实参数量。
4. **控制器与压缩边界**：默认 **2 点**能力门禁；增量逼近（`max_step_compression`）；过冲硬顶（`>target×1.15` 拒绝）；达目标止损。
5. **CNN 约束**：只剪 BasicBlock 内中间通道，残差 I/O 宽度不变。一步到目标策略仅保留约 **`target≤2`**（4x 硬一步会因缺中间恢复变差）。

---

## 3. 主结果表（formal100_full）

**设定**：`checkpoints/cifar_resnet18_baseline_formal100.pth`（best val 91.84%，dense test **90.36%**）；产物 `results/cifar_p12_comparison_gpu_formal100_full/`；≥4x search 压缩均在目标 ±15%。

| 目标 | iterative test | search test | 读法 |
|------|----------------|-------------|------|
| 1.5x | 90.52±0.30 | 90.64±0.94 | 接近 |
| 2.0x | **90.66±0.12** | 90.31±0.52 | iterative 略优 |
| 4.0x | **89.85±0.16** | 89.37±0.16 | iterative 略优 |
| 6.0x | 88.22±0.38 | 88.25±0.56 | 持平 |
| 8.0x | 86.67±0.16 | **87.40±0.97** | search 略优 |
| 10.0x | 83.76±0.43 | **86.03±0.33** | search 更优（约 +2.3） |

**主主张**：存在 **压缩率 crossover** → 写作 **regime-dependent**。oneshot ≥4x 仍崩溃（负结果保留）。

---

## 4. 机制与稳健性

### 4.1 Path / Gate 消融（同 L1、2 epoch 恢复）

| 档 | path (no-gate−uniform) | gate (gated−no-gate) | verdict |
|----|------------------------|----------------------|---------|
| 4x | ~+0.2 | ~−0.1 | `inconclusive_close` |
| 8x | ~+0.2 | ~+1.4 | `gate_dominant` |
| 10x | ~+2.1 | ~+1.3 | `path_and_gate` |

- **10x**：uniform 82.66 → no-gate 84.78 → gated **86.03**（路径与门禁都贡献）
- **8x**：门禁主导；确认非 10x 特例
- **4x**：三臂几乎重合；search 与 iterative **最终层宽可相同**，~0.5 点差距来自恢复轨迹

### 4.2 预算对齐（matched-up iterative = 6 epoch）

| 目标 | iterative | search | delta (it−se) |
|------|-----------|--------|---------------|
| 2x | 90.56±0.07 | 90.18±0.43 | +0.38 |
| 4x | 90.04±0.18 | 89.43±0.32 | +0.61 |

抬高 iterative 预算后仍 ≥ search → 低压缩差距 **不是**「iterative 恢复不够」。

### 4.3 一步到目标

| 目标 | 现象 |
|------|------|
| 2x | search≈iterative（差距可抹平） |
| 4x | 一步直接 ~4.02x；最终层宽与多步/iterative **相同**，但 test 变差 → **缺中间恢复** |
| ≥8x | search 仍高于 iterative；高压缩优势未明显丢失 |

**跟进**：代码一步策略收窄为仅 `target≤2`。

---

## 5. 辅助结果

| 线 | 要点 |
|----|------|
| **MNIST MLP** | 协议可复现；iterative/search+恢复在 1.5x–10x 仍约 97–98%；oneshot 无恢复高压缩崩溃 |
| **CIFAR 恢复消融**（多 seed，2.01x Wanda） | L1 test 87.25±0.82；L2 89.22±0.09（含适配器参数）；L3 88.51±0.28。主路径仍以 Level-1 为准；**不得**写 L2/L3 无条件更优 |

---

## 6. 负结果与作废（必须保留）

| 项 | 写法 |
|----|------|
| oneshot 高压缩崩溃 | 负结果，说明恢复必要 |
| 历史 search=1.00x / 过冲 7.66x / ≥4x 欠压 | **作废或不作主对照**；已用门禁顺序、止损、增量逼近、过冲硬顶修复 |
| smoke（20 epoch）vs formal（100 epoch） | **分列**；主表只用 formal |
| SQuAD 短/中 SGD 恢复后 F1≈0 | **负对照保留**：弱恢复撑不起生成式 QA |
| SQuAD LoRA Informal 中 dense F1 < pruned | **不解释为剪枝更好**；n=64 噪声 + 任务适配 |

---

## 7. 能写 / 不能写

### 能写

- 物理结构化剪枝 + 可审计自主搜索 + 严格三路协议
- 同压缩预算下 CIFAR **regime-dependent** crossover
- 高压缩优势可拆为增量路径与 2pt 门禁；贡献随压缩率变化
- 4x 同结构下差距来自恢复轨迹；硬一步会因缺中间恢复变差
- SQuAD：弱恢复失败（负对照）+ LoRA@1.5x Informal 表（附录）；本格 iterative ≥ oneshot ≈ search

### 不能写

- [×] search **系统全面**优于 iterative（含 LLM Informal）
- [×] LoRA / 自蒸馏无条件优于 Level-1（CIFAR）；但 LLM 侧可写「对齐文献的 LoRA 恢复使 SQuAD F1 可读」
- [×] smoke 与 formal 混为同一主表
- [×] 用历史欠压/过冲 search 的高分与高压缩 iterative 混比刷优势
- [×] 把未同协议、未同压缩预算对齐的外部论文数字直接当「我们更好」的证据
- [×] 把 n=64 SQuAD Informal 升级为论文 LLM 主表 / regime 复现

---

## 8. 外部压缩 baseline / 自动搜索 vs 人工设计

> 导师要求：调研其他模型压缩工作，以其效果作 baseline，检验 **自动搜索是否优于人工设计**。
> 执行挂接：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md) K6-lit；总路线：[PROJECT_PLAN.md](../PROJECT_PLAN.md)。

**研究问题**：在同压缩预算（及尽可能同恢复预算）下，`autonomous_search` 是否优于人工设计的压缩方案（含外部方法与内部固定策略）。

| 类别 | 对应 | 角色 |
|------|------|------|
| 人工设计（内部已有） | `oneshot` / `iterative` | 主表已有；保留为协议内人工基线 |
| 自动搜索 | `autonomous_search` | 待与人工设计公平对照 |
| 人工设计（外部） | 结构化剪枝 / LLM 剪枝代表工作 | Related Work 调研；能复现则进 Experiments |

### 文献筛选（质量门禁）

调研优先选 **高质量、有一定权威性** 的论文；边缘/不可复现预印本不当作主 baseline。进实验复现短名单须至少满足下列前 3 条中的 **2 条**：

1. **顶会/顶刊或高引用奠基工作**：ICLR / NeurIPS / ICML / ACL / EMNLP / CVPR 等，或领域公认奠基（如 LTH）
2. **方法被广泛引用或官方实现可核**：有开源、复现报告或多篇后续引用
3. **与本课题可比**：结构化剪枝 / LLM 剪枝 / 彩票或迭代剪枝；能陈述压缩率与任务
4. **可对齐或可声明差异**：同压缩带、恢复预算可说清；否则只进 Related Work，不进主表

**排除**：无出处博客数字、无法核验的私有结果、与物理结构化剪枝完全不可比却硬比的数字。

**默认调研池**（从中勾选 2–4 个入短名单；其余 Related Work 一句带过）：

| 方向 | 代表工作 | 权威性依据（简述） |
|------|----------|-------------------|
| 奠基 | Lottery Ticket Hypothesis (Frankle & Carbin) | ICLR；领域奠基 |
| 幅值/激活剪枝 | Wanda (Sun et al.) | 高引用 LLM 剪枝；项目已用 |
| 一刀剪枝 | SparseGPT (Frantar & Alistarh) | ICML；LLM 一刀剪枝代表 |
| 结构化 LLM | LLM-Pruner / SliceGPT 等顶会结构化路线 | 与「物理改形状」更可比 |
| 迭代/人工设计对照 | 经典 IMP / 均匀结构化 iterative | 对应「人工设计」一侧 |

### 做法

1. 按质量门禁从默认调研池勾选 2–4 个工作，整理短表（venue/依据、方法、设定、指标、是否可复现）。
2. 优先选与本设定可对齐者：同模型或明确声明差异；同压缩带；恢复预算可陈述。
3. 能在本协议下复现 → CIFAR 主表之后的 External baselines 小节，或 Phase K 的 `/mnt/data2/results/qwen_glue_*`（优先）/ `qwen_*`（SQuAD）同协议对照。
4. 不能公平复现 → 只进 Related Work 文献对比，**不进主表、不进主主张**。

**约束**：外部 baseline 是补强对照，**不替换**内部 iterative；叙事仍为 regime-dependent，禁止未证成的「search 系统全面更优」。LLM 侧：**KG.5 GLUE** = 过渡/门禁；**SQuAD** = 弱恢复负对照 + LoRA@1.5x Informal 附录（非主表）；结论见 WORK_LOG §4.17。

---

## 9. 建议论文章节映射

| 章节 | 放什么 |
|------|--------|
| Introduction | 问题：同预算下结构化剪枝路径；贡献 1–3 条 + regime-dependent 预告；预告自动搜索 vs 人工设计 |
| Related Work | **优先权威谱系**（顶会/高引用奠基），不堆低质量引用；人工设计 vs 搜索式压缩；外部 baseline 候选短表 |
| Method | §2 方法要点；协议图（train→prune/recover→val select→freeze→test once） |
| Experiments | MNIST 协议正确性 → CIFAR formal100 主表（§3）→ **External baselines（若复现）** |
| Ablations | §4 机制表 + 预算对齐 + 一步诊断；恢复 L1/2/3 |
| Discussion | 为何必须同压缩；门禁科学含义；负结果；smoke≠formal；自动 vs 人工的边界 |
| LLM / 迁移 | KG.5 GLUE 过渡 `[√]`；K6 预算对齐 `[√]`；**SGD 加深 = 负对照（F1≈0）**；**LoRA@1.5x Informal 可读**（iterative 38.9 / oneshot 30.6 / search 30.0；n=64）→ 附录/讨论，**不写 LLM regime 主复现**；K6-lit 短表已有 |
| Limitations / Future | 弱恢复下生成式 QA 极脆；MLP-only 名义 4x 不可达；Informal≠正式主表；外部 baseline 先文献 |

---

## 10. 主产物路径（写论文时引用）

| 用途 | 路径 |
|------|------|
| 正式基线 | `checkpoints/cifar_resnet18_baseline_formal100.pth` |
| 主表 | `results/cifar_p12_comparison_gpu_formal100_full/` |
| 10x / 8x / 4x 机制消融 | `results/cifar_crossover_path_ablation{,_8x,_4x}/` |
| 预算对齐 / 一步复验 | `results/cifar_p12_budget_match_key/`、`results/cifar_p12_lowcomp_step_key/` |
| 恢复消融（多 seed） | `results/cifar_recovery_ablation_multiseed/` |
| MNIST 主 sweep | `results/p12_comparison_gpu_sweep/` |
| KG.5 GLUE 过渡（非正式） | `/mnt/data2/results/qwen_glue_kg5/` |
| K6 SQuAD 小扫（短恢复负对照） | `/mnt/data2/results/qwen_k6/` |
| K6 加深 SGD 1.5x（负对照） | `/mnt/data2/results/qwen_k6_recover_1p5x/` |
| K6 LoRA Informal 1.5x | `/mnt/data2/results/qwen_k6_recover_lora_1p5x/`、`..._methods/` |
| K6-lit 短表 | `docs/K6_LIT_BASELINE_SHORTLIST.md` |

详细索引与复现命令见 [EVIDENCE_PACK.md](EVIDENCE_PACK.md) §8。
