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

---

## 7. 能写 / 不能写

### 能写

- 物理结构化剪枝 + 可审计自主搜索 + 严格三路协议
- 同压缩预算下 CIFAR **regime-dependent** crossover
- 高压缩优势可拆为增量路径与 2pt 门禁；贡献随压缩率变化
- 4x 同结构下差距来自恢复轨迹；硬一步会因缺中间恢复变差

### 不能写

- [×] search **系统全面**优于 iterative
- [×] LoRA / 自蒸馏无条件优于 Level-1
- [×] smoke 与 formal 混为同一主表
- [×] 用历史欠压/过冲 search 的高分与高压缩 iterative 混比刷优势

---

## 8. 建议论文章节映射

| 章节 | 放什么 |
|------|--------|
| Introduction | 问题：同预算下结构化剪枝路径；贡献 1–3 条 + regime-dependent 预告 |
| Method | §2 方法要点；协议图（train→prune/recover→val select→freeze→test once） |
| Experiments | MNIST 协议正确性 → CIFAR formal100 主表（§3） |
| Ablations | §4 机制表 + 预算对齐 + 一步诊断；恢复 L1/2/3 |
| Discussion | 为何必须同压缩；门禁科学含义；负结果；smoke≠formal |
| Limitations / Future | Transformer/Qwen 仅规划（[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md)）；实现前须扩盘 |

---

## 9. 主产物路径（写论文时引用）

| 用途 | 路径 |
|------|------|
| 正式基线 | `checkpoints/cifar_resnet18_baseline_formal100.pth` |
| 主表 | `results/cifar_p12_comparison_gpu_formal100_full/` |
| 10x / 8x / 4x 机制消融 | `results/cifar_crossover_path_ablation{,_8x,_4x}/` |
| 预算对齐 / 一步复验 | `results/cifar_p12_budget_match_key/`、`results/cifar_p12_lowcomp_step_key/` |
| 恢复消融（多 seed） | `results/cifar_recovery_ablation_multiseed/` |
| MNIST 主 sweep | `results/p12_comparison_gpu_sweep/` |

详细索引与复现命令见 [EVIDENCE_PACK.md](EVIDENCE_PACK.md) §8。
