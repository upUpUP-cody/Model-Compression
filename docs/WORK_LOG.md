# 工作日志（详细）

> 用途：记录做了什么、关键结论与创新点。供论文/答辩/交接使用。
> 对应精简版：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md)
> 日期：2026-08-13（2.4–2.6：压缩比 / sweep / 搜索门禁，2026-08-14）
> 主机：RTX 4090 Linux，torch 2.13.0+cu130
> 代码：`8b61c86`（搜索门禁：recovery 后再 decide）；sweep/formal 结果在 `results/`（不进 Git）

---

## 1. 研究目标（一句话）

在 **train / validation / test 严格隔离** 的前提下，用 **Wanda 驱动的物理结构化剪枝 + 自主搜索**，在 MNIST MLP 上完成可复现对照，再把同一协议迁移到 **CIFAR-10 + ResNet-18**。

核心问题不是“再压一点准确率”，而是：

1. 剪枝是否真的按 Wanda 索引做 **物理结构缩小**（不是掩码稀疏）
2. 选择是否只看 validation，test 是否只在冻结后出现一次
3. 自主搜索相对 one-shot / 迭代剪枝，在压缩率升高时是否更稳

---

## 2. 已完成工作（按时间）

### 2.1 P1.2 MNIST GPU 统计补全（已完成）

**做了什么**

- 修复 `dense_small`：小 MLP 必须在 **train split 上从零训练**，不能用随机初始化当对照
- 新增多 seed runner、压缩率扫描、mean/std 聚合
- 在 RTX 4090 上跑完：smoke、formal、3 seed、6 档压缩率 × 3 seed

**产物**

| 实验 | 目录 |
|------|------|
| MNIST sweep 聚合 | `results/p12_comparison_gpu_sweep/` |
| 报告 | `results/p12_comparison_gpu_sweep/AGGREGATE_REPORT.md` |

**重要结论（MNIST）**

1. **带恢复的迭代剪枝 / 自主搜索** 在 1.5x–10x 压缩下，test 准确率仍约 **97–98%**
2. **one-shot（无恢复）** 超过约 2x 后崩溃（4x 已接近随机），这是预期，不是实现错误
3. 修复后的 **dense_small** 稳定在约 **96%**，不再是 ~13% 的“未训练对照”
4. 自主搜索的价值主要体现在 **高压缩率下仍接近 baseline**，而不是在 1.5x 时超过 dense

这构成论文里“方法有效”的第一块证据：协议正确 + 正负结果都保留。

### 2.2 P2 代码交付：CIFAR-10 + ResNet（已实现，协议刚开始跑）

**做了什么（代码，不跑长实验也算交付）**

| 模块 | 内容 |
|------|------|
| 数据 | CIFAR train/val split + `split_hash`；test 封存 |
| 模型 | 项目内 ResNet-18（32×32，10 类）；`dense_small` 用 `base_width=32` |
| 剪枝 | **只剪 BasicBlock 内部 conv1 通道**，同步 BN 与 conv2 输入；残差 I/O 宽度不变 |
| 搜索 | `MlpBackend` / `CnnBackend`；默认仍 MLP，`model.type: resnet_cifar` 走 CNN |
| 恢复 | Level 1 微调；Level 2 LoRA；Level 3 自蒸馏（接口已写，消融未跑） |
| 协议 | CIFAR P1.2 runner，输出与 MNIST 目录分离 |

**创新 / 设计要点（相对“直接套 MLP 剪枝”）**

1. **显式 CNN 依赖图，禁止用 MLP 层名枚举套 ResNet**
   残差块若剪错输出通道，shortcut 对不齐，forward 直接炸。我们选择剪 **块内中间宽度**，保证残差两端维度不变。这是工程正确性，也是方法可迁移的前提。

2. **同一套六方法对照协议从 MLP 迁到 CNN**
   dense / dense_small / oneshot magnitude / oneshot Wanda / 迭代 Level1 / autonomous search。论文叙事可以是“协议不变，架构升级”，而不是另起一套实验。

3. **Wanda 对 Conv 的定义**
   通道分数 = 卷积核 L2 × 该通道平均激活幅度（对 NCHW 在 N,H,W 上聚合）。与 Linear 的 Wanda 同构，便于写进方法章节。

4. **恢复分级预留**
   Level 1 全参数微调、Level 2 LoRA、Level 3 师生蒸馏，准备做消融，而不是只报一个“finetune 一下”。

### 2.3 CIFAR 基线 + P1.2 Smoke（2026-08-13 已跑通）

**前置坑**

- torchvision 默认从多伦多大学下载 CIFAR，国内极慢；后改为本机下载 + rsync，MD5 `c58f30108f718f92721af3b95e74349a` 校验通过

**基线**

- 配置：`configs/cifar_resnet_baseline_gpu.yaml`，**20 epoch smoke 基线**（不是 100+ epoch 正式基线）
- Checkpoint：`checkpoints/cifar_resnet18_baseline.pth`
- 参数量：11,173,962
- split：train 45000 / val 5000，`split_hash=89c06fb2...`
- 最佳 **validation 88.86%**（epoch 16）；耗时约 **200 s**

**Smoke study**

- 目录：`results/cifar_p12_comparison_gpu_smoke/cifar_p12_comparison_cifar_p12_gpu_smoke_a145fe93fd29_1`
- 六方法冻结 + `report-test`（官方 test 10000，仅此一次）
- GPU 指标：`peak_allocated_bytes=338333184`，吞吐约 9723 samples/s

**Smoke 数字（20 epoch 基线 + recovery_epochs=0）**

| 方法 | Val Acc | Test Acc |
|------|---------|----------|
| dense_baseline | 88.86 | 87.88 |
| dense_small | 10.30 | 10.04 |
| oneshot_magnitude | 88.34 | 87.11 |
| oneshot_wanda | 88.08 | 87.07 |
| iterative_structured_level1 | 87.44 | 86.39 |
| autonomous_search | 87.90 | 86.85 |

**Smoke 结论（必须写清楚边界）**

1. **协议通了**：train/val 选择、test 冻结后评估、manifest 含 split 与 GPU benchmark。这是 P2.4 的门禁，不是论文最终数字。
2. **轻剪几乎不掉点**：只剪少量 conv1 通道、且 **不做恢复** 时，oneshot / 搜索相对 baseline 只掉约 0.8–1.5 个点。说明结构剪枝路径正确，权重能 load 进缩小后的网络。
3. **dense_small ≈ 10% 不是 bug**：smoke 里 `recovery_epochs: 0`，小 ResNet **未训练**。MNIST 上同类问题已用“从零训练”修好；CIFAR formal 必须打开 recovery/训练 epoch，否则不能当对照。
4. **不要用这次 smoke 比较方法优劣**：压缩极轻、无恢复、基线只有 20 epoch。方法对比要等 formal / multiseed / sweep。

**跑实验时修的两个实现问题**

1. `report-test` 原先总是按完整 ResNet-18 `load_state_dict`，剪枝后通道对不上。现按 `layer_keep_indices` 重建结构再加载；`dense_small` 走小宽度模型。
2. GPU benchmark 把 `batch_size` 传了两次，smoke 第一次在写 manifest 时崩溃；去掉重复参数后重跑成功。

### 2.4 CIFAR Formal + 3-seed Multiseed（2026-08-13 已跑通）

**实测时长（比事前估计更短）**

| 实验 | 墙钟 |
|------|------|
| Formal study | 134 s |
| Formal report-test | 20 s |
| 3-seed multiseed（含每个 seed 的 report-test + 聚合） | 229 s |

**Formal（1 seed=42，recovery 3 epoch）**

目录：`results/cifar_p12_comparison_gpu/cifar_p12_comparison_cifar_p12_gpu_study_12815300d99f`

| 方法 | Val Acc | Test Acc | 参数量 |
|------|---------|----------|--------|
| dense_baseline | 88.86 | 87.88 | 11,173,962 |
| dense_small | 69.08 | 68.57 | 2,797,610 |
| oneshot_magnitude | 84.80 | 83.47 | 11,083,412 |
| oneshot_wanda | 81.92 | 81.08 | 11,083,412 |
| iterative_structured_level1 | 89.10 | 88.38 | 11,113,980 |
| autonomous_search | 89.66 | 88.06 | 11,150,882 |

**Multiseed test acc mean ± std（seeds 42/43/44）**

目录：`results/cifar_p12_comparison_gpu_multiseed/aggregate/aggregate_summary.json`

| 方法 | Test mean | Test std |
|------|-----------|----------|
| dense_baseline | 87.88 | 0.00 |
| dense_small | 66.87 | 0.32 |
| oneshot_magnitude | 87.11 | 0.00 |
| oneshot_wanda | 87.07 | 0.00 |
| iterative_structured_level1 | 87.15 | 0.35 |
| autonomous_search | 87.88 | 0.14 |

**结论**

1. **dense_small 已从 10% 拉到约 67%**：3 个 epoch 从零训练有效，但仍明显低于 20 epoch 的大模型，符合预期。
2. **Level 1 恢复有效**：formal 上 iterative / search 的 test 回到 baseline 附近甚至略高；oneshot 无恢复则掉 4–7 个点。
3. **实际压缩几乎为 1.00x**：只剪 BasicBlock 内 conv1，YAML `target_compression_ratio: 2.0` 没有真正达到。oneshot 跨 seed 方差为 0，因为同一 checkpoint + 同一比例是确定性的。
4. 在压缩几乎为 0 时，**不能**用这组数字声称自主搜索优于迭代剪枝。要先让剪枝能打到目标压缩比，再跑 sweep。

### 2.4 CNN 目标压缩比（2026-08-14 已验证）

**先前 1.00x 的原因**：formal/multiseed YAML 只手写了 1–2 个 `layer*.conv1: 0.3`，不是 GPU 慢，也不是 conv1-only 剪不动。

**conv1-only 实测上限**（ResNet-18 CIFAR `base_width=64`，均匀剪全部 8 个块内 conv1）：

| 均匀剪枝比例 | 实际压缩比 |
|--------------|------------|
| 0.30 | 1.42x |
| 0.50 | 1.97x |
| 0.80 | 4.70x |
| 0.95 | 15.10x |

2x / 4x / 10x **都能达到**，因此没有扩展到块输出通道 / shortcut。

**配置**：`apply_compression_target` / `ensure_compression_target` 对全部 prunable conv1 做均匀比例二分；formal / multiseed / sweep / smoke 不再手写单层 0.3。CNN 搜索会先提出「全部 conv1 同一比例」候选，否则 `candidates_per_round=1` 时仍会只剪一层。

**2x smoke 验证**（recovery 0 epoch，复用 `checkpoints/cifar_resnet18_baseline.pth`）

目录：`results/cifar_p12_comparison_gpu_smoke/cifar_p12_comparison_cifar_p12_gpu_smoke_866113f8e78a`

| 方法 | 实际压缩 | Val Acc | Test Acc |
|------|----------|---------|----------|
| dense_baseline | 1.00x | 88.86 | 87.88 |
| dense_small | 3.99x（固定小宽度，未训练） | 10.30 | 10.04 |
| oneshot_magnitude | **2.01x** | 46.40 | 45.80 |
| oneshot_wanda | **2.01x** | 22.94 | 23.52 |
| iterative_structured_level1 | **2.01x** | 46.40 | 45.80 |
| autonomous_search | **2.01x** | 23.12 | 23.62 |

剪枝臂压缩落在 1.7–2.3 门禁内。无恢复时准确率大幅下降是预期，不能用来比较方法优劣。

### 2.5 CIFAR 6×3 压缩率 Sweep（2026-08-14 已跑通）

**命令**

```bash
./scripts/run_gpu.sh python experiments/run_cifar_p12_multiseed.py \
  --config configs/cifar_p12_gpu_sweep.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth --sweep
```

**产物**

| 项 | 路径 |
|----|------|
| 根目录 | `results/cifar_p12_comparison_gpu_sweep/` |
| 聚合 JSON | `results/cifar_p12_comparison_gpu_sweep/aggregate/aggregate_summary.json` |
| 报告 | `results/cifar_p12_comparison_gpu_sweep/AGGREGATE_REPORT.md` |

墙钟约 **22 分钟**（18 study）。`recovery_epochs: 2`。

**实际压缩（3 seed 均值）**

| 目标 | oneshot / iterative | autonomous_search |
|------|---------------------|-------------------|
| 1.5x–10x | 全部落在目标 ±15% 内（例：2.01x、4.03x、10.16x） | **始终 1.00x** |

**Test acc mean±std（摘录）**

| 目标 | oneshot_mag | oneshot_wanda | iterative | search |
|------|-------------|---------------|-----------|--------|
| 1.5x | 69.12±0.00 | 69.96±0.89 | **87.73±0.69** | 87.88±0.00 |
| 2.0x | 45.80±0.00 | 22.23±1.21 | **87.43±0.33** | 87.88±0.00 |
| 4.0x | 17.48±0.00 | 12.14±0.17 | **86.52±0.64** | 87.88±0.00 |
| 10.0x | 10.53±0.00 | 11.33±0.21 | **83.42±0.39** | 87.88±0.00 |

**结论**

1. **oneshot / iterative 已真正打到目标压缩比**；不再是旧 formal 的 ~1.00x。
2. **one-shot 无恢复在 ≥2x 后崩溃**（负结果保留，与 MNIST 同构）。
3. **迭代 Level-1 恢复是当前 CIFAR 主证据**：10x 仍约 83.4% test（baseline 87.88）。
4. **自主搜索未压缩**：Cheap Critic + `max_accuracy_drop_points: 2.0` 在恢复前因 `capability_gap_exceeded` 拒绝全层候选；search 的 87.88% 只是 baseline，**不能**声称搜索优于迭代。
5. 旧 ~1.00x formal/multiseed **不得**与本 sweep 混写。

### 2.6 修复 CNN 搜索门禁（2026-08-14）

**根因**：Cheap Critic 在恢复前用 `max_accuracy_drop_points` 硬拒；全层 2x 剪枝后 Critic 约 20%，触发 `capability_gap_exceeded`，恢复从未执行。

**修复**：[`src/autonomous_search.py`](../src/autonomous_search.py) — Critic 只做短名单排序；对 `recovery_top_k` 先 recovery，再用恢复后 validation 做 `decide_action`。最终 2 点门禁不变。

**2x formal 短验证**（`max_accuracy_drop_points: 2.0`，recovery 3 epoch）

目录：`results/cifar_p12_comparison_gpu/cifar_p12_comparison_cifar_p12_gpu_study_514cff755ed7`

| 方法 | 实际压缩 | Val Acc | Test Acc |
|------|----------|---------|----------|
| dense_baseline | 1.00x | 88.86 | 87.88 |
| dense_small | 3.99x | 69.08 | 68.57 |
| oneshot_magnitude | 2.01x | 46.40 | 45.80 |
| oneshot_wanda | 2.01x | 22.92 | 23.49 |
| iterative_structured_level1 | 2.01x | 88.98 | 88.01 |
| autonomous_search | **7.66x** | 85.92 | 85.13 |

Search 三轮均 `accept`：Cheap Critic 21.9% / 14.8% / 8.2%，恢复后 val 88.12 → 87.66 → 85.92。门禁 `compression_ratio > 1.05` 通过。同目标 2x 下 iterative 压缩更保守（停在 2.01x）但 test 略高；search 在多轮接受后压到 7.66x，test 85.13。旧 sweep 的 search=1.00x 结论作废。

---

## 3. 创新点（写论文时可用的表述）

按“能写进 related work 对比”的粒度，而不是营销口号：

1. **自主搜索 + 物理结构化剪枝，而不是幅值掩码**
   候选用 Wanda 索引真正改 Linear/Conv 形状，参数量下降可测、可部署。

2. **可审计搜索**
   每个候选有 fingerprint、Cheap Critic、淘汰原因、rollback 快照；不是只报最终模型。

3. **Pareto 用 validation accuracy vs 真实参数量**
   不用 `acc / compression` 这种无量纲伪指标。

4. **严格三路数据协议**
   选择只看 validation；test 只在 manifest 冻结后评估一次。这是可复现声明的核心。

5. **CNN 迁移的结构约束**
   ResNet 剪枝单位是块内中间通道，残差对齐作为硬约束，而不是事后 pad。

6. **对照完整**
   同一预算下比较 dense、同规模 dense_small、one-shot、迭代恢复、自主搜索；保留 one-shot 在高压缩失败的负结果。

当前 **还不能声称** 的（避免写过头）：

- CIFAR 上自主搜索已经系统优于迭代剪枝（单次 formal：search 7.66x / 85.13% test，iterative 2.01x / 88.01%；需同压缩预算对照后再下结论）
- LoRA / 自蒸馏已经有效（只实现了接口）
- 达到论文级 CIFAR 精度（20 epoch 基线约 88%，正式基线通常要 100+ epoch）

---

## 4. 下一步（按优先级）

1. 恢复消融 Level 1/2/3（迭代路径已证明 Level 1 有效）
2. 同压缩预算下重跑 search vs iterative 对照（或带修复后的 6×3 sweep）
3. Qwen/SQuAD：**规划占位，不实现**

---

## 5. 不进 Git 的东西

`results/`、`data/`、`checkpoints/`、`venv/` 默认不提交。日志本文在 `docs/`，记录结论；原始数字以 study 目录里的 JSON 为准。
