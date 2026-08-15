# 工作日志（详细）

> 用途：记录做了什么、关键结论与创新点。供论文/答辩/交接使用。
> 对应精简版：[WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md)
> 日期：2026-08-13（2.4–2.8 + Phase H 证据包：2026-08-14）
> 主机：RTX 4090 Linux，torch 2.13.0+cu130
> 代码：搜索门禁 + `target_compression_reached` 止损；sweep/formal/消融结果在 `results/`（不进 Git）

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

### 2.7 恢复消融 Level 1/2/3（2026-08-14）

**做了什么**

- 对齐 GPU 配置与 `target_compression_ratio: 2.0`（去掉手写单层 0.3）；runner 调用 `ensure_compression_target`
- 固定同一 Wanda 剪枝候选，对比 Level 1 全参微调 / Level 2 LoRA / Level 3 自蒸馏（各 3 epoch）
- 修复 LoRA：适配器与 base 同设备；AMP 下 LoRA 残差在 fp32 计算；stride 卷积与 base 对齐

**命令**

```bash
./scripts/run_gpu.sh python experiments/run_cifar_recovery_ablation.py \
  --config configs/cifar_recovery_ablation.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth
```

**产物**

| 项 | 路径 |
|----|------|
| 目录 | `results/cifar_recovery_ablation/` |
| JSON | `results/cifar_recovery_ablation/ablation_summary.json` |
| 报告 | `results/cifar_recovery_ablation/ABLATION_REPORT.md` |

墙钟约 **2.8 分钟**（含剪枝与三档恢复）。

**设定**：baseline val **88.86%**；剪枝后（恢复前）val **23.32%**；实际压缩 **2.01x**（11,173,962 → 5,564,062）。

| Level | 方法 | pruned 参数 | recovered 参数 | 恢复后 best val | 耗时 |
|------|------|-------------|----------------|-----------------|------|
| 1 | 全参微调 | 5,564,062 | 5,564,062 | **88.02%** | 32.3 s |
| 2 | LoRA | 5,564,062 | 5,578,464（含适配器） | **89.78%** | 48.9 s |
| 3 | 自蒸馏 | 5,564,062 | 5,564,062 | **89.16%** | 37.2 s |

**结论**

1. **压缩门禁通过**：2.01x 落在目标 2.0x ±15% 内；三档共用同一 `keep_indices`。
2. **三档接口均可在 GPU/fp16 跑通**；相对剪枝后 23.32%，均大幅拉回。
3. **本单次 run 中 Level 2/3 的 val 略高于 Level 1**（89.78 / 89.16 vs 88.02），且 Level 2/3 略高于 baseline 88.86%。这是 **单 seed、仅 validation、3 epoch** 的消融结果，**不能**据此宣称 LoRA/蒸馏系统优于全参微调。
4. Level 2 的 `recovered_params` 含 LoRA 适配器（约 +14k），报告部署体积时应区分 pruned 基础参数与适配器开销。
5. 迭代路径上 Level 1 仍是主证据；Level 2/3 已从「仅接口」升为「同候选可对比的一次消融」。

### 2.8 同压缩预算 sweep_v2（2026-08-14）

**问题**：门禁修复后的 2x formal 里 search 连 accept 三轮叠到 **7.66x**，与 iterative 的 **2.01x** 不可比。旧 sweep 的 search=1.00x 已作废。

**修复**：[`src/autonomous_search.py`](../src/autonomous_search.py) 增加 `target_compression_ratio`；`accept` 后若 `compression >= target` 则 `stop` / `target_compression_reached`。CIFAR/MNIST `_autonomous` 均传入目标比。

**2x 短验证**（`results/cifar_p12_comparison_gpu_stop_smoke/...`）：search 与 iterative 均为 **2.008x**，并出现 `target_compression_reached`（不再过冲到 7.66x）。

**命令**

```bash
./scripts/run_gpu.sh python experiments/run_cifar_p12_multiseed.py \
  --config configs/cifar_p12_gpu_sweep.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth --sweep
```

配置：`output_root=results/cifar_p12_comparison_gpu_sweep_v2`，6 档 × 3 seed，`recovery_epochs=2`。墙钟约 **29 分钟**。

**产物**

| 项 | 路径 |
|----|------|
| 根目录 | `results/cifar_p12_comparison_gpu_sweep_v2/` |
| 聚合 JSON | `results/cifar_p12_comparison_gpu_sweep_v2/aggregate/aggregate_summary.json` |
| 报告 | `results/cifar_p12_comparison_gpu_sweep_v2/AGGREGATE_REPORT.md` |

**实际压缩（3 seed 均值）**

| 目标 | oneshot / iterative | autonomous_search |
|------|---------------------|-------------------|
| 1.5x / 2.0x | 1.50x / 2.01x | **同档达标**（1.50x / 2.01x） |
| 4.0x | 4.03x | 均值 **3.02x**（seed42=4.03；43/44=2.51） |
| 6.0x–10.0x | 达标 | 常 **欠压**（部分 seed 停在 1.00x 或约 3x） |

**Test acc mean±std（摘录，同目标）**

| 目标 | oneshot_mag | oneshot_wanda | iterative | search |
|------|-------------|---------------|-----------|--------|
| 1.5x | 69.12±0.00 | 69.96±0.89 | **87.73±0.69** | 87.55±0.58 |
| 2.0x | 45.80±0.00 | 22.23±1.21 | **87.43±0.33** | 86.95±0.55 |
| 4.0x | 17.48±0.00 | 12.14±0.17 | **86.52±0.64** | 86.53±0.39（压缩未对齐） |
| 10.0x | 10.53±0.00 | 11.33±0.21 | **83.42±0.39** | 86.95±0.66（压缩未对齐） |

**结论**

1. **过冲已消除**：达标 accept 后会止损；1.5x/2.0x 上 search 与 iterative **同压缩**，可公平对照。
2. **同压缩（1.5x/2.0x）下 iterative 略优于或接近 search**（test 差约 0.2–0.5 点）；**不能**声称 search 系统优于 iterative。
3. **≥4x 时 search 常因 2 点能力门禁拒掉全层激进候选而欠压**；此时 search 的高 test 来自更小压缩，**不得**与 iterative 的高压缩数字直接比优劣。
4. 旧 sweep（search=1.00x）与 7.66x formal **不得**与本表混写。iterative 在真·高压缩上仍是主证据（10x test 约 83.4%）。

### 2.9 Phase H 证据包（2026-08-14）

交付 [`docs/EVIDENCE_PACK.md`](EVIDENCE_PACK.md)：协议图、MNIST/CIFAR 主表、消融、负结果与作废说明、能写/不能写、产物索引、论文提纲。旧 CIFAR smoke/旧 sweep/过冲 formal 目录已清理，历史数字以本日志为准。

### 2.10 Phase I CIFAR 加固（2026-08-15）

顺序：I.A 欠压修复 → sweep_v3 → I.A′ 边界加固 → I.B 100 epoch 正式基线 → I.C 多 seed 消融 → **正式全表**。

#### I.A 增量逼近（修 sweep_v2 欠压）

根因文档：[`docs/CIFAR_SEARCH_UNDERCOMPRESSION.md`](CIFAR_SEARCH_UNDERCOMPRESSION.md)。
策略：保留 2 点门禁；`max_step_compression=1.75` 多轮逼近；`max_iterations` 提高。

短验证（20 epoch 基线，seed42）：search **4.025x / 10.072x**（目标 ±15% 内）。

**sweep_v3**（`results/cifar_p12_comparison_gpu_sweep_v3/`，不覆盖 v2）：多数 ≥4x 已同压缩可比；残留 outlier：`4x seed43` 过冲到 6.32x；`10x seed42` 欠压到 8.43x。

#### I.A′ 压缩边界（过冲/欠压轻量修）

1. target 模式下禁止回退全目标 `configured_ratios`（近目标时该 fallback 会过冲）。
2. 候选压缩 `> target*1.15` 过滤。
3. `max_iterations=12`。

短验证：`4x seed43 → 4.025x`；`10x seed42 → 10.072x`（`results/cifar_p12_ia_prime_validate/`）。

#### I.B 正式基线 100 epoch

| 项 | 值 |
|----|-----|
| 配置 | `configs/cifar_resnet_baseline_gpu_formal.yaml` |
| 权重 | `checkpoints/cifar_resnet18_baseline_formal100.pth`（不覆盖 20 epoch） |
| 墙钟 | ~15.7 分钟（941s） |
| Best val | **91.84%**（相对 20 epoch smoke ~88.86% val） |

**关键对照**（2x + 10x × seeds 42/43/44）：`results/cifar_p12_comparison_gpu_formal100_key/`

| 目标 | iterative comp / test | search comp / test | 同压缩 |
|------|----------------------|--------------------|--------|
| 2.0x | 2.01x / **90.66±0.12** | 2.01x / 90.31±0.52 | 是 |
| 10.0x | 10.16x / 83.76±0.43 | **10.14x / 86.03±0.33** | 是 |

dense_baseline test **90.36%**。同压缩下：2x iterative 略优；**10x search 高于 iterative**（约 +2.3 点）。仍不据此宣称 search「全面系统更优」。

#### I.C 恢复消融多 seed

产物：`results/cifar_recovery_ablation_multiseed/`（含 test；seeds 42/43/44；2.01x Wanda）

| Level | val mean±std | test mean±std | 备注 |
|------|--------------|---------------|------|
| 1 | 87.93±0.87 | 87.25±0.82 | 全参微调；主路径 |
| 2 | 90.04±0.07 | 89.22±0.09 | LoRA；recovered 参数含适配器 |
| 3 | 89.64±0.42 | 88.51±0.28 | 自蒸馏；与 L1 同 pruned 参数量 |

多 seed 下 L2/L3 test 均值高于 L1，但 L2 参数量略增；讨论须注明设定差异，避免写成无条件「系统更优」。

#### 正式全表 formal100_full（2026-08-15）

**产物**：`results/cifar_p12_comparison_gpu_formal100_full/`
**配置**：`configs/cifar_p12_gpu_formal100_full.yaml`
**设定**：正式基线；1.5/2/4/6/8/10 × seeds 42/43/44；I.A′ 搜索边界；墙钟约 **47 分钟**；盘约 **+1.9G**。

**压缩验收**：全部 ≥4x search 落在目标 ±15%（无欠压/过冲 outlier）。

**Test mean±std（同压缩可比）**

| 目标 | iterative | search | 相对 |
|------|-----------|--------|------|
| 1.5x | 90.52±0.30 | **90.64±0.94** | 接近（search 略高，方差大） |
| 2.0x | **90.66±0.12** | 90.31±0.52 | iterative 略优 |
| 4.0x | **89.85±0.16** | 89.37±0.16 | iterative 略优 |
| 6.0x | 88.22±0.38 | **88.25±0.56** | 持平 |
| 8.0x | 86.67±0.16 | **87.40±0.97** | search 略优 |
| 10.0x | 83.76±0.43 | **86.03±0.33** | search 明显更优（约 +2.3 点） |

dense_baseline test **90.36%**。oneshot 在 ≥4x 仍崩溃（负结果保留）。

**研究读法**：低–中压缩（约 ≤4x）iterative 略稳或略优；高压缩（约 ≥8x）search 同压缩下更强。这是 **压缩率相关的 crossover**，不是「search 全面系统更优」。smoke v3 / key / full 必须分列。

#### 2.11 Crossover 机制消融（2026-08-15）

**问题**：10x 上 search 为何高于一次剪到目标的方法？

**产物**：`results/cifar_crossover_path_ablation/`
**设定**：formal100 基线；目标 10x；seeds 42/43/44；同 Level-1、2 epoch 恢复预算。
**臂**：
1. **uniform**：Wanda 一次剪到 10x + L1
2. **incremental_no_gate**：与 search 相同增量步长，但 `max_accuracy_drop_points=100`（门禁关闭）
3. **search_gated**：复用 formal100_full 的 autonomous_search（增量 + 2 点门禁）

| Arm | test mean±std | compression |
|-----|---------------|-------------|
| uniform | 82.66±0.72 | 10.16x |
| incremental_no_gate | 84.78±0.60 | 10.22x |
| search_gated | **86.03±0.33** | 10.14x |

**Verdict：`path_and_gate`**
- 增量路径相对 uniform 约 **+2.1** test 点
- 2 点门禁相对无门禁增量再约 **+1.3** test 点
- 两者都贡献；不是单一因素

仍不勾「search 系统全面优于 iterative」（全表仍是 crossover）。

#### 2.12 Crossover 稳健性消融（8x，2026-08-15）

**问题**：10x 的 `path_and_gate` 是否只是极端压缩特例？

**产物**：`results/cifar_crossover_path_ablation_8x/`
**设定**：同 formal100 基线与三臂协议；目标 **8.0x**；seeds 42/43/44。
**Search-gated**：复用 formal100_full 的 `ratio_8_seed_*`（不重跑）。
**配置**：`configs/cifar_crossover_path_ablation_8x.yaml`

| Arm | test mean±std | compression |
|-----|---------------|-------------|
| uniform | 85.76±0.81 | 8.06x |
| incremental_no_gate | 85.96±0.46 | 8.14x |
| search_gated | **87.40±0.97** | 8.08x |

**Verdict：`gate_dominant`**
- 增量路径相对 uniform 仅约 **+0.2** test 点（低于 1.0 点阈值）
- 2 点门禁相对无门禁增量约 **+1.4** test 点
- crossover 起点档（8x）上，**门禁选路是主贡献**；路径单独几乎不抬分

**与 10x 对照（稳健性读法）**

| 档 | path (no-gate − uniform) | gate (gated − no-gate) | verdict |
|----|--------------------------|------------------------|---------|
| 8x | ~+0.2 | ~+1.4 | `gate_dominant` |
| 10x | ~+2.1 | ~+1.3 | `path_and_gate` |

机制不是 10x 单点幻觉：两档上 **门禁都稳定贡献约 +1.3～1.4**；路径贡献随压缩加剧而变大（8x 可忽略 → 10x 显著）。仍不勾「search 系统全面优于 iterative」。

#### 2.13 低压缩差距诊断（4x，2026-08-15）

**问题**：全表上 4x iterative（89.85±0.16）略高于 search（89.37±0.16）——是缺路径/门禁，还是低压缩本就难分？

**产物**：`results/cifar_crossover_path_ablation_4x/`（含 `PROCESS_COMPARE.md`）
**配置**：`configs/cifar_crossover_path_ablation_4x.yaml`
**设定**：同三臂协议；目标 **4.0x**；search_gated 复用 formal100_full `ratio_4_seed_*`。

| Arm | test mean±std | compression |
|-----|---------------|-------------|
| uniform | 89.27±0.14 | 4.02x |
| incremental_no_gate | 89.45±0.39 | 4.02x |
| search_gated | 89.37±0.16 | 4.02x |

**Verdict：`inconclusive_close`**（三臂两两差距均 <1.0 点）

**Process（seed 42/43）**
- search 均 3 次 accept：约 1.75x → 3.09x → 4.02x，无 reject
- 最终 `layer_keep_indices` 与 **iterative_structured_level1 完全相同**（同结构、同参数量）
- 因此 ~0.5 test 点差距更像 **恢复轨迹/优化差异**（多步恢复 vs 一次剪到目标+一次恢复），不是「缺门禁」或「选错层宽」

**跨档机制表**

| 档 | path | gate | verdict | 相对 iterative |
|----|------|------|---------|----------------|
| 4x | ~+0.2 | ~−0.1 | `inconclusive_close` | search 略低 ~0.5 |
| 8x | ~+0.2 | ~+1.4 | `gate_dominant` | search 略高 |
| 10x | ~+2.1 | ~+1.3 | `path_and_gate` | search 明显高 |

**诊断结论**：低压缩上 path/gate 几乎不分胜负；主张应定为 **regime-dependent**。要把 `[×]` 改成系统全面更优，需改探索/恢复预算公平性后再关键复验——**不是**放宽 2pt 门禁。

#### 2.14 恢复预算对齐（2026-08-15）

**问题**：4x 上 search 累计约 6 epoch 恢复、iterative 仅 2 epoch——iterative 略优是否因为「恢复不够」？

**产物**：`results/cifar_p12_budget_match_key/`（含 `BUDGET_REPORT.md`）
**配置**：`configs/cifar_p12_budget_match_key.yaml`
**设定**：formal100 基线；2x/4x × seeds 42/43/44；`iterative_recovery_epochs=6`；search 仍每 accept 2 epoch；方法含 dense 锚点。

| target | iterative test | search test | delta (it−se) |
|--------|----------------|-------------|---------------|
| 2x | 90.56±0.07 | 90.18±0.43 | +0.38 |
| 4x | 90.04±0.18 | 89.43±0.32 | +0.61 |

相对 formal100（iterative 仅 2 epoch）：4x iterative 从 89.85→90.04（略升），search 基本不变。
**结论**：抬高 iterative 预算后仍 ≥ search；低压缩差距 **不是**「iterative 恢复不够」。多步搜索轨迹假说保留。「系统全面更优」仍 `[×]`。

#### 2.15 低压缩一步到目标（2026-08-15）

**改动**：`target<=4` 时有效 `max_step_compression = max(configured, target)`（允许一轮逼近目标）；`target>4` 仍用 1.75。门禁仍 2pt。

**产物**：`results/cifar_p12_lowcomp_step_key/`（含 `LOWCOMP_STEP_REPORT.md`）
**配置**：`configs/cifar_p12_lowcomp_step_key.yaml`
**设定**：formal100 基线；2/4/8/10 × seeds 42/43/44；iterative 仍 2 epoch 恢复。

| target | iterative | search | delta (se−it) |
|--------|-----------|--------|---------------|
| 2x | 90.56±0.07 | **90.57±0.08** | +0.01 |
| 4x | **89.27±0.15** | 88.87±0.11 | −0.39 |
| 8x | 86.13±0.56 | **86.99±0.53** | +0.86 |
| 10x | 84.14±0.72 | **85.60±0.15** | +1.46 |

**读法**：
- 2x 基本抹平；**4x 仍未达标**（且 search 相对 formal100 同档略降）
- 8x/10x search 仍高于 iterative，高压缩优势未明显丢失
- 一步策略 **不能**单独把「系统全面更优」翻成 `[√]`；regime-dependent 叙事保留

#### 2.16 4x 一步变差诊断（2026-08-15）

**产物**：`results/cifar_p12_lowcomp_step_key/FOURX_PROCESS_COMPARE.md`

| seed | formal traj (n_accept) | lowcomp traj | widths formal==lowcomp==iterative | search test F→L |
|------|------------------------|--------------|-------------------------------------|-----------------|
| 42 | 1.75→3.09→4.02 (3) | 4.02 (1) | 是 | 89.52→88.81 |
| 43 | 1.75→3.09→4.02 (3) | 4.02 (1) | 是 | 89.21→89.03 |
| 44 | 1.75→3.09→4.02 (3) | 4.02 (1) | 是 | 89.39→88.78 |

**结论**：最终层宽相同；变差来自 **缺少中间恢复**，不是选错结构。
**代码跟进**：`effective_max_step_compression` 收窄为仅 `target<=2`（4x 起恢复增量路径）。不重跑全表；「系统全面更优」仍 `[×]`。

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

- [×] CIFAR 上自主搜索已经**全面系统**优于迭代剪枝（formal100 全表：≤4x iterative 略优或接近；≥8x search 更高 → crossover）
- [×] LoRA / 自蒸馏已无条件系统优于 Level 1
- [×] 把 smoke（20 epoch）与正式全表混写为同一主结果

**可以写进机制讨论的**：
- [√] 高压缩优势可拆为 **增量路径** 与 **2 点门禁选路**；贡献随压缩率变化（4x 难分；8x 门禁主导；10x 路径+门禁），见 §2.11–2.13
- [√] 低压缩（4x）search/iterative 最终结构可相同，小幅差距来自恢复轨迹而非层宽选择
- [√] 4x 硬一步到目标会跳过中间恢复、同结构下 test 变差；一步策略仅适用于约 ≤2x（§2.16）

---

## 4. 下一步（按优先级）

1. [√] 叙事定稿：regime-dependent；「系统全面更优」仍 `[×]`
2. [√] Step 2–3 预算对齐与一步关键复验；§2.16 诊断后一步策略收窄为 `target<=2`
3. [√] Step 4：Phase J 规划 — [PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md)
4. **之后**：Phase K 须先扩盘（约 30G+）；视觉域主实验链可停，不重跑全表；写论文见 [PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md)

Phase H / Phase I 证据：[EVIDENCE_PACK.md](EVIDENCE_PACK.md)。

---

## 5. 不进 Git 的东西

`results/`、`data/`、`checkpoints/`、`venv/` 默认不提交。日志本文在 `docs/`，记录结论；原始数字以 study 目录里的 JSON 为准。
