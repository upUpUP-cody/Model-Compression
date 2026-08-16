# Autonomous Lottery Ticket Discovery - 项目实施计划

> 本文件是项目的完整长期路线图、阶段目标和全局成功标准。当前 P0/P1/P2 的执行顺序、研究协议和阶段验收条件见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。详细实验结论见 [docs/WORK_LOG.md](docs/WORK_LOG.md) / [docs/WORK_LOG_BRIEF.md](docs/WORK_LOG_BRIEF.md)。
>
> **更新状态（2026-08-16）**
>
> - **已完成**：MVP；MNIST / CIFAR P1.2；formal100 主表与 crossover 机制；Phase K0–K5 **SQuAD 管线冒烟**；KG.0–KG.4 **SST-2 冒烟**。
> - **当前优先**：论文收口 — CIFAR 主叙事 + GLUE 过渡；SQuAD 记为 **恢复不足局限**（加深 1.5x 后 F1 仍塌）。见 [docs/PHASE_K_QWEN_PLAN.md](docs/PHASE_K_QWEN_PLAN.md)。
> - **不默认**：重跑 CIFAR formal100；跳过 GLUE 直接开 SQuAD 正式对照；宣称 search 系统全面更优。
> - **主机**：1×RTX 4090（默认单卡；仅 Agent 判定需双卡拆任务时再加第 2 卡）；权重/数据 `/mnt/data`；LLM 运行产物 `/mnt/data2`。
> - **资源告警**：加卡 / 配置优化触发标准见 [CLAUDE.md](CLAUDE.md)「LLM / GPU 与配置优化告知」。

## 项目概述

本项目旨在复现和实现《Autonomous Lottery Ticket Discovery》论文中提出的自主彩票发现方法，该方法能够在不依赖预训练父网络的情况下，自主发现高性能的稀疏神经网络子结构。

当前研究叙事已从「先跑通 MNIST MLP」推进到「同一 train/val/test 协议迁移到 CIFAR ResNet」；自主搜索相对 one-shot / 迭代剪枝的价值，必须在 **同压缩预算** 下解读（见 WORK_LOG §2.8）。

---

## 阶段划分与实施步骤

### **阶段 0: 环境搭建与基础设施 (预计 3-5 天)**

#### 步骤 0.1: 项目结构初始化
- **任务**: 创建标准的 Python 项目结构
- **产出**:
  ```
  Model-Compression/
  ├── src/
  │   ├── models/          # 神经网络模型定义
  │   ├── pruning/         # 剪枝算法实现
  │   ├── recovery/        # 恢复策略实现
  │   ├── controller/      # 压缩控制器
  │   ├── evaluation/      # 评估与指标
  │   └── utils/           # 工具函数
  ├── configs/             # 配置文件
  ├── experiments/         # 实验脚本
  ├── tests/               # 单元测试
  ├── data/                # 数据集目录
  ├── checkpoints/         # 模型检查点
  ├── logs/                # 训练日志
  └── results/             # 实验结果
  ```

#### 步骤 0.2: 依赖安装
- **任务**: 创建 `requirements.txt` 并安装核心依赖
- **核心依赖**:
  - PyTorch >= 2.0.0
  - torchvision
  - transformers (用于 Transformer 模型)
  - numpy, pandas
  - matplotlib, seaborn (可视化)
  - wandb 或 tensorboard (实验跟踪)
  - pytest (测试框架)
- **验证**: 运行 `python -c "import torch; print(torch.__version__)"`

#### 步骤 0.3: 数据集准备
- **任务**: 下载并预处理基准数据集
- **数据集选择** (按难度递增):
  1. **MNIST**: 简单图像分类 (用于初期验证) [完成] **[CPU 友好，自动下载]**
  2. **CIFAR-10**: 图像分类 [完成] **[已在 RTX 4090 跑通 P1.2]**
  3. **SQuAD 2.0**: 问答任务 (论文主要基准) [必须 GPU] **[P3 规划占位，现阶段不实现]**
  4. **MMLU**: 多任务语言理解 (可选) [必须 GPU] **[必须 GPU]**
- **产出**:
  - `data/mnist/` 或 `data/cifar10/`
  - `data/squad/` (train.json, dev.json) — 未准备
  - 数据加载器：`src/utils/data_loader.py`（含 CIFAR train/val split + `split_hash`）

---

### **阶段 A: 基础组件实现 (预计 7-10 天)**

#### 步骤 A.1: 实现 Dense Baseline 模型
- **状态**: [完成] 已完成（MNIST MLP 基线模型和检查点可用）
- **任务**: 构建并训练完整的密集神经网络作为性能基准
- **模型选择**:
  - **简单**: 3-5 层 MLP (用于 MNIST) [完成] **[CPU 可运行]**
  - **中等**: ResNet-18 (用于 CIFAR-10) [完成] **[已在 RTX 4090 跑通；项目内 `src/models/resnet_cifar.py`]**
  - **复杂**: BERT-base 或 Qwen-0.5B (用于 SQuAD) [必须 GPU] **[P3 规划占位，现阶段不实现]**
- **实现文件**: `src/models/dense_baseline.py`、`src/models/resnet_cifar.py`
- **验证指标**:
  - 记录 baseline 准确率/F1 分数
  - GPU 内存占用
  - 推理延迟
  - 参数量
  - CIFAR smoke 基线约 validation 88.86% / test 87.88%（20 epoch，非论文级 100+ epoch）

#### 步骤 A.2: 实现结构化剪枝算子
- **状态**: [完成] MLP 物理裁剪 + **ResNet BasicBlock 内 conv1** 结构化剪枝（残差 I/O 不变）
- **任务**: 实现论文中的结构化剪枝方法
- **实现文件**: `src/pruning/structured_pruning.py`、`src/pruning/cnn_structured_pruning.py`、`src/pruning/pruning_backend.py`
- **核心功能**:
  ```python
  class StructuredPruning:
      def prune_mlp_layer(self, layer, ratio):
          """剪枝 MLP 层的神经元"""
          pass

      def prune_attention_head(self, attention, ratio):
          """剪枝 Transformer 注意力头"""
          pass

      def apply_mask(self, model, pruning_mask):
          """应用剪枝掩码到模型"""
          pass
  ```
- **验证**: 单元测试确保剪枝后模型可运行

#### 步骤 A.3: 实现敏感度分析 (Self-Diagnosis)
- **状态**: [完成] 已完成（Wanda 神经元重要性已接入自主搜索）
- **任务**: 实现层级敏感度评估方法
- **实现文件**: `src/pruning/sensitivity.py`
- **核心指标**:
  1. **Gradient Sensitivity**: 梯度范数
  2. **Weight Magnitude**: 权重绝对值
  3. **Wanda Score**: 权重×激活值
  4. **SparseGPT Score** (可选,用于 Transformer)
- **函数签名**:
  ```python
  def compute_layer_sensitivity(
      model,
      dataloader,
      method='wanda'
  ) -> dict:
      """
      返回: {layer_name: sensitivity_score}
      """
      pass
  ```

---

### **阶段 B: 前沿分析与候选生成 (预计 5-7 天)**

#### 步骤 B.1: 实现 Capability Frontier Profiling
- **状态**: [完成] `ParetoFrontier` / `FrontierPoint` 已实现并接入自主搜索（validation accuracy vs 真实参数量）；独立 profiling 实验脚本可按需复用
- **任务**: 定义并计算压缩前沿
- **实现文件**: `src/evaluation/frontier.py`
- **核心概念**:
  ```python
  class FrontierProfile:
      def __init__(self):
          self.capability_gap = None      # 性能差距
          self.compression_gap = None     # 压缩差距
          self.context_gap = None         # 上下文差距

      def compute_capability_gap(self, child_acc, parent_acc):
          """计算子网络与父网络的性能差距"""
          return parent_acc - child_acc

      def is_on_frontier(self, sparsity, accuracy):
          """判断是否在帕累托前沿上"""
          pass
  ```

#### 步骤 B.2: 实现 Lottery Ticket Proposal
- **状态**: [完成] MVP 已完成（自主搜索生成确定性层/比例候选并跳过重复配置；独立 proposal 模块尚未拆出）
- **任务**: 基于敏感度生成候选子网络
- **实现文件**: `src/pruning/ticket_proposal.py`
- **策略**:
  1. **Region Selection**: 选择要剪枝的层/模块
     - 基于敏感度排序
     - 跳过首尾层 (保持输入输出维度)
  2. **Pruning Ratio**: 确定剪枝比例
     - 固定比例: 30%, 50%, 70%, 90%
     - 自适应比例: 根据 capability gap 调整
- **产出**: 候选子网络配置列表

#### 步骤 B.3: 实现 Cheap Critic 评估
- **状态**: [完成] 已完成（严格样本上限、无副作用评估和指标测试已覆盖）
- **任务**: 快速评估候选子网络质量
- **实现文件**: `src/evaluation/cheap_critic.py`
- **方法**:
  - 使用小样本数据 (100-500 samples)
  - 计算 loss 而非完整准确率
  - 评估时间 < 1 分钟/候选
- **函数**:
  ```python
  def quick_evaluate(model, mini_dataset):
      """快速评估,返回 loss"""
      pass
  ```

---

### **阶段 C: 恢复策略实现 (预计 7-10 天)**

#### 步骤 C.1: Level 0 - 无恢复
- **状态**: [完成] 已完成（Cheap Critic 与完整验证可直接评估剪枝候选）
- **任务**: 直接评估剪枝后的子网络
- **用途**: 作为对照基准

#### 步骤 C.2: Level 1 - 仅重建
- **状态**: [完成] 已完成（搜索循环已调用 `quick_recovery`，每次剪枝后重新创建优化器）
- **任务**: 在压缩损伤后重新初始化并训练
- **实现文件**: `src/recovery/reconstruction.py`
- **核心逻辑**:
  ```python
  def reconstruction_recovery(child_model, train_loader, epochs=5):
      """
      简单的微调训练
      """
      optimizer = torch.optim.Adam(child_model.parameters())
      for epoch in range(epochs):
          train_one_epoch(child_model, train_loader, optimizer)
      return child_model
  ```

#### 步骤 C.3: Level 2 - LoRA 前沿数据恢复
- **状态**: [完成] 接口已实现（Conv2d/Linear LoRA）；CIFAR 2x 同候选消融已跑通。**单 seed、仅 validation**；未证明系统优于 Level 1
- **任务**: 使用 LoRA 适配器恢复剪枝后模型
- **实现文件**: `src/recovery/lora_recovery.py`、`src/recovery/recovery_dispatch.py`
- **硬件需求**: [建议 GPU] **[已在 RTX 4090 / fp16 验证]**
- **要点**:
  1. 冻结 base 权重，仅训练低秩适配器
  2. 适配器与 base 同设备；AMP 下残差在 fp32 计算；stride 与 base 对齐
  3. 报告部署体积时区分 pruned 基础参数与适配器开销

#### 步骤 C.4: Level 3 - Self-Distillation
- **状态**: [完成] 接口已实现；CIFAR 2x 同候选消融已跑通。**单 seed**；未证明系统优于 Level 1
- **任务**: 使用冻结 baseline 作为教师进行知识蒸馏
- **实现文件**: `src/recovery/self_distillation.py`
- **硬件需求**: [建议 GPU] **[已在 RTX 4090 验证]**
- **损失函数**: soft KL（温度）+ hard CE；见实现与 `configs/cifar_recovery_ablation.yaml`

---

### **阶段 D: 压缩控制器实现 (预计 5-7 天)**

#### 步骤 D.1: 实现简单的启发式控制器
- **状态**: [完成] 已完成（`accept`、`reject`、`rollback`、`regrow` 和重复候选优先级均已测试）
- **任务**: 基于规则的决策逻辑
- **实现文件**: `src/controller/heuristic_controller.py`
- **决策规则**:
  ```python
  class HeuristicController:
      def decide_action(self, profile, candidate, history):
          """
          返回: 'accept', 'reject', 'rollback', 'regrow'
          """
          if profile.capability_gap < threshold:
              return 'accept'
          elif candidate.quality < min_quality:
              return 'reject'
          elif len(history.failures) > max_failures:
              return 'rollback'
          else:
              return 'regrow'
  ```

#### 步骤 D.2: 实现自适应规则控制器 (可选)
- **状态**: [未开始] 未开始
- **任务**: 基于历史动态调整决策
- **实现文件**: `src/controller/adaptive_controller.py`
- **特性**:
  - 根据 capability gap 调整剪枝比例
  - 根据 failure mode 调整恢复策略
  - 动态调整搜索预算

#### 步骤 D.3: 实现 LLM-based 控制器 (高级,可选)
- **状态**: [未开始] 未开始
- **任务**: 使用语言模型进行推理决策
- **实现文件**: `src/controller/llm_controller.py`
- **步骤**:
  1. 构造 prompt 模板
  2. 调用 OpenAI API 或本地 LLM
  3. 解析 LLM 输出为决策

---

### **阶段 E: 端到端流程集成 (预计 5-7 天)**

#### 步骤 E.1: 实现主搜索循环
- **状态**: [完成] MVP 已完成（Wanda 评分、候选深拷贝、Cheap Critic、恢复、全量验证、接受/回滚及 JSON 安全历史）
- **任务**: 整合所有模块,实现完整的自主搜索
- **实现文件**: `src/autonomous_search.py`
- **伪代码**:
```python
def autonomous_lottery_ticket_discovery(
    parent_model,
    train_loader,
    val_loader,
    max_iterations=10,
    target_sparsity=0.9
):
    """
    主搜索循环
    """
    history = SearchHistory()
    current_model = parent_model

    for iteration in range(max_iterations):
        # 1. Self-Diagnosis
        sensitivity = compute_layer_sensitivity(current_model, train_loader)

        # 2. Profile Frontier
        profile = compute_frontier_profile(current_model, val_loader)

        # 3. Propose Candidates
        candidates = generate_ticket_candidates(
            current_model,
            sensitivity,
            k=5
        )

        # 4. Cheap Evaluation
        scored_candidates = [
            (cand, cheap_critic(cand)) for cand in candidates
        ]
        best_candidate = max(scored_candidates, key=lambda x: x[1])

        # 5. Controller Decision
        action = controller.decide_action(profile, best_candidate, history)

        if action == 'accept':
            # 6. Recovery
            current_model = apply_recovery(
                best_candidate,
                train_loader,
                level=2
            )
            history.add_success(current_model, profile)

        elif action == 'rollback':
            current_model = history.get_last_accepted()

        elif action == 'regrow':
            current_model = regrow_layers(current_model)

        # 7. Full Evaluation
        accuracy = full_evaluate(current_model, val_loader)
        print(f"Iteration {iteration}: Accuracy={accuracy:.2f}%")

        # 8. Check Stopping Condition
        if profile.capability_gap < 0.01 and get_sparsity(current_model) >= target_sparsity:
            break

    return current_model, history
```

#### 步骤 E.2: 实现配置管理
- **状态**: [完成] MVP 已完成（CPU 专用 YAML、必填项和数值范围校验、统一随机种子）
- **任务**: 创建配置文件系统
- **实现文件**: `configs/default.yaml`
- **配置项**:
```yaml
model:
  type: "mlp"  # mlp, resnet, bert
  hidden_dims: [512, 256, 128]

pruning:
  method: "wanda"  # wanda, magnitude, gradient
  target_sparsity: 0.9
  structured: true

recovery:
  level: 2  # 0-3
  lora_rank: 8
  distillation_temp: 2.0

controller:
  type: "heuristic"  # heuristic, adaptive, llm
  capability_threshold: 0.05
  max_failures: 3

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 20

search:
  max_iterations: 10
  candidates_per_round: 5
  search_budget_hours: 24
```

#### 步骤 E.3: 实现日志与可视化
- **状态**: [完成] MVP 已完成（JSONL、summary JSON、CSV、checkpoint 和 PNG 搜索历史图）
- **任务**: 记录实验过程并可视化结果
- **实现文件**: `src/utils/logger.py`, `src/utils/visualizer.py`
- **功能**:
  1. 保存每轮迭代的指标
  2. 绘制稀疏度-准确率曲线
  3. 可视化搜索轨迹
  4. 导出实验报告

---

### **阶段 F: 实验验证 (预计 7-10 天)**

#### 步骤 F.1: 单元测试
- **状态**: [完成] 全量 `python -m pytest tests -q` → **121 passed, 1 skipped**（含 CNN 剪枝、压缩目标、搜索止损、LoRA/蒸馏等）
- **任务**: 为每个模块编写测试
- **测试文件**: `tests/test_*.py`
- **覆盖**:
  - 剪枝操作正确性（MLP + ResNet conv1）
  - 敏感度计算准确性
  - 恢复策略与 dispatch
  - 控制器决策逻辑与搜索止损

#### 步骤 F.2: 小规模验证实验（MNIST）
- **状态**: [完成] MNIST CPU 冒烟 + **MNIST P1.2 GPU** smoke/formal/3-seed/6×3 sweep
- **任务**: 在 MNIST 上验证完整流程与六方法对照
- **实验脚本**: `experiments/run_p12_comparison.py`、`experiments/run_p12_multiseed.py`
- **验证点**:
  - Dense baseline 与 dense_small（从零训练）协议正确
  - One-shot 高压缩崩溃（负结果保留）
  - 迭代 / 自主搜索在高压缩下仍接近 baseline（约 97–98%）

#### 步骤 F.2b: CIFAR-10 + ResNet-18 P1.2（P2）
- **状态**: [完成] 协议与主证据已齐；详见 WORK_LOG §2.2–2.8 与 [docs/P2_EXECUTION_PLAN.md](docs/P2_EXECUTION_PLAN.md)
- **任务**: 同一六方法协议迁移到 CNN
- **要点**:
  - 只剪 BasicBlock 内 `conv1`；残差 I/O 宽度不变
  - `target_compression_ratio` 推导全部 prunable conv1；搜索 `target_compression_reached` 止损
  - 产物：`results/cifar_p12_comparison_gpu_sweep_v2/`、`results/cifar_recovery_ablation/`
- **结论边界（必须遵守）**:
  - **1.5x / 2.0x**：search 与 iterative **同压缩**，可公平对照；iterative 略优或接近，**不能**声称 search 系统更优
  - **≥4x**：search 常因 2 点能力门禁欠压；iterative 仍打到目标（10x test 约 83.4%）
  - 旧 sweep search=1.00x 与 7.66x 过冲 formal **不得**与同压缩表混写
  - 20 epoch 基线约 88% val，非正式论文级精度

#### 步骤 F.3: 复现论文实验 E1（Qwen / SQuAD）
- **状态**: [未开始] **P3 规划占位，现阶段不实现**（启动门禁见下文 Phase J/K）
- **任务**: Dense Baseline 对比 (对应论文 Table 1)
- **硬件需求**: [必须 GPU] **[必须 GPU ≥16GB，Qwen-0.5B 模型大，CPU 完全不可行]**
- **实验配置**:
  - Model: Qwen-0.5B
  - Dataset: SQuAD 2.0
  - 对比: Dense vs One-shot Wanda vs Iterative IMP
- **预期结果**:
  - Dense: ~85% F1
  - One-shot 90% sparsity: ~60% F1
  - Our method 90% sparsity: >75% F1

#### 步骤 F.4: 复现论文实验 E2
- **状态**: [未开始] **P3 规划占位**（需 GPU 和多次完整训练）
- **任务**: 热稀疏度曲线 (对应论文 Figure 2)
- **硬件需求**: [必须 GPU] **[必须 GPU ≥16GB，需要多次完整训练]**
- **实验配置**:
  - 扫描稀疏度: 30%, 50%, 70%, 90%, 95%
  - 记录每个稀疏度下的最佳准确率
- **可视化**: 绘制 Pareto frontier

#### 步骤 F.5: One-shot vs Iterative vs Search 对照
- **状态**: [部分完成] MNIST 与 CIFAR 六方法协议已跑；CIFAR sweep_v2 提供同压缩（1.5x/2.0x）与高压缩欠压边界
- **任务**: 验证自主方法相对传统路径的表现
- **验证**: **当前证据不支持**「CIFAR 上 search 系统优于 iterative」；高压缩主证据仍是 iterative Level-1

#### 步骤 F.6: 消融实验
- **状态**: [部分完成] CIFAR 固定 2x Wanda 候选上 Level 1/2/3 已跑（`results/cifar_recovery_ablation/`）；缺多 seed / test 冻结
- **任务**: 验证各组件的贡献
- **对比组**:
  1. 无恢复 (Level 0) — oneshot 臂已覆盖
  2. 仅重建 (Level 1) — [完成] 消融 + 迭代路径主证据
  3. LoRA 恢复 (Level 2) — [完成] 单次消融接口验证
  4. 自蒸馏 (Level 3) — [完成] 单次消融接口验证
  5. 完整方法 — 搜索路径已有，但高压缩同预算仍受限

---

### **阶段 G: 优化与扩展 (可选,预计 5-7 天)**

#### 步骤 G.1: 性能优化
- **状态**: [未开始]
- **任务**: 加速搜索过程
- **硬件需求**: [必须 GPU] **[必须 GPU，混合精度训练需要 CUDA]**
- **优化点**:
  - 并行化候选评估
  - 缓存中间结果
  - 混合精度训练 (FP16) [必须 GPU] **[需要 GPU with Tensor Cores]**
  - 梯度累积

#### 步骤 G.2: 支持更多模型架构
- **状态**: [未开始]
- **任务**: 扩展到其他模型
- **硬件需求**: [必须 GPU] **[所有大型模型都必须 GPU ≥16GB]**
- **候选**:
  - Vision Transformer (ViT) [必须 GPU] **[GPU ≥12GB]**
  - GPT-2 / LLaMA [必须 GPU] **[GPU ≥16GB]**
  - Diffusion Models [必须 GPU] **[GPU ≥24GB]**

#### 步骤 G.3: 支持更多剪枝策略
- **状态**: [未开始]
- **任务**: 集成其他 SOTA 剪枝方法
- **方法**:
  - SparseGPT
  - Optimal Brain Compression (OBC)
  - Movement Pruning

---

## 实施建议

### 开发顺序
1. **由简到繁**: 先在 MNIST 验证,再到 CIFAR-10,最后到 SQuAD/LLM
2. **模块化**: 每个阶段独立开发和测试
3. **持续集成**: 每完成一个步骤就运行已有测试

### 里程碑检查点
- **Checkpoint 1**: 阶段 A 完成 → Dense baseline 训练成功 [完成]
- **Checkpoint 2**: 阶段 C.2 完成 → One-shot pruning + 简单恢复可运行 [完成]
- **Checkpoint 3**: 阶段 E.1 完成 → 端到端搜索循环可运行 [完成]
- **Checkpoint 4**: 阶段 F.2 完成 → MNIST 实验验证通过 [完成]
- **Checkpoint 4b**: 阶段 F.2b 完成 → CIFAR P1.2 + sweep_v2 / 消融 [完成]
- **Checkpoint 5**: 阶段 F.3 完成 → 论文主实验（Qwen/SQuAD）复现 [未开始 / P3]

### 时间估算
- **最小可行版本 (MVP)**: 阶段 0 + A + B + C.1-C.2 + E → **已完成**
- **视觉域完整证据 (P1+P2)**: MNIST + CIFAR P1.2 → **已完成**
- **论文 LLM 复现**: 阶段 F.3–F.4 → 待 Phase J/K，约数周（视数据与显存）
- **扩展版本**: 阶段 G → 可选

### 技术难点预警
1. **Frontier 计算**: 需要大量候选评估,计算成本高 [建议 GPU] **[GPU 推荐，CPU 耗时 10x+]**
2. **LoRA 恢复**: 超参数敏感；CIFAR 消融已通，系统优势未证明 [建议 GPU]
3. **控制器设计**: 决策逻辑复杂,需多次迭代 [完成] **[CPU 可运行]**
4. **同压缩对照**: 搜索多轮叠加会过冲；已用 `target_compression_reached` 止损；高压缩下能力门禁仍可能导致欠压
5. **大模型实验**: GPU 内存需求高 [必须 GPU] **[必须 GPU ≥16–24GB VRAM]**

---

## 硬件配置需求总结

### 图例说明
- [完成] **[CPU 可运行]**: 可以在 CPU 上完成，性能可接受
- [建议 GPU] **[建议 GPU]**: CPU 可运行但速度慢 5-100 倍，强烈建议使用 GPU
- [必须 GPU] **[必须 GPU]**: CPU 环境不可行，必须使用 GPU

### 当前硬件（RTX 4090 已可用）
**已在本机/4090 完成的任务:**
- [完成] 阶段 0–E：MVP 与 MNIST 路径
- [完成] MNIST P1.2 GPU sweep
- [完成] CIFAR ResNet-18 基线、P1.2 对照、sweep_v2、恢复消融
- [完成] Level 2/3 恢复接口在 GPU/fp16 上的冒烟与消融

**仍需规划 / 更大资源的任务:**
- [必须 GPU] SQuAD/Qwen 实验 (必须 GPU ≥16GB) — **Phase J/K，现阶段不实现**
- [√] 可选 CIFAR 100+ epoch 强基线再扫（Phase I 关键对照已完成）

### 建议策略
1. **近端**：Phase H 证据包装（文档），不新开长实验除非明确需要
2. **已完成**：Phase I CIFAR 加固
3. **LLM**：仅当 Phase H 审查通过且数据/显存就绪后进入 Phase K 实现

---

## 成功标准

### 最低标准 (MVP)
- [完成] Dense baseline 已具备 MNIST MLP 检查点；准确率已在 P1.2 中记录
- [完成] One-shot / 结构化剪枝可执行，高压缩负结果保留
- [完成] Level 1 恢复能在 CIFAR/MNIST 上显著拉回剪枝后精度
- [完成] 端到端搜索循环可运行完整（含审计历史）

### 视觉域目标标准 (P1 + P2，相对论文 LLM 目标的中间层)
- [√] train/val/test 隔离协议在 MNIST 与 CIFAR 上可复现
- [√] 同压缩预算下可比较 search vs iterative（CIFAR 1.5x–10x 正式全表）
- [√] 物理结构化剪枝参数量可测（非掩码稀疏）
- [√] 论文级 CIFAR 精度（100 epoch 正式基线 + 关键对照）— Phase I
- [√] 压缩率 crossover 证据（≤4x iterative 略优/接近；≥8x search 同压缩更高）
- [√] crossover 机制消融（10x=`path_and_gate`；8x=`gate_dominant`；4x=`inconclusive_close`）
- [×] 证明 search **系统全面**优于 iterative — **当前证据不支持**（应为 regime-dependent）

### 目标标准 (论文 LLM 复现)
- [√] 先在 **GLUE 正式标准（SST-2 + RTE + QNLI）** 上给出同预算压缩对照信号（Phase K §KG.5；短文本闭集 NLU）
- [ ] 再在 **SQuAD** 上达到可报告的 F1/EM（长文抽答；± 约定容差；**KG.5 门禁已满足**）
- [ ] 优于 One-shot Wanda baseline 至少 10%（同协议对齐后）
- [ ] 优于传统 IMP / 人工设计 iterative 至少 5%（同压缩预算）

### 优秀标准 (超越论文)
- [ ] 在论文未测试的数据集上验证
- [ ] 提出改进的控制器策略
- [ ] 支持 3 种以上模型架构
- [ ] 搜索时间优化 50% 以上

---

## 参考资料

### 论文
1. **主论文**: Autonomous Lottery Ticket Discovery
2. **相关工作**:
   - The Lottery Ticket Hypothesis (Frankle & Carbin, 2019)
   - Wanda: Pruning by Weights and Activations (Sun et al., 2023)
   - SparseGPT (Frantar & Alistarh, 2023)
   - **用途**：相关工作不仅列引用，还要服务 **人工设计 baseline vs autonomous_search** 对照（调研短表 + 能复现则同预算实验）；**调研优先顶会/高引用权威工作**，详见 [`docs/PAPER_RESULTS_OUTLINE.md`](docs/PAPER_RESULTS_OUTLINE.md) §8 质量门禁、[`docs/PHASE_K_QWEN_PLAN.md`](docs/PHASE_K_QWEN_PLAN.md) K6-lit

### 代码参考
- PyTorch Pruning Tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- Wanda 官方实现: https://github.com/locuslab/wanda
- PEFT (LoRA): https://github.com/huggingface/peft

### 数据集
- MNIST: `torchvision.datasets.MNIST`
- CIFAR-10: `torchvision.datasets.CIFAR10`
- GLUE: https://gluebenchmark.com/（Phase K **正式标准**：SST-2 + RTE + QNLI；短文本闭集 NLU；冒烟仅用 SST-2）
- SQuAD 2.0: https://rajpurkar.github.io/SQuAD-explorer/（长文阅读理解 / 抽答；**KG.5 门禁已通过，可开**小矩阵）

---

## 后续大框架（Phase H / I / J / K）

P0（搜索正确性）→ P1（MNIST P1.2）→ P2（CIFAR P1.2）**已完成**。后续不再从「阶段 0」起步，而按下列大框架推进。详细数字与「能写/不能写」见 [docs/WORK_LOG.md](docs/WORK_LOG.md)。

```mermaid
flowchart TD
  done[P0_P1_P2_done] --> H[Phase_H_EvidencePack]
  H --> I[Phase_I_Optional_CIFAR]
  H --> J[Phase_J_Qwen_PlanOnly]
  I --> J
  J --> K[Phase_K_Implement_iff_gates]
```

### Phase H — 证据包装与论文叙事（近端默认下一步）

- **状态**：[完成] 已交付 [`docs/EVIDENCE_PACK.md`](docs/EVIDENCE_PACK.md)
- **性质**：文档为主，不默认新开长实验
- **内容**：
  - 从 WORK_LOG / sweep_v2 / 消融抽出协议图、同压缩表、负结果（oneshot 崩溃、search ≥4x 欠压）
  - 固化「能写 / 不能写」清单（与 WORK_LOG §3 一致）
- **产出**：证据索引 + 论文方法/实验/讨论提纲
- **验收**：读者仅凭文档能复述协议与主结论，且不把欠压 search 与高压缩 iterative 混比

### Phase I — CIFAR 加固（含正式全表，已完成）

- [√] 高压缩同预算：增量逼近 + 过冲硬顶（I.A / I.A′）；sweep_v3 + outlier 短验证
- [√] 更强基线（100 epoch）+ 关键对照 2x/10x × 3 seed（I.B）
- [√] 恢复消融多 seed + test（I.C）
- [√] **正式全表**：`formal100` × 1.5–10x × 3 seed（`results/cifar_p12_comparison_gpu_formal100_full/`）
- [√] crossover 机制消融 10x（`path_and_gate`）+ 8x（`gate_dominant`）+ 4x 低压缩诊断（`inconclusive_close`）
- [×] 勾选「search 系统全面优于 iterative」— **不做**（regime-dependent；全表仍是 crossover）
- **不阻塞** Phase H；也不自动启动 Qwen 实现
- 证据已写入 [`docs/EVIDENCE_PACK.md`](docs/EVIDENCE_PACK.md) / [`docs/WORK_LOG.md`](docs/WORK_LOG.md)

### Phase J — Qwen/SQuAD 规划占位（对齐 P2.9，默认不写代码）

- **状态**：[√] 规划文档已交付 [`docs/PHASE_J_QWEN_PLAN.md`](docs/PHASE_J_QWEN_PLAN.md)
- 剪枝单元：attention head / FFN 中间维
- 指标：F1 / EM；硬件：VRAM ≥16GB
- **启动门禁**：Phase H 证据包审查通过 + 数据与显存就绪
- **磁盘**：实现前须先提醒用户扩盘（Qwen 权重/缓存/多次 run 通常还需 **30G+** 空闲）
- 本阶段只写接口草图与实验矩阵，**不实现** Transformer 剪枝 / SQuAD pipeline；**不下载**权重

### Phase K — LLM 实现（KG.5/K6/加深恢复/lit 短表 `[√]`；下一档 = 论文 Limitations 收口）

- 执行计划：[docs/PHASE_K_QWEN_PLAN.md](docs/PHASE_K_QWEN_PLAN.md)（**§1.1 任务区分**）
- 模型锁定：`Qwen2.5-1.5B-Instruct`；大文件：`/mnt/data`（结果：`/mnt/data2`）
- [√] K0–K5 管线；KG.5 GLUE 门禁；K6 预算对齐小扫
- [√] **加深恢复 1.5x**：`/mnt/data2/results/qwen_k6_recover_1p5x/` — 剪枝 F1 仍≈0 → **不扩 2x**
- [√] K6-lit 短表：[`docs/K6_LIT_BASELINE_SHORTLIST.md`](docs/K6_LIT_BASELINE_SHORTLIST.md)
- [ ] 论文 Limitations / frozen test（延后）；**不预设** search 全面更优

---

## 下一步行动

1. [√] formal100 全表 + 机制链；叙事定稿为 **regime-dependent**；「系统全面更优」仍 `[×]`
2. [√] 预算对齐、一步复验与 4x 诊断（一步策略收窄为 `target<=2`）
3. [√] Phase J 规划：[`docs/PHASE_J_QWEN_PLAN.md`](docs/PHASE_J_QWEN_PLAN.md)
4. [√] 论文成果提纲：[`docs/PAPER_RESULTS_OUTLINE.md`](docs/PAPER_RESULTS_OUTLINE.md)
5. [√] Phase K SQuAD 管线冒烟：[`docs/PHASE_K_QWEN_PLAN.md`](docs/PHASE_K_QWEN_PLAN.md)（K0–K5）
6. [√] KG.0–KG.5 GLUE；K6 SQuAD 小扫（预算对齐；`/mnt/data2/results/qwen_k6/`）
7. **之后**：加深 SQuAD 恢复 → 再比方法 / frozen test；并行 K6-lit；不默认重跑 formal100

执行顺序与验收细节仍以 [EXECUTION_PLAN.md](EXECUTION_PLAN.md) 与 [docs/P2_EXECUTION_PLAN.md](docs/P2_EXECUTION_PLAN.md) 为准；实验结论以 WORK_LOG 为准；写论文以 PAPER_RESULTS_OUTLINE 为准；LLM 以 PHASE_K_QWEN_PLAN 为准。
