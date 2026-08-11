# Autonomous Lottery Ticket Discovery - 项目实施计划

## 项目概述

本项目旨在复现和实现《Autonomous Lottery Ticket Discovery》论文中提出的自主彩票发现方法，该方法能够在不依赖预训练父网络的情况下，自主发现高性能的稀疏神经网络子结构。

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
  1. **MNIST**: 简单图像分类 (用于初期验证) ✅ **[CPU 友好，自动下载]**
  2. **CIFAR-10**: 图像分类 ⚠️ **[CPU 可训练但慢，建议 GPU]**
  3. **SQuAD 2.0**: 问答任务 (论文主要基准) 🔥 **[必须 GPU，CPU 不可行]**
  4. **MMLU**: 多任务语言理解 (可选) 🔥 **[必须 GPU]**
- **产出**: 
  - `data/mnist/` 或 `data/cifar10/`
  - `data/squad/` (train.json, dev.json)
  - 数据加载器脚本 `src/utils/data_loader.py`

---

### **阶段 A: 基础组件实现 (预计 7-10 天)**

#### 步骤 A.1: 实现 Dense Baseline 模型
- **任务**: 构建并训练完整的密集神经网络作为性能基准
- **模型选择**:
  - **简单**: 3-5 层 MLP (用于 MNIST) ✅ **[CPU 可运行]**
  - **中等**: ResNet-18/50 (用于 CIFAR-10) ⚠️ **[需要 GPU，CPU 训练极慢 10-100x]**
  - **复杂**: BERT-base 或 Qwen-0.5B (用于 SQuAD) 🔥 **[必须 GPU ≥8GB，CPU 不可行]**
- **实现文件**: `src/models/dense_baseline.py`
- **验证指标**:
  - 记录 baseline 准确率/F1 分数
  - GPU 内存占用
  - 推理延迟
  - 参数量

#### 步骤 A.2: 实现结构化剪枝算子
- **任务**: 实现论文中的结构化剪枝方法
- **实现文件**: `src/pruning/structured_pruning.py`
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
- **任务**: 直接评估剪枝后的子网络
- **用途**: 作为对照基准

#### 步骤 C.2: Level 1 - 仅重建
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
- **任务**: 使用论文提出的 LoRA 方法恢复
- **实现文件**: `src/recovery/lora_recovery.py`
- **硬件需求**: ⚠️ **[建议 GPU，CPU 训练慢 5-20x]**
- **步骤**:
  1. **Frontier Mining**: 从历史搜索中收集前沿数据
     - 存储 {稀疏度, 准确率, 配置} 元组
  2. **LoRA 适配器**: 添加低秩适配层
     ```python
     class LoRALayer(nn.Module):
         def __init__(self, in_features, out_features, rank=8):
             self.lora_A = nn.Parameter(torch.randn(in_features, rank))
             self.lora_B = nn.Parameter(torch.randn(rank, out_features))
     ```
  3. **训练**: 仅训练 LoRA 参数,冻结原始权重

#### 步骤 C.4: Level 3 - Self-Distillation
- **任务**: 使用父网络作为教师进行知识蒸馏
- **实现文件**: `src/recovery/self_distillation.py`
- **硬件需求**: 🔥 **[必须 GPU，需同时加载父子网络，内存需求 2x]**
- **损失函数**:
  ```python
  def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
      """
      组合蒸馏损失和任务损失
      """
      soft_loss = F.kl_div(
          F.log_softmax(student_logits / T, dim=1),
          F.softmax(teacher_logits / T, dim=1),
          reduction='batchmean'
      ) * (T * T)
      
      hard_loss = F.cross_entropy(student_logits, labels)
      
      return alpha * soft_loss + (1 - alpha) * hard_loss
  ```

---

### **阶段 D: 压缩控制器实现 (预计 5-7 天)**

#### 步骤 D.1: 实现简单的启发式控制器
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
- **任务**: 基于历史动态调整决策
- **实现文件**: `src/controller/adaptive_controller.py`
- **特性**:
  - 根据 capability gap 调整剪枝比例
  - 根据 failure mode 调整恢复策略
  - 动态调整搜索预算

#### 步骤 D.3: 实现 LLM-based 控制器 (高级,可选)
- **任务**: 使用语言模型进行推理决策
- **实现文件**: `src/controller/llm_controller.py`
- **步骤**:
  1. 构造 prompt 模板
  2. 调用 OpenAI API 或本地 LLM
  3. 解析 LLM 输出为决策

---

### **阶段 E: 端到端流程集成 (预计 5-7 天)**

#### 步骤 E.1: 实现主搜索循环
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
- **任务**: 为每个模块编写测试
- **测试文件**: `tests/test_*.py`
- **覆盖**:
  - 剪枝操作正确性
  - 敏感度计算准确性
  - 恢复策略收敛性
  - 控制器决策逻辑

#### 步骤 F.2: 小规模验证实验
- **任务**: 在 MNIST 上验证完整流程
- **实验脚本**: `experiments/exp_mnist_baseline.py`
- **验证点**:
  - Dense baseline 达到 >98% 准确率
  - One-shot pruning 到 90% 稀疏度后准确率
  - 自主搜索恢复到 >95% 准确率

#### 步骤 F.3: 复现论文实验 E1
- **任务**: Dense Baseline 对比 (对应论文 Table 1)
- **硬件需求**: 🔥 **[必须 GPU ≥16GB，Qwen-0.5B 模型大，CPU 完全不可行]**
- **实验配置**:
  - Model: Qwen-0.5B
  - Dataset: SQuAD 2.0
  - 对比: Dense vs One-shot Wanda vs Iterative IMP
- **预期结果**:
  - Dense: ~85% F1
  - One-shot 90% sparsity: ~60% F1
  - Our method 90% sparsity: >75% F1

#### 步骤 F.4: 复现论文实验 E2
- **任务**: 热稀疏度曲线 (对应论文 Figure 2)
- **硬件需求**: 🔥 **[必须 GPU ≥16GB，需要多次完整训练]**
- **实验配置**:
  - 扫描稀疏度: 30%, 50%, 70%, 90%, 95%
  - 记录每个稀疏度下的最佳准确率
- **可视化**: 绘制 Pareto frontier

#### 步骤 F.5: 复现论文实验 E3
- **任务**: One-shot vs Iterative 对比
- **验证**: 自主方法优于传统迭代剪枝

#### 步骤 F.6: 消融实验
- **任务**: 验证各组件的贡献
- **对比组**:
  1. 无恢复 (Level 0)
  2. 仅重建 (Level 1)
  3. LoRA 恢复 (Level 2)
  4. 自蒸馏 (Level 3)
  5. 完整方法

---

### **阶段 G: 优化与扩展 (可选,预计 5-7 天)**

#### 步骤 G.1: 性能优化
- **任务**: 加速搜索过程
- **硬件需求**: 🔥 **[必须 GPU，混合精度训练需要 CUDA]**
- **优化点**:
  - 并行化候选评估
  - 缓存中间结果
  - 混合精度训练 (FP16) 🔥 **[需要 GPU with Tensor Cores]**
  - 梯度累积

#### 步骤 G.2: 支持更多模型架构
- **任务**: 扩展到其他模型
- **硬件需求**: 🔥 **[所有大型模型都必须 GPU ≥16GB]**
- **候选**:
  - Vision Transformer (ViT) 🔥 **[GPU ≥12GB]**
  - GPT-2 / LLaMA 🔥 **[GPU ≥16GB]**
  - Diffusion Models 🔥 **[GPU ≥24GB]**

#### 步骤 G.3: 支持更多剪枝策略
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
- **Checkpoint 1**: 阶段 A 完成 → Dense baseline 训练成功
- **Checkpoint 2**: 阶段 C.2 完成 → One-shot pruning + 简单恢复可运行
- **Checkpoint 3**: 阶段 E.1 完成 → 端到端搜索循环可运行
- **Checkpoint 4**: 阶段 F.2 完成 → MNIST 实验验证通过
- **Checkpoint 5**: 阶段 F.3 完成 → 论文主实验复现

### 时间估算
- **最小可行版本 (MVP)**: 阶段 0 + A + B + C.1-C.2 + E → 约 3-4 周
- **完整实现**: 阶段 0-F → 约 6-8 周
- **扩展版本**: 阶段 0-G → 约 8-10 周

### 技术难点预警
1. **Frontier 计算**: 需要大量候选评估,计算成本高 ⚠️ **[GPU 推荐，CPU 耗时 10x+]**
2. **LoRA 恢复**: 超参数敏感,需要细致调参 ⚠️ **[GPU 推荐]**
3. **控制器设计**: 决策逻辑复杂,需多次迭代 ✅ **[CPU 可运行]**
4. **大模型实验**: GPU 内存需求高 🔥 **[必须 GPU ≥24GB VRAM]**

---

## 硬件配置需求总结

### 图例说明
- ✅ **[CPU 可运行]**: 可以在 CPU 上完成，性能可接受
- ⚠️ **[建议 GPU]**: CPU 可运行但速度慢 5-100 倍，强烈建议使用 GPU
- 🔥 **[必须 GPU]**: CPU 环境不可行，必须使用 GPU

### 当前硬件 (CPU only)
**可以完成的任务:**
- ✅ 阶段 0: 环境搭建（已完成）
- ✅ 步骤 A.1: MNIST + MLP 模型训练
- ✅ 步骤 A.2-A.3: 剪枝算子实现与测试
- ✅ 阶段 B: 前沿分析与候选生成（逻辑实现）
- ✅ 步骤 C.1-C.2: 基础恢复策略
- ✅ 阶段 D: 控制器实现（纯逻辑）
- ✅ 步骤 F.1-F.2: 单元测试和 MNIST 小规模验证

**需要 GPU 的任务:**
- ⚠️ CIFAR-10 训练 (CPU 可行但慢)
- 🔥 SQuAD/Qwen 实验 (必须 GPU ≥16GB)
- 🔥 论文实验复现 E1-E3 (必须 GPU ≥16GB)
- 🔥 大规模模型实验 (必须 GPU ≥24GB)

### 建议策略
**当前 CPU 环境:**
1. 完成阶段 0-A（MNIST 验证）
2. 实现所有算法逻辑和控制器
3. 在小规模数据上验证完整流程

**未来 GPU 扩展:**
1. 使用 Google Colab / Kaggle (免费 GPU)
2. 租用云 GPU (AutoDL/恒源云，~1-3元/小时)
3. 本地安装 CUDA (如有 NVIDIA 显卡)

---

## 成功标准

### 最低标准 (MVP)
- [ ] Dense baseline 在 MNIST 上 >98% 准确率
- [ ] One-shot pruning 到 90% 稀疏度可执行
- [ ] 简单恢复策略能提升 5-10% 准确率
- [ ] 端到端搜索循环可运行完整

### 目标标准 (论文复现)
- [ ] 在 SQuAD 上达到论文报告的 F1 分数 (±2%)
- [ ] 稀疏度-准确率曲线与论文 Figure 2 一致
- [ ] 优于 One-shot Wanda baseline 至少 10%
- [ ] 优于传统 IMP 至少 5%

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

### 代码参考
- PyTorch Pruning Tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- Wanda 官方实现: https://github.com/locuslab/wanda
- PEFT (LoRA): https://github.com/huggingface/peft

### 数据集
- MNIST: `torchvision.datasets.MNIST`
- CIFAR-10: `torchvision.datasets.CIFAR10`
- SQuAD 2.0: https://rajpurkar.github.io/SQuAD-explorer/

---

## 下一步行动

请确认:
1. 是否从 **阶段 0** 开始实施?
2. 初始目标模型选择: **MNIST + MLP** 还是直接 **CIFAR-10 + ResNet**?
3. GPU 资源: 本地训练还是云端?
4. 预计完成时间: MVP (4周) 还是完整版 (8周)?

确认后我将开始生成具体的代码框架。
