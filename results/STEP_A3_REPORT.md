# 步骤 A.3 完成报告

## 任务概述
实现敏感度分析 (Self-Diagnosis)，用于评估每一层对模型性能的重要性

## 实施内容

### 1. 敏感度分析器实现
- **文件**: `src/pruning/sensitivity.py`
- **核心类**: `SensitivityAnalyzer`

### 2. 三种敏感度评估方法

#### (1) Weight Magnitude (权重幅度)
- **计算**: 权重矩阵的 L2 范数
- **速度**: ⚡ 最快（仅读取权重）
- **公式**: `Sensitivity = ||W||_2`
- **优点**: 计算极快，无需数据
- **缺点**: 未考虑激活值，可能不准确

#### (2) Wanda Score (权重×激活)
- **计算**: 权重幅度 × 平均激活幅度
- **速度**: ⚡⚡ 中等（需前向传播）
- **公式**: `Wanda = ||W||_2 × |Activation|`
- **优点**: 考虑实际运行时的激活模式
- **缺点**: 需要数据，稍慢

#### (3) Gradient Sensitivity (梯度敏感度)
- **计算**: 梯度的累积范数
- **速度**: ⚡⚡⚡ 较慢（需反向传播）
- **公式**: `Sensitivity = Σ||∇W||_2`
- **优点**: 反映对损失函数的影响
- **缺点**: 计算成本高

### 3. 实验脚本
- **文件**: `experiments/exp_sensitivity_analysis.py`
- 对比三种方法的敏感度分数
- 基于不同方法进行剪枝对比

## 实验结果

### 1. 层级敏感度排名 (Wanda 方法)

| 排名 | 层名称 | 敏感度分数 | 解释 |
|------|--------|-----------|------|
| 1 | classifier | 12.50 | **最敏感**，输出层 |
| 2 | features.0 | 2.94 | 第一层，输入特征提取 |
| 3 | features.8 | 1.27 | 第三隐藏层 |
| 4 | features.4 | 1.03 | 第二隐藏层 |

**关键发现**:
- 分类器层最敏感（应该保留）
- 浅层（features.0）比深层更敏感
- 中间层（features.4）敏感度最低（可以更激进地剪枝）

### 2. 基于敏感度的剪枝对比 (50% 剪枝)

| 方法 | 剪枝后准确率 | 准确率下降 | 备注 |
|------|------------|-----------|------|
| Magnitude | 68.61% | 29.82% | 仅基于权重大小 |
| Wanda | 59.11% | 39.32% | 考虑激活值 |

**意外发现**:
- 在这个实验中，Magnitude 方法表现更好！
- 这可能因为 MNIST 相对简单，权重大小已经是好的指标
- 在更复杂的任务（Transformer、大模型）上 Wanda 通常更优

### 3. 不同方法的对比总结

| 特性 | Magnitude | Wanda | Gradient |
|------|-----------|-------|----------|
| 计算速度 | ⚡⚡⚡ 极快 | ⚡⚡ 中等 | ⚡ 慢 |
| 是否需要数据 | ✗ 不需要 | ✓ 需要 | ✓ 需要 |
| 考虑激活 | ✗ | ✓ | ✗ |
| 考虑梯度 | ✗ | ✗ | ✓ |
| 内存开销 | 低 | 中 | 高 |
| 推荐场景 | 快速原型 | 通用剪枝 | 精细调优 |

## 技术实现亮点

### 1. 钩子机制捕获激活值
```python
def get_activation(name):
    def hook(model, input, output):
        activations[name].append(output.detach())
    return hook

# 注册前向钩子
hooks.append(module.register_forward_hook(get_activation(name)))
```

### 2. 神经元级别重要性
```python
# 权重幅度: (out_features,)
weight_magnitude = torch.norm(weight, p=2, dim=1)

# 激活幅度: (out_features,)
avg_activation = torch.mean(torch.abs(all_activations), dim=0)

# Wanda 分数
wanda = weight_magnitude * avg_activation
```

### 3. 统一接口设计
```python
# 统一的层级敏感度接口
sensitivity = analyzer.compute_layer_sensitivity(
    dataloader, 
    method='wanda',  # 或 'magnitude', 'gradient'
    num_batches=10
)
```

## 关键发现与洞察

### 1. 敏感度与剪枝策略
- **高敏感度层**: 应该保留或轻度剪枝
  - 分类器层 (classifier)
  - 输入层 (features.0)
- **低敏感度层**: 可以更激进地剪枝
  - 中间隐藏层 (features.4)

### 2. 方法选择建议
- **快速实验**: 使用 Magnitude（无需数据，秒级完成）
- **生产剪枝**: 使用 Wanda（平衡精度与速度）
- **精细调优**: 使用 Gradient（最准确但最慢）

### 3. 与论文对应
- 论文中的 **Self-Diagnosis** 阶段对应我们的敏感度分析
- 用于指导 **Lottery Ticket Proposal** 阶段的候选生成
- 帮助 **Controller** 做出更明智的剪枝决策

## 生成文件

```
src/pruning/
├── __init__.py
├── structured_pruning.py
└── sensitivity.py ✓ 新增

experiments/
└── exp_sensitivity_analysis.py ✓ 新增

results/
└── sensitivity_analysis_results.json ✓ 新增
```

## 验证状态

✓ **三种敏感度方法实现完成**
✓ **神经元级别重要性评估**
✓ **层级敏感度排名**
✓ **基于敏感度的剪枝实验**
✓ **钩子机制正确捕获激活值**

## 与项目计划的对应

| 计划项 | 实现状态 | 备注 |
|--------|---------|------|
| Gradient Sensitivity | ✓ | 梯度范数累积 |
| Weight Magnitude | ✓ | L2 范数 |
| Wanda Score | ✓ | 权重×激活 |
| SparseGPT Score | ✗ | 可选，未实现 |

## 下一步计划

步骤 A.3 ✓ 已完成，**阶段 A (基础组件实现) 全部完成！**

可以继续：

### 选项 1: 阶段 B - 前沿分析与候选生成
- **步骤 B.1**: 实现 Capability Frontier Profiling
- **步骤 B.2**: 实现 Lottery Ticket Proposal
- **步骤 B.3**: 实现 Cheap Critic 评估

### 选项 2: 阶段 C - 恢复策略实现（推荐）
- **步骤 C.1**: Level 0 - 无恢复（已有数据）
- **步骤 C.2**: Level 1 - 简单重建（微调）⭐
- **步骤 C.3**: Level 2 - LoRA 恢复

**建议**: 先实现恢复策略（步骤 C），因为：
1. One-shot pruning 性能太差，急需恢复
2. 可以立即验证剪枝+恢复的效果
3. 为后续端到端流程打下基础

## 代码质量

- ✓ 模块化设计
- ✓ 完整的文档字符串
- ✓ 独立的单元测试
- ✓ 清晰的接口定义
- ✓ 可扩展的架构

## 性能表现

| 操作 | 耗时 | 备注 |
|------|------|------|
| Magnitude | < 1秒 | 仅读取权重 |
| Wanda (10 batches) | ~10秒 | 前向传播 |
| Gradient (20 batches) | ~30秒 | 反向传播 |

**总耗时**: 2-3 分钟（完整实验）

## 备注

- 敏感度分析是自主搜索的核心组件
- 为 Lottery Ticket Proposal 提供决策依据
- 可以动态调整剪枝策略
- 支持未来扩展更多评估方法（如 SparseGPT）
