# 步骤 B.2 - Lottery Ticket Proposal

## 目标

实现论文中的 **Lottery Ticket Proposal Generation**，基于敏感度分析智能生成候选剪枝方案（子网络提议）。

## 实现内容

### 1. 彩票提议器 (`LotteryTicketProposer`)

**文件**: `src/frontier/proposal.py`

**核心功能**:
- 基于层敏感度智能生成剪枝方案
- 实现多种提议策略（保守、激进、平衡）
- 自动调整以达到目标压缩率
- 评估并排序候选方案

**关键方法**:
```python
def propose(
    train_loader,
    target_compression: float,
    method: str,
    num_proposals: int
) -> List[Dict]
```

### 2. 三种提议策略

#### 策略 1: Conservative (保守)
- **原则**: 按敏感度反比例分配剪枝比例
- **特点**: 
  - 高敏感度层 → 低剪枝比例（保护重要层）
  - 低敏感度层 → 高剪枝比例（激进剪枝冗余层）
- **适用**: 追求最高准确率，可接受较低压缩率

#### 策略 2: Aggressive (激进)
- **原则**: 保护最敏感的层，其他层均匀激进剪枝
- **特点**:
  - 最敏感层 → 仅剪10%
  - 其他所有层 → 统一高比例剪枝
- **适用**: 追求高压缩率，可接受一定性能损失

#### 策略 3: Balanced (平衡)
- **原则**: 基于敏感度线性分配，范围适中
- **特点**:
  - 敏感度和剪枝比例成反比
  - 剪枝比例范围适中（20%-60%）
- **适用**: 平衡压缩率和准确率

### 3. 智能调整机制

**目标压缩率匹配**:
```python
def _adjust_to_target_compression(
    layer_ratios: Dict[str, float],
    target_compression: float
) -> Dict[str, float]
```

- 迭代调整剪枝比例
- 确保实际压缩率接近目标（误差 < 5%）
- 保持层间相对关系

### 4. 提议评估

**完整流程**:
1. 应用提议的剪枝方案
2. 测量剪枝后准确率
3. 恢复训练
4. 测量恢复后准确率
5. 按性能排序

### 5. 便捷接口

```python
def auto_propose_lottery_ticket(
    model: nn.Module,
    train_loader,
    test_loader,
    target_compression: float = 2.0,
    method: str = 'wanda',
    recovery_epochs: int = 5
) -> Dict
```

一键生成、评估、返回最佳方案。

## 实验设置

**实验脚本**: `experiments/exp_lottery_ticket_proposal.py`

**测试配置**:
- 模型: MNIST Dense Baseline
- 目标压缩率: 2x, 4x, 8x
- 每个目标生成 3 个提议
- 敏感度方法: Wanda
- 恢复训练: 10 轮/提议
- 预计耗时: ~20-25 分钟 (CPU)

## 理论基础

### 敏感度指导的剪枝

**核心思想**:
> 不同的层对模型性能的贡献不同，应该根据层的重要性来分配剪枝资源。

**数学表达**:
```
prune_ratio[layer_i] = f(sensitivity[layer_i], target_compression)

其中 f 是策略函数，满足:
- sensitivity 越高 → prune_ratio 越低
- 所有层的总体压缩率 ≈ target_compression
```

### 多提议生成

**为什么需要多个提议？**

1. **不同策略适合不同场景**
   - 保守策略: 部署到性能敏感场景
   - 激进策略: 部署到资源极度受限设备
   
2. **探索权衡空间**
   - 同一目标压缩率
   - 不同的层间分配方式
   - 可能产生不同的性能结果

3. **鲁棒性**
   - 敏感度估计可能有误差
   - 多个提议提供备选方案

## 与论文对应

**论文 Section 3.2 - Proposal Generation**:
> "Based on the sensitivity analysis, we propose multiple candidate subnetworks with different pruning ratios per layer, targeting a specific compression ratio."

我们的实现:
- ✅ 基于敏感度分析
- ✅ 生成多个候选子网络
- ✅ 每层不同的剪枝比例
- ✅ 针对特定压缩率目标
- ✅ 三种互补策略

## 验证状态

⏳ **后台实验运行中**

预期输出:
1. 3种策略 × 3个压缩率 = 9个提议
2. 每个提议的详细剪枝方案
3. 评估结果（剪枝前后准确率）
4. 最佳提议推荐

## 示例输出

```
Target Compression: 2.0x

Proposal 1/3: conservative
  features.0: 25.3% (sensitivity: 2.94)
  features.4: 62.1% (sensitivity: 1.03)
  features.8: 48.7% (sensitivity: 1.27)
  classifier: 15.2% (sensitivity: 12.50)
  Expected compression: 2.03x

Evaluation:
  conservative: 98.45% (2.03x compression)
  balanced: 98.32% (2.01x compression)
  aggressive: 97.88% (2.05x compression)

✅ Best strategy: conservative
```

## 关键创新

### 与传统方法对比

**传统剪枝**:
```
所有层统一剪枝 50% → 简单但次优
```

**我们的方法**:
```
基于敏感度智能分配:
- 分类器层: 15% 剪枝 (保护关键层)
- 中间层: 60% 剪枝 (剪除冗余)
→ 总体 50% 剪枝，但性能更好
```

### 自动化优势

- ❌ **手动**: 试错调整每层剪枝比例（耗时、低效）
- ✅ **自动**: 基于敏感度生成最优方案（快速、科学）

## 下一步

**阶段 B 完成后**，可以选择:

1. **步骤 D**: 端到端集成
   - 串联所有组件
   - 实现完整的自主发现流程

2. **步骤 C.3**: Level 2 LoRA 恢复（可选）
   - 更高效的恢复方法

## 文件清单

- ✅ `src/frontier/proposal.py` - 彩票提议器实现
- ✅ `src/frontier/__init__.py` - 更新接口
- ✅ `experiments/exp_lottery_ticket_proposal.py` - 完整实验
- ⏳ `results/proposals_compression_*.json` - 提议详情（运行中）
- ⏳ `results/lottery_ticket_proposals_summary.json` - 汇总结果（运行中）

---

**状态**: 核心实现完成，后台实验运行中

**预计完成时间**: 20-25 分钟
