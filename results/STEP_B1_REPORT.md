# 步骤 B.1 - Capability Frontier Profiling

## 目标

实现论文中的 **Capability Frontier Profiling**，自动探索模型在不同剪枝比例下的性能边界，找到帕累托前沿（Pareto Frontier）。

## 实现内容

### 1. 前沿分析器 (`FrontierProfiler`)

**文件**: `src/frontier/profiling.py`

**核心功能**:
- 自动探索多个剪枝比例（10%, 20%, ..., 90%）
- 对每个剪枝比例进行恢复训练
- 记录性能-压缩率的前沿点
- 分析前沿特性，找到关键配置点

**关键方法**:
```python
def profile(
    train_loader,
    test_loader,
    prune_ratios: List[float],
    recovery_epochs: int,
    verbose: bool
) -> Dict
```

### 2. 前沿分析

**自动识别的关键点**:

1. **最佳权衡点 (Best Tradeoff)**
   - 最大化效率分数: `准确率 / 压缩率`
   - 平衡性能和压缩的最优选择

2. **最大可接受压缩 (Max Acceptable)**
   - 准确率下降 < 1% 的最大压缩
   - 实用部署的推荐配置

3. **性能拐点 (Knee Point)**
   - 性能下降加速的转折点
   - 超过此点后，压缩代价迅速增大

4. **最高准确率 (Best Accuracy)**
   - 在所有前沿点中准确率最高的配置

5. **最大压缩 (Best Compression)**
   - 压缩率最高的配置及其性能

### 3. 可视化

生成两个关键图表:

- **准确率 vs 剪枝比例**: 显示剪枝前后的性能变化
- **帕累托前沿**: 准确率 vs 压缩率的权衡曲线

### 4. 便捷接口

```python
def find_optimal_pruning_ratio(
    model: nn.Module,
    train_loader,
    test_loader,
    target_accuracy_drop: float = 1.0,
    recovery_epochs: int = 5
) -> Tuple[float, Dict]
```

一键找到最优剪枝配置。

## 实验设置

**实验脚本**: `experiments/exp_frontier_profiling.py`

**测试配置**:
- 模型: MNIST Dense Baseline (98.43% 基准)
- 探索点: 10%, 20%, 30%, ..., 90% (9个点)
- 恢复训练: 每个点 10 轮
- 预计耗时: ~15-20 分钟 (CPU)

## 验证状态

⏳ **后台实验运行中**

预期输出:
1. 每个剪枝比例的详细结果
2. 前沿分析报告
3. 关键配置点推荐
4. 可视化曲线图

## 理论基础

### 帕累托前沿 (Pareto Frontier)

在压缩和性能的多目标优化中，前沿上的点满足:
- 无法在不损失性能的情况下进一步压缩
- 无法在不增加参数的情况下提升性能

### 效率分数

```
Efficiency Score = Accuracy / Compression Ratio
```

平衡性能和压缩的综合指标。

## 与论文对应

**论文 Section 3.1 - Self-Diagnosis**:
> "We profile the model's capability frontier by systematically exploring different pruning ratios and measuring the accuracy-compression trade-off."

我们的实现:
- ✅ 系统探索多个剪枝比例
- ✅ 测量准确率-压缩率权衡
- ✅ 识别关键配置点
- ✅ 可视化前沿曲线

## 下一步

步骤 B.2: **Lottery Ticket Proposal**
- 基于敏感度分析
- 智能生成候选子网络
- 提议最优剪枝方案

## 文件清单

- ✅ `src/frontier/profiling.py` - 前沿分析器实现
- ✅ `src/frontier/__init__.py` - 模块接口
- ✅ `experiments/exp_frontier_profiling.py` - 完整实验
- ⏳ `results/frontier_profiling_results.json` - 实验结果（运行中）
- ⏳ `results/frontier_curve.png` - 可视化曲线（运行中）

---

**状态**: 核心实现完成，后台实验运行中

**预计完成时间**: 15-20 分钟
