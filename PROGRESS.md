# 项目事实进度

更新日期：2026-08-13

## 当前实现状态

已提交 CPU-first MNIST MLP 自主结构化剪枝 MVP（`e40ca88`）。该 MVP 包含：

- 顺序 MLP 的物理结构化剪枝，以及 BatchNorm 和下游 Linear 的同步更新。
- Cheap Critic 的无副作用、小样本评估。
- 基础启发式控制器和自主搜索循环。
- CPU YAML 配置、JSONL/CSV/summary/checkpoint/图表产物基础设施。
- 面向上述模块的合成数据自动化测试。

历史记录显示提交前运行过 `python -m pytest tests -q`，结果为 `46 passed`。该记录只说明当时提交前的状态；后续修改完成后必须重新验证。

## 当前阻塞与限制

- 搜索计算 Wanda 重要性，但尚未验证其结果实际决定物理剪枝的保留索引。
- 候选池的完整审计、controller 跨 run 状态隔离、rollback 的恢复并停止语义尚未完成。
- Level 1 recovery 尚未保证返回最佳 validation 权重。
- 当前 frontier/proposal 仍存在旧的并行实现，未形成真实 Pareto archive。
- 当前数据加载和实验入口尚未严格分离 train、validation 和 official test。
- 已清理历史 smoke 实验产物；它们不能作为当前可复核结果。

## 正在进行

P0 搜索正确性与审计链路加固。详细工作项和验收条件见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)；长期项目方向见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

当前优先级：

1. Wanda 到显式 `keep_indices` 的可验证数据通路。
2. 单次敏感度遍历、完整候选事件、deterministic shortlist 和 rollback stop。
3. 最佳 validation recovery 与规范 Pareto frontier。

## 验证命令

```bash
python -m pytest tests -q
```

当前 MVP 搜索入口：

```bash
python experiments/run_autonomous_search.py --config configs/mnist_mlp_autonomous_cpu.yaml
```

执行真实 MNIST 实验前，先确认数据集和基线 checkpoint 可用。预计超过五分钟的实验启动后必须记录实验名称、预计时长和输出路径；official test 集不得参与搜索或模型选择。

## 文档关系

- [PROJECT_PLAN.md](PROJECT_PLAN.md)：完整项目总计划、长期路线图和全局目标。
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)：当前执行计划、研究协议和阶段验收。
- [ROADMAP.md](ROADMAP.md)：当前阶段和里程碑导航。
- [MVP_DEVELOPMENT_PLAN.md](MVP_DEVELOPMENT_PLAN.md)：已完成 MVP 的归档。
- [README.md](README.md)：项目入口与实际操作命令。
