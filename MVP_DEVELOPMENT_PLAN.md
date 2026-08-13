# CPU-first MNIST MLP MVP 归档

## 归档范围

本文件记录已完成的 MVP 范围，不再作为后续开发任务来源，也不替代项目总计划或当前执行计划。长期项目方向见 [PROJECT_PLAN.md](PROJECT_PLAN.md)；后续任务、研究协议和验收条件以 [EXECUTION_PLAN.md](EXECUTION_PLAN.md) 为准。

- 对应提交：`e40ca88`（Implement CPU autonomous pruning MVP）。
- 目标：建立 CPU-first、MNIST MLP 专用的自主结构化剪枝流程。
- 实现时间：2026-08-12。

## 已交付内容

1. 物理结构化剪枝
   - `src/pruning/structured_pruning.py` 支持顺序 MLP 隐藏 `Linear` 神经元裁剪。
   - 裁剪会同步更新中间 `BatchNorm1d` 和下游 `Linear` 输入维度。
   - 最终 `classifier` 不允许裁剪；剪枝模型可前向、反向传播，并需使用新 optimizer。

2. 搜索基础组件
   - `src/evaluation/cheap_critic.py` 提供严格样本上限、无副作用的快速评估。
   - `src/controller/heuristic_controller.py` 提供基础 `accept`、`reject`、`rollback`、`regrow` 决策。
   - `src/autonomous_search.py` 负责候选深拷贝、快速筛选、Level 1 恢复、完整 validation 与 JSON-safe 历史记录。

3. 配置与产物
   - CPU YAML 配置位于 `configs/mnist_mlp_autonomous_cpu.yaml` 和 smoke 配置。
   - `src/utils/experiment_artifacts.py` 保存 resolved config、JSONL、CSV、summary、checkpoint 和搜索图。
   - `experiments/run_autonomous_search.py` 可直接运行 MVP 入口。

4. 测试
   - 为结构化剪枝、Cheap Critic、控制器、自主搜索和 artifacts 添加了合成数据 pytest。
   - 历史验证：提交前执行 `python -m pytest tests -q`，结果为 `46 passed`。

## 归档边界与已知限制

- 仅支持顺序 MLP；不支持残差、分支、共享层、CNN、Transformer 或 GPU。
- Wanda 分数已被计算，但当前版本尚未保证这些分数驱动实际物理保留索引；实际搜索的正确性加固属于 P0 工作。
- 当前 `regrow` 是降低后续剪枝强度，不是恢复已删除网络结构。
- rollback、候选完整审计、最佳 recovery 权重恢复和规范 Pareto frontier 仍需按主计划实现。
- 历史 smoke 运行目录已清理，不能作为当前可复核结果。
- 不将根目录的临时检查脚本视为 MVP 依赖；此前的 `test_frontier.py` 已清理。

## 持续运行约束

- 所有新增 Python `print()` 必须使用 ASCII 安全文本，避免 Windows GBK 编码问题。
- 预计超过五分钟的后台实验启动后，必须说明实验名称、预计时长和输出文件路径。
