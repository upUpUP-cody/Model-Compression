# 项目路线图

[PROJECT_PLAN.md](PROJECT_PLAN.md) 记录完整项目总计划、长期路线图和全局目标；[EXECUTION_PLAN.md](EXECUTION_PLAN.md) 定义当前 P0/P1/P2 的执行顺序、研究协议和阶段验收。当前实现是 CPU-first MNIST MLP MVP，不应将其视为论文复现结果。

## 当前阶段

P0：加固搜索正确性和审计链路。

- 让 Wanda 分数实际决定结构化剪枝保留索引。
- 记录所有候选、统一 controller 生命周期，并实现可验证的 rollback 终止语义。
- 让 Level 1 恢复返回最佳 validation 模型。
- 实现真实参数量基础上的 Pareto frontier，迁移或弃用旧 frontier/proposal 并行逻辑。

## 下一里程碑

P1：建立可复现的 CPU MNIST 研究协议。

- 固定 train/validation split，封存 official test。
- 补充配置、manifest、运行标识和完整候选/前沿产物。
- 完成 smoke、单 seed 调试和至少三个 seed 的固定预算对照研究。

## 后续边界

P2：仅在 P0/P1 验收后进入 GPU 与架构扩展。

- 先支持 CIFAR-10 与小型 ResNet，并为 CNN 设计显式结构依赖。
- 再加入 GPU 指标、LoRA 和 self-distillation。
- 最后才规划 Qwen/SQuAD 及论文级实验。

## 文档入口

- [完整项目总计划](PROJECT_PLAN.md)：长期路线图和全局目标。
- [当前执行计划](EXECUTION_PLAN.md)：P0/P1/P2 执行顺序、研究协议和阶段验收。
- [项目入口](README.md)：环境、实际命令和当前范围。
- [事实进度](PROGRESS.md)：已经验证的内容、阻塞和当前任务。
- [MVP 归档](MVP_DEVELOPMENT_PLAN.md)：提交 `e40ca88` 的已交付范围。
- [工作流](WORKFLOW.md)：开发、实验和产物管理规则。
