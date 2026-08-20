# docs/ 目录说明

按用途三层分开，**不要**把步骤过程/实验结果与项目总览混放。

| 目录 | 放什么 | 例子 |
|------|--------|------|
| [`project/`](project/) | **项目层面**：总进度、交付、工作日志、结构、PDF 要求摘录、报告模板 | E-MAP、WORK_LOG、MENTOR_DELIVERY |
| [`process/`](process/) | **各步骤过程**：执行入口、续跑说明、RUN_PROGRESS | NEXT_E8_E9、E8_RUN_PROGRESS |
| [`results/`](results/) | **各实验 ID 结果镜像**（人类可读报告） | E0_dense_baseline.md … |
| [`refs/`](refs/) | 纲领 PDF（唯一实验标准） | experiment_plan.pdf |

机器可读正式产物仍在：`/mnt/data2/results/E{n}_*/`（不进 Git）。

**入口**：下一步执行 → [`process/NEXT_E8_E9.md`](process/NEXT_E8_E9.md) · 总进度 → [`project/EXPERIMENT_E_MAP.md`](project/EXPERIMENT_E_MAP.md)
