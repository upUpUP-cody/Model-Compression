# Model Compression

面向 MNIST MLP 的 CPU-first 自主结构化剪枝研究原型。当前实现聚焦顺序 MLP：生成结构化剪枝候选、快速评估、Level 1 恢复、validation 筛选和实验产物记录。

当前 MVP 已提交为 `e40ca88`。它是可运行的工程骨架，不是论文复现结果；Wanda 到实际保留索引、规范 Pareto frontier 和可复现实验协议仍在加固。

## 当前支持范围

- CPU-first MNIST MLP with an opt-in CUDA execution layer for the P1.2 comparison protocol.
- 隐藏 `Linear` 神经元的物理结构化剪枝。
- 同步更新中间 `BatchNorm1d` 和下游 `Linear`。
- Cheap Critic、基础启发式控制器、Level 1 recovery 和 JSON-safe 产物记录。
- 合成数据单元测试。

不支持残差网络、分支或共享层、CNN、Transformer、LoRA 和自蒸馏。GPU 基础层已部署，但 CUDA smoke 和正式实验必须在 GPU 主机完成。

## 环境

建议 Python 3.10+，并安装项目实际使用的依赖：

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision numpy pandas matplotlib pyyaml pytest
```

MNIST 数据会由数据加载逻辑在需要时下载。真实实验前请确认网络、数据目录和基线 checkpoint 已准备完成。

## 验证

运行自动化测试：

```bash
python -m pytest tests -q
```

历史提交前结果为 `46 passed`，修改代码后应以当前命令的结果为准。

## 运行 CPU MVP

```bash
python experiments/run_autonomous_search.py --config configs/mnist_mlp_autonomous_cpu.yaml
```

默认入口需要 `checkpoints/mnist_dense_baseline.pth` 和 MNIST 数据。产物写入配置中的 `logging.output_root`，包括 resolved config、JSONL 事件、CSV、summary、checkpoint 和可用时的搜索图。不要提交临时 `results/` 产物。

运行预计超过五分钟的实验时，需要先记录实验名称、预计时长和输出目录。搜索、恢复和模型选择只能使用 train/validation；official test 集仅用于冻结模型后的最终报告。

## 文档

- [PROJECT_PLAN.md](PROJECT_PLAN.md)：完整项目总计划、长期路线图和全局目标。
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)：当前 P0/P1/P2 执行计划、研究协议和阶段验收。
- [PROGRESS.md](PROGRESS.md)：事实进度与当前阻塞。
- [ROADMAP.md](ROADMAP.md)：简要阶段导航。
- [MVP_DEVELOPMENT_PLAN.md](MVP_DEVELOPMENT_PLAN.md)：MVP 提交归档。
- [WORKFLOW.md](WORKFLOW.md)：开发、实验和产物管理规则。
- [docs/GPU_WORKFLOW.md](docs/GPU_WORKFLOW.md)：CUDA 主机安装、自检、smoke、冻结报告和正式 study 步骤。

## 代码质量约束

- 为行为变更补充聚焦的 pytest。
- 所有新增 Python `print()` 输出必须只使用 ASCII 字符，避免 Windows GBK 编码错误。
- 预计超过五分钟的后台实验必须在启动后报告名称、时长和输出路径。
