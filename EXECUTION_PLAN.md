# Autonomous Lottery Ticket Discovery 当前执行计划

> 本文件是当前 P0/P1/P2 阶段的执行顺序、研究协议和验收条件。长期整体路线图、阶段目标和全局成功标准见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。`MVP_DEVELOPMENT_PLAN.md` 仅记录已完成 MVP 的归档范围；`ROADMAP.md`、`PROGRESS.md`、`README.md` 和 `WORKFLOW.md` 只提供导航、事实状态或操作规则。

## 当前状态

- 已完成：CPU-first MNIST MLP 自主结构化剪枝 MVP，提交为 `e40ca88`。
- 已完成：物理结构化剪枝、Cheap Critic、基础启发式控制器、自主搜索循环、CPU/GPU YAML 配置、实验产物记录和合成数据测试。
- 已完成：MNIST P1.2 GPU smoke/formal study（RTX 4090）、多 seed 对照与 6 档压缩率扫描聚合。
- **进行中：P2 CIFAR-10 + ResNet GPU 扩展**（详见 [docs/P2_EXECUTION_PLAN.md](docs/P2_EXECUTION_PLAN.md)）。
- 历史验证：提交前曾运行 `python -m pytest tests -q`；P2 代码交付后需重新全量验证。
- 当前边界：MNIST 路径保持 MLP-only；CIFAR/ResNet 结构化剪枝、P1.2 协议、LoRA/自蒸馏恢复已在 P2 分支实现中。
- Qwen/SQuAD 论文实验仍为规划项，未启动实现。

## 研究协议

- `train`：仅用于训练、敏感度计算和恢复。
- `validation`：仅用于 Cheap Critic、控制器决策、frontier、模型选择和恢复过程中的最佳模型选择。
- `test`：仅用于冻结配置和模型后的最终报告；不得参与搜索、候选选择、恢复或超参数选择。
- 所有实验固定并记录 Python、NumPy 和 PyTorch 随机种子；CPU 实验还应记录线程设置、数据切分和基线检查点标识。
- 所有新增 Python `print()` 必须仅使用 ASCII 字符，避免 Windows GBK 编码错误。
- 启动预计超过五分钟的后台实验后，必须立即报告实验名称、预计时长和精确输出路径。
- 临时 `results/` 产物和本地数据集不提交到 Git，除非项目另行明确要求保留可复核结果。

## 已完成 MVP

### 结构化剪枝

`src/pruning/structured_pruning.py` 已提供顺序 MLP 的物理神经元剪枝：裁剪隐藏 `Linear` 输出后，同步更新中间 `BatchNorm1d` 和下游 `Linear` 输入维度。实现会保留 dtype、device、训练状态和参数梯度要求，并禁止裁剪最终 `classifier`。

### 评估、控制和搜索

- `src/evaluation/cheap_critic.py`：严格样本上限、`torch.inference_mode()`、加权 loss/accuracy、参数统计和模型状态保护。
- `src/controller/heuristic_controller.py`：基础 `accept`、`reject`、`rollback`、`regrow` 决策。
- `src/autonomous_search.py`：候选深拷贝、Cheap Critic、Level 1 恢复、完整验证和 JSON-safe 搜索历史。
- `src/utils/experiment_artifacts.py`：解析配置、JSONL、CSV、summary、checkpoint 和搜索图的基础产物记录。

这些模块构成可运行 MVP，不构成 Wanda 驱动搜索有效性或论文结果的证据。

## P0：搜索正确性与审计链路（已完成）

> 以下条目已在 MNIST MLP 上验收通过；CIFAR 迁移见 P2.3。

### P0.1 Wanda 到物理保留索引

涉及文件：`src/autonomous_search.py`、`src/pruning/structured_pruning.py`、`src/pruning/sensitivity.py`、`tests/test_autonomous_search.py`、`tests/test_structured_pruning.py`、新增 `tests/test_sensitivity.py`。

1. 定义不可变、JSON-safe 的 `CandidateSpec`，记录层比例、每层 `keep_indices`、重要性方法、fingerprint 和实际结构参数量。
2. 每轮仅遍历一次训练数据，计算所有隐藏 `Linear` 的 Wanda 神经元分数。
3. 按稳定排序由 Wanda 分数生成 `keep_indices`，低分神经元优先删除；搜索必须将这些索引显式传给结构化剪枝，不得静默回退到 L2 权重范数。
4. 将敏感度计算改为流式累计激活统计；使用 `try/finally` 清理 hook，并恢复调用前的训练模式。
5. 对空 dataloader、无隐藏 Linear、无效 batch 数和极端剪枝比例给出明确错误或最小保留策略。
6. 在事件中记录 candidate spec、索引摘要或哈希、实际参数量、压缩比和重要性方法。

验收：构造 Wanda 与 L2 排序相反的合成模型时，实际保留索引严格遵循 Wanda；一次搜索轮只执行一次敏感度数据流；hook 无泄漏，模型模式保持不变。

### P0.2 候选预算与控制器状态机

涉及文件：`src/autonomous_search.py`、`src/controller/heuristic_controller.py`、`tests/test_autonomous_search.py`、`tests/test_heuristic_controller.py`。

1. `candidates_per_round` 作为真实候选预算；支持确定性单层候选和受限双层组合，并按实际物理结构参数量筛除无收益或重复候选。
2. 对所有候选运行 Cheap Critic，并记录 fingerprint、参数量、指标和淘汰原因；不能只记录最终胜者。
3. 使用确定性 shortlist 排序：真实参数下降、Cheap Critic loss、压缩收益、fingerprint。CPU MVP 默认仅恢复 top-1，并在配置中预留 top-k。
4. 将已尝试 fingerprint 和连续失败计数限制在单次 run 的 `SearchHistory` 或 `SearchState` 中；同一 controller 执行第二次 run 不得继承第一次的候选状态。
5. 将 MVP 中的 `regrow` 明确记录为“降低下一轮剪枝强度”，不宣称已删除结构被恢复。
6. `rollback` 必须恢复最近一次 accepted 的 CPU `state_dict` 和结构元数据，记录终止原因，并立刻结束 run。

验收：全部 Cheap Critic 候选均出现在历史中；连续两个 run 不发生错误重复拒绝；rollback 后模型结构、权重和参数量等于最近 accepted 快照，且 run 停止。

### P0.3 Level 1 恢复返回最佳验证模型

涉及文件：`src/recovery/reconstruction.py`、`src/autonomous_search.py`、新增 `tests/test_reconstruction.py`。

1. 将恢复接口中的评估数据命名为 `validation_loader`，禁止使用 official test 集。
2. 验证指标改善时保存 CPU `state_dict`、best epoch 和 best validation accuracy；训练结束后加载最佳权重再返回。
3. 使用语义明确的历史字段；过渡期可兼容旧字段，但新事件只写 validation 语义。
4. 覆盖零 epoch、空 loader、最佳 epoch 非最后 epoch、真实参数更新和父模型隔离。

## P0：规范 Pareto Frontier

涉及文件：新增或重构 `src/evaluation/frontier.py`、`src/evaluation/__init__.py`、`src/autonomous_search.py`、`src/frontier/profiling.py`、`src/frontier/proposal.py`、相关实验入口和新增 `tests/test_frontier.py`。

1. 定义 `FrontierPoint`：validation accuracy、实际参数量、压缩比、恢复耗时、candidate spec、seed、run 和 iteration。
2. 使用 Pareto 支配：A 的 validation accuracy 不低于 B、参数量不高于 B，且至少一个指标严格更优时，A 支配 B。
3. 为所有完成完整 validation 的候选更新 frontier archive；提供“固定 accuracy drop 下最大压缩”和“固定参数预算下最高 accuracy”查询。
4. 以真实结构剪枝后的参数量为准，废弃 `accuracy / compression_ratio` 作为所谓最佳权衡指标，废弃旧 proposal 的独立参数估算和原地剪枝流程。
5. 旧 `src/frontier/` 模块迁移前只保留兼容导入和明确弃用说明；新实验入口统一使用配置、随机种子和 artifacts。

验收：覆盖支配、重复、同参数、单点、空点和非排序输入；frontier archive 可 JSON 序列化。

## P1：可复现 CPU/GPU MNIST 实验协议（已完成）

> MNIST P1.2 GPU 统计补全（multiseed + sweep + aggregate）已完成。CIFAR 对照见 P2.4。

完成门槛：每次结果均可追溯到数据切分、配置、基线、候选历史、frontier 和最终一次 test 评估。

### P1.1 数据、配置和产物

涉及文件：`src/utils/data_loader.py`、`src/utils/experiment_artifacts.py`、`configs/mnist_mlp_autonomous_cpu.yaml`、smoke 配置、`experiments/exp_mnist_baseline.py`、`experiments/run_autonomous_search.py` 和新增测试。

1. 为 MNIST 提供固定 seed 的 train/validation split，并单独返回封存的 official test loader。
2. 扩展配置：split、seed 或 seed 列表、baseline checkpoint 标识、敏感度预算、候选预算、recovery top-k、目标参数或压缩预算、接受阈值、运行标签和设备策略。
3. 将 CPU-only 检查限制在 CPU profile；通用配置 schema 预留 GPU 字段，但不提前实现未测试的 GPU 执行路径。
4. 产物 manifest 记录 resolved config、git SHA、平台和依赖、seed、split hash、checkpoint hash、完整命令、候选事件、frontier JSON/CSV、最终 checkpoint 和 test report。
5. 使用 config hash 或稳定 run id，避免仅依赖秒级时间戳；summary 必须包含停止原因和“无候选接受”状态。
6. 记录结构参数量、序列化模型字节数、固定 CPU 线程数下预热后的推理延迟和吞吐，以及候选评估和恢复耗时。

### P1.2 CPU 对照研究

1. 先运行一轮 smoke，核对 checkpoint 复载、artifact 完整性和 test 隔离。
2. 再执行单 seed 调试实验，最后运行至少三个 seed 的固定预算实验。
3. 在相同 train split、初始化策略和恢复预算下比较：dense baseline、同尺寸 dense 小模型、one-shot magnitude、one-shot Wanda、迭代结构化剪枝加 Level 1，以及 autonomous search。
4. 按实际结构压缩目标扫描约 1.5x、2x、4x、6x、8x、10x，报告每 seed 结果与 mean/std，并保留正负结果。

验收：同 seed 的 split 和关键 smoke 输出可复现；test 不参与搜索；三 seed 汇总可完整追溯。

## P2：GPU 与架构扩展（进行中）

> 详细步骤、命令与时长估算见 [docs/P2_EXECUTION_PLAN.md](docs/P2_EXECUTION_PLAN.md)

### P2.1 CIFAR-10 数据与 ResNet 基线 [进行中]
### P2.2 CNN 结构化剪枝 [进行中]
### P2.3 Wanda/搜索 CNN 迁移 [进行中]
### P2.4 CIFAR P1.2 GPU 协议 [进行中]
### P2.5 GPU 吞吐/显存/并行 [进行中]
### P2.6 LoRA 恢复 [进行中]
### P2.7 自蒸馏恢复 [进行中]
### P2.8 恢复消融 [进行中]
### P2.9 Qwen/SQuAD [planned only]

## 执行顺序与全局验收

1. 完成 P2 文档与 CIFAR 数据/ResNet 基线（P2.1）。
2. 完成 CNN 结构化剪枝与单测（P2.2）。
3. 完成 Wanda/search backend 迁移，确保 MNIST 测试不退化（P2.3）。
4. 运行 CIFAR P1.2 smoke → formal → multiseed → sweep（P2.4）。
5. 写入 manifest GPU benchmark 指标（P2.5）。
6. CIFAR 协议稳定后，完成 LoRA/自蒸馏恢复与消融（P2.6–P2.8）。
7. Qwen/SQuAD 仅作规划占位（P2.9）。

每个功能修改后运行对应 pytest 文件；阶段结束运行 `python -m pytest tests -q`。任何真实 MNIST 长实验开始前，先确认数据集、基线检查点、配置和输出目录，并按本文件的运行约束向用户报告预计时长和产物路径。
