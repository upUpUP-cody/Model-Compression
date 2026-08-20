# 开发与实验工作流

本文件只定义可执行的协作、实验和产物管理规则。当前任务优先级、研究协议和验收条件以 [EXECUTION_PLAN.md](EXECUTION_PLAN.md) 为准；长期项目边界见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 本地 CPU 工作区

当前本地工作区用于 CPU-first MNIST MLP 的代码开发、合成数据测试、短时验证和文档维护。

```bash
python -m pytest tests -q
python experiments/run_autonomous_search.py --config configs/mnist_mlp_autonomous_cpu.yaml
```

不要假设任何脚本、数据集或 checkpoint 已存在。真实 MNIST 实验前先确认：

1. 配置文件和输出目录正确。
2. MNIST 数据集可用或允许下载。
3. 基线 checkpoint 与模型配置匹配。
4. train、validation 和 official test 的用途符合项目研究协议。

## 实验运行规则

- `train` 仅用于训练、敏感度计算和恢复。
- `validation` 用于 Cheap Critic、控制器、frontier 和模型选择。
- `test` 仅用于冻结模型与配置后的最终报告，不能参与搜索或恢复。
- 固定并保存 Python、NumPy 和 PyTorch 的随机种子；后续实验还应保存数据切分和基线检查点标识。
- 不将未验证的历史结果或已删除的临时产物写成当前可复核证据。
- 预计超过五分钟的后台实验启动后，必须立即说明实验名称、预计时长和输出文件路径。
- 所有新增 Python `print()` 输出仅使用 ASCII 字符，避免 Windows GBK 编码错误。

## Git 与文件管理

- 代码、配置、测试和项目文档进入版本控制。
- 不提交数据集、缓存、临时日志、临时 `results/` 产物或大模型 checkpoint，除非项目明确决定保留可复核产物。
- 提交前检查 `git status` 和 `git diff`，不要覆盖或回退他人已有修改。
- 不执行破坏性 Git 操作，例如 `git reset --hard` 或 `git checkout --`，除非已获得明确授权。
- 当前 `.claude/settings.json` 是本地用户配置，不应作为项目任务的一部分修改或提交。

## GPU 扩展

P1.2 的 CUDA 基础层已部署。GPU 任务必须在新 NVIDIA 主机上先执行 `python scripts/check_gpu.py --device cuda:0`，再运行 `configs/mnist_p12_gpu_smoke.yaml`，通过冻结后的 `report-test` 后才能运行正式 `configs/mnist_p12_gpu_study.yaml`。完整命令和故障处理见 [docs/GPU_WORKFLOW.md](docs/GPU_WORKFLOW.md)。

在云服务器上运行实验时，使用受控配置和独立输出目录。同步代码前确认当前分支和提交；同步结果前检查结果体积、敏感信息和可复核性。云服务器不会自动执行或停止任务，实验调度和监控必须由实际运行命令、日志和使用者确认。

## 验证顺序

1. 修改模块后先运行对应的 pytest 文件。
2. 阶段完成后运行 `python -m pytest tests -q`。
3. 再运行不下载数据的合成端到端测试。
4. 最后才执行 MNIST smoke 或长实验，并检查 JSON、JSONL、CSV、PNG 和 checkpoint 是否可读取。

## 导师交付表与问题上报

- **交付表唯一入口**：导师要看的结果表只维护 [`docs/MENTOR_DELIVERY.md`](docs/MENTOR_DELIVERY.md)。改数字必须对照产物 JSON 或 [`docs/EVIDENCE_PACK.md`](docs/EVIDENCE_PACK.md)，禁止凭记忆改表。
- **实验问题**：写入 [`docs/WORK_LOG.md`](docs/WORK_LOG.md)「问题/阻塞」；精简结论同步 [`docs/WORK_LOG_BRIEF.md`](docs/WORK_LOG_BRIEF.md)。
- **重大问题当轮上报**：P0（主表数字对不上产物、无法复现、口径与导师口头冲突、要改 RQ 优先级/加卡/换模型）必须在**当轮聊天明确告诉用户**，并记入 MENTOR_DELIVERY「需导师拍板」——不埋在文档里等交付才发现。
- **分级**：P0 立刻说 + 待拍板；P1（负结果、协议不对称、单 seed、已修复的覆盖/超剪）如实记入已知问题；P2（测试/文档小修）日志即可。细则见 MENTOR_DELIVERY §5。
