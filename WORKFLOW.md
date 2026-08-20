# 开发与实验工作流

当前执行优先级以 [docs/project/EXPERIMENT_E_MAP.md](docs/project/EXPERIMENT_E_MAP.md) 为准；长期边界见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。结构见 [docs/project/STRUCTURE.md](docs/project/STRUCTURE.md)。

**计划语言**：Cursor Plan / 实验方案正文与 todos 用**中文**（路径与 E 号可英文）；见 `.cursor/rules/plans-chinese.mdc`。

## 本地工作区

```bash
python -m pytest tests -q
# Legacy MNIST (archived configs/runners):
python experiments/run_autonomous_search.py --config archive/configs_legacy/mnist_mlp_autonomous_cpu.yaml
```

`experiments/run_*.py` shims load `archive/experiments_legacy/`. Prefer Stage A entrypoints under `experiments/stage_a/`.

## 实验运行规则

- `train` 仅用于训练、敏感度计算和恢复。
- `validation` 用于 Cheap Critic、控制器、frontier 和模型选择。
- `test` 仅用于冻结后最终报告。
- 预计超过五分钟的后台实验启动后，说明实验名称、预计时长和输出路径。
- Python `print()` 仅用 ASCII（Windows GBK）。
- **E 实验产物**：`/mnt/data2/results/E{n}_*/` + PDF 排版 `e{n}_report.md`（见 E_REPORT_TEMPLATE）。

## Git 与文件管理

- 不提交数据集、缓存、大 checkpoint、临时 results。
- 历史文件在 `archive/`（不删）。本地 `.claude/settings.json` 不提交。

## GPU

`python scripts/check_gpu.py --device cuda:0`。旧 GPU 交接见 `archive/archive/docs/GPU_WORKFLOW.md`。

## 验证顺序

1. 相关 pytest → `python -m pytest tests -q`
2. Stage A：`bash scripts/run_stage_a.sh` / `experiments/stage_a/run_e*.py`

## 导师交付与问题上报

- 交付入口：[docs/project/MENTOR_DELIVERY.md](docs/project/MENTOR_DELIVERY.md)；E 进度链到各 `e{n}_report.md`。
- P0 当轮聊天明确说；P1 记 WORK_LOG / MENTOR 已知问题。
