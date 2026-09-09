# Skills 映射

Skill 是对外调用面；实现在 `src/compression_harness/`。权威副本在 `harness/skills/`，Cursor 发现路径为 `.cursor/skills/harness-<name>/`。

| Skill | 触发词（示例） | 模块 | 状态 |
|-------|----------------|------|------|
| compress | 压模型、跑 recipe、quantize INT8 | `executor` + torchao | Phase B 真实 INT8 |
| evaluate | 评这个 run、相对 dense 掉点 | `evaluator` | 真实 / dry-run |
| diagnose | 为什么掉点、层敏感 | `diagnoser` | stub |
| recover | LoRA 救一下 | （Phase D） | 契约 only |
| profile | compressibility profile | `actions.profile` | stub |
| plan-next | 下一步实验 | `planner` / `controller` | 单次 auto |
| **iterate** | 短步迭代、逐层剪枝量化、掉点回退 | `iterative_controller` + plugins | **Phase II `--real`** |
| report | 压缩报告、逐步时长、多维 score | `reporter` | 已接（含 round/plugin/layers） |

## 迭代语义（重要）

- **lossless**：`max_relative_drop=0.01`，阈值内继续短步
- **lossy_bounded**：`max_relative_drop=0.05`，**>5% 立即回退 last_good 并停**
- 不设压缩比硬 KPI；优先保性能
- 默认 stub；加 `--real` 走 Wanda / torchao 分层 + 真评测
- 产物默认：`/mnt/data2/results/harness_experiments/`（可用 `harness/experiments` 符号链接；`source scripts/env_llm.sh`）

## 调用约定

```bash
source venv/bin/activate && source scripts/env_llm.sh
# stub（快速）
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossless_iter.yaml
# 真权重短步（Phase II）
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossy5_iter.yaml --real
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossy5_iter.yaml --real --resume iter_...
```
