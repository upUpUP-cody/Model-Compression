# Compression Harness

压缩闭环（Goal → Plan → Recipe → Execute → Eval → Diagnose → Memory），不是压缩算法工具箱。
Agent / 人通过 **Skill** 调用能力；实现位于 `src/compression_harness/`。

## 本版范围（Phase B MVP + Report）

- **小模型**：`/mnt/data/models/Qwen2.5-3B`（base）
- **近无损**：`max_relative_drop: 0.02`；不强制压缩比
- **真实路径**：torchao INT8 weight-only + hellaswag@64（`acc_norm`）
- Skill：`compress` / `evaluate` / `diagnose` / `recover` / `profile` / `plan-next` / **`report`**
- `auto` 结束自动写报告（逐步时长、bit 步长、压缩比、多维 score 占位表）
- `llm_compressor` / `autoround` / `recover` 仍为 stub；`--dry-run` / `--stub` 可走 Phase A 假跑

## 与 Model-Compression 主仓关系

| | 说明 |
|--|------|
| 可复用 | `src/evaluation/capability.py`、Stage A 经验 |
| 独立 | 产物在 `/mnt/data2/results/harness_experiments/`（`harness/experiments` 符号链接），不改 E1–E9 Gate |
| 环境 | `source venv/bin/activate && source scripts/env_llm.sh` |

## 快速命令

```bash
# Schema
PYTHONPATH=harness/src python harness/scripts/validate_schemas.py

# Dry-run（无 GPU 权重改动）
PYTHONPATH=harness/src python -m compression_harness.cli auto --dry-run

# Phase B 真实自动近无损（约 20-45 分钟，单卡）
PYTHONPATH=harness/src:src python -m compression_harness.cli auto \
  --goal harness/recipes/goal_near_lossless_small.yaml

# 事后报告（不重跑评测）
PYTHONPATH=harness/src:src python -m compression_harness.cli report \
  --run-id run_001_global_int8_torchao

# 持久短步迭代（默认 stub；加 --real 为 Phase II 真插件）
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossless_iter.yaml

# 或
bash harness/scripts/smoke_phase_b.sh
```

## 目录

```text
harness/
  docs/ schemas/ recipes/ skills/
  src/compression_harness/
  scripts/validate_schemas.py  smoke_phase_b.sh
  experiments/   -> /mnt/data2/results/harness_experiments（run_* / iter_* / reports）
```

## 文档索引

- [VISION](docs/VISION.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [RECIPE_DSL](docs/RECIPE_DSL.md)
- [PHASES](docs/PHASES.md) · [SKILLS](docs/SKILLS.md)
