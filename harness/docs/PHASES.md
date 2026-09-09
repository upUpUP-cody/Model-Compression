# 实施阶段

## Phase A — 脚手架（已完成）

- `harness/` 目录、文档、近无损 Goal schema、温和 recipes
- Skill 挂到 `.cursor/skills/`
- 模块与 Adapter stub；`validate_schemas.py` + `dry_run_auto`

## Phase B — MVP（已完成）

冻结约定（单次 `auto`）：Qwen2.5-3B + torchao INT8 + hellaswag@64；`max_relative_drop: 0.02`。

语义：**旧**「进入浮动带即停、不追压缩」。仍可用于一次性 `cli auto`。

## Phase I — 持久短步迭代框架（已完成）

与 Phase B **不同**：

| | Phase B `auto` | Phase I/II `iterate` |
|--|----------------|-------------------|
| 停机 | 合格即停 | 阈值内**继续**短步；**越界回退 last_good 并停** |
| 步长 | 整模 INT8 一轮 | 逐层窗 prune/quantize 短步 |
| 任务档 | near_lossless 2% | **lossless ≤1%** / **lossy_bounded ≤5%** |
| 压缩比 | 不强制 | 不强制（90%/80% 均可） |

- Goal：`goal_lossless_iter.yaml` / `goal_lossy5_iter.yaml`（`max_rounds: 6`）
- 默认 stub 插件：`prune_stub` / `quantize_stub` / `evaluate_stub`
- 状态：默认写 `/mnt/data2/results/harness_experiments/iter_*/state.json`（`harness/experiments` 为同路径符号链接）可 `--resume`
- CLI/Skill：`iterate` / `harness-iterate`

## Phase II — 真短步插件（已落地）

- `--real`：`prune_wanda`（MLP intermediate + 小 Δsparsity）/ `quantize_torchao_layers`（按层 INT8）/ `evaluate_real`（hellaswag@64）
- Accept 时写 `ckpt/rXXX/`（`weights.pt` + `prune_ops` 回放）；越界时 reload last_good
- 大产物盘：`EXPERIMENTS_DIR` → `/mnt/data2/results/harness_experiments`（`HARNESS_EXPERIMENTS_DIR` / `TMPDIR` 见 `scripts/env_llm.sh`）
- 冻结模型：`/mnt/data/models/Qwen2.5-3B`；单卡
- 验收 e2e（单卡）：`iter_20260909T091323Z`（lossy5）与 `iter_20260909T091938Z`（lossless）均 `max_rounds_reached`、6/6 accept、`report.md` + `ckpt/r006` 非空；日志 `/mnt/data2/results/harness_phase_ii_e2e.log`

```bash
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossy5_iter.yaml --real
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossless_iter.yaml --real
```

## Phase III — Diagnoser 驱动下一步

- 层敏感 cheap probe；自动减小步长 / 换层序

## Phase D — 扩展

- recover；跨模型 knowledge；UI
