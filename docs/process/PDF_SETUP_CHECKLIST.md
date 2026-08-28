# PDF 设置核对清单（第一批 · 一次一个 E）

> 来源：`docs/refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf`  
> **重要**：E0 的 **Qwen2.5-3B-Instruct** 与 E1/E2 的 **Qwen2.5-3B**（base，无 Instruct）**不是同一模型**，禁止混用。

## 模型对照

| E | PDF Model 字面 | 磁盘路径（本仓库） |
|---|---------------|-------------------|
| E0 | Qwen2.5-3B-Instruct | `/mnt/data/models/Qwen2.5-3B-Instruct` |
| E1 / E2 | Qwen2.5-3B | `/mnt/data/models/Qwen2.5-3B`（**E1 开跑前需下载**） |

## 共用：六维协议（`mode=scan`，E0/E1/E2+）

| 维 | 任务 | 指标 | 固定 limit |
|----|------|------|------------|
| PPL | wikitext | word_perplexity | 4 |
| Math | gsm8k | exact_match (flexible-extract) | 64 |
| Knowledge | mmlu | acc, 5-shot | 128 |
| Reasoning | bbh | exact_match,get-answer, 3-shot；**max_gen_toks=1024** | 64 |
| Instruction | ifeval | prompt_level_strict_acc | 64 |
| Code | humaneval | pass@1（补全协议） | 32 |

- 实现：[`src/evaluation/capability.py`](../../src/evaluation/capability.py)
- `seed=42`；环境见 [`scripts/env_llm.sh`](../../scripts/env_llm.sh)（含 `HF_ALLOW_CODE_EVAL=1`）
- **E1/E2 eval batch_size=8**（formal batch=8（1024）；探针 `bbh1024_batch8_probe.json` peak≈8.0GB；OOM fallback 8→4→2；剪枝校准仍为 1）；**E0 Instruct 保持 batch=1**
- **禁止混协议**：2048 时代 Reasoning 分数仅作 `pre_bbh1024_*` 归档；Gate A / 正式表只用 1024
- **禁止混模型**：E0 Instruct 分数不可与 E1/E2 base 对比；E1 Delta 只用 base dense

---

## E0 Dense baseline（已完成）

| PDF 项 | 定义 | 状态 |
|--------|------|------|
| Model | Qwen2.5-3B-Instruct | **done** |
| Compression | None | done |
| Evaluation | 六维 scan | done |
| Seeds / GPU | 1 / 1×24GB | done |
| 产物 | P(M0) + 资源行 | [`/mnt/data2/results/E0_dense_baseline/`](/mnt/data2/results/E0_dense_baseline/) |

旧 SST-2 部分向量 **作废**。

**Reasoning 协议补丁**：`experiments/stage_a/run_e0_reasoning_gate.py`（Instruct、only Reasoning、batch=1、max_gen_toks=1024）→ `e0_reasoning_gate_passed.flag`。门禁仅文档一致性，**不参与 E1 Delta / Gate A**。

---

## E1 One-shot sparsity curve（**done** · Reasoning max_gen_toks=1024）

| PDF 项 | 定义 | 本仓库对齐 |
|--------|------|------------|
| Model | Qwen2.5-3B（base） | [`configs/stage_a/e1_oneshot_curve.yaml`](../../configs/stage_a/e1_oneshot_curve.yaml) → `/mnt/data/models/Qwen2.5-3B` |
| Method | Wanda | **已实现**：[`wanda_importance_mlp`](../../src/experiments/qwen_k5_comparison.py) + [`prune_mlp_wanda`](../../src/experiments/stage_a_common.py)；`pruning.method: wanda` |
| Sparsity | 10%, 20%, …, 70% | `sparsity_grid: [0.10 … 0.70]` |
| Recovery | None | `recovery: none` |
| Seeds / GPU | 1 / 1×24GB | `seed: 42`, `cuda:0` |
| Evaluation | capability-specific degradation | 每档 sparsity 跑 **六维 scan**；输出 P(s) 与 **Delta vs base 3B dense** |
| 核心图 | performance vs sparsity | 分维 PNG + Math/Knowledge/Reasoning 合图 |
| 成功条件 | 非线性 / 不同能力不同 cliff | [`detect_capability_cliffs`](../../src/experiments/stage_a_common.py) 启发式 + 报告人工读图 |
| Gate | Gate A（与 E2 合判） | E1 落盘后供 Gate A 输入 |

### E1 协议要点（验收必读）

1. **Dense 参照**：E1 的 Delta 以 **base 3B 未剪枝** 六维向量为基准；**不得**用 E0 Instruct 的 P(M0) 做 E1 Delta。
2. **评测**：与 E0 **相同** scan limit / seed / 任务定义（上表）；Reasoning **max_gen_toks=1024**；eval **batch=8**（formal；OOM→4→2）。
3. **Runner**：[`experiments/stage_a/run_e1_oneshot_curve.py`](../../experiments/stage_a/run_e1_oneshot_curve.py)（Wanda 剪枝 + `eval_capability_vector`；支持 `--smoke` / `--dry-run`）。
4. **Wanda 校准**：SST-2 train LM（`pruning.calibration`）；与六维 eval **分离**。
5. **开跑前检查**：base 权重存在；旧 E1 SST-2 / Instruct / magnitude proxy 产物视为作废。
6. **Code 维**：HumanEval 已改为**补全协议**（`apply_chat_template=False` + `gen_kwargs` 停止符）。正式补跑前必须过 **E0 Code 门禁**（Instruct、scan limit=32、seed=42、`only_dimensions=[Code]`，见 `experiments/stage_a/run_e0_code_gate.py`）。门禁通过后由 `run_e1_code_patch.py` 覆盖 E1 Code：main 的 dense+10–40% 已补；shard 的 dense+10/50/60/70% 须在 **70% 全维曲线结束后**再补（含 70% Code）。`run_e1_code_patch` 必须清空 `skip_dimensions`，否则会与 shard YAML 的 `skip_dimensions: [Code]` 叠加仍跳过 Code。单卡时曲线与 Code patch **不可并行**。Gate A 主看 **PPL / Math / Knowledge / Reasoning**（必要时 Instruction）。
7. **Reasoning 维**：BBH 正式协议为 **max_gen_toks=1024**（校准见 `bbh1024_calib.json`）。补跑：`run_e0_reasoning_gate.py` → `run_e1_reasoning_patch.py`（dense + 全曲线）；dense 须相对校准 \|Δ\|≤0.02（J2）。2048 分数仅 `pre_bbh1024_*` 归档。

### E1 开跑命令（对齐完成后、你确认再执行）

```bash
source venv/bin/activate && source scripts/env_llm.sh
export PYTHONPATH=/root/Model-Compression
# 双卡 3+3 首次启动
bash scripts/launch_e1_dual_after_20.sh
# 中断后续跑（禁止 --fresh）
bash scripts/resume_e1_dual_33.sh
# 单卡仅续 70%（shard partial）
bash scripts/resume_e1_single_70.sh
```

**断点续跑**：每完成一档 sparsity 写入 `e1_checkpoint.json`（主目录 / shard 目录各一份）。中断后运行 `resume_e1_dual_33.sh`；GPU1 有 shard checkpoint 时不再重复 seed。

预计墙钟：双卡 3+3 约 **8–12 小时**（eval batch=4）。

---

## E2 Iterative vs One-shot（**paused → 1024 patch 后 resume · 双卡按 seed 拆**）

| PDF 项 | 本仓库对齐 |
|--------|------------|
| Model | Qwen2.5-3B base → `/mnt/data/models/Qwen2.5-3B` |
| Method | Wanda（同 E1）；`incremental_step_sparsity: 0.05`（累计 +5pp，非固定 0.05 复合） |
| Target sparsity | 40%, 50%, 60% |
| Baseline A / B | One-shot vs 5% incremental；**seed=42 oneshot 从 E1 导入**；43/44 现场重剪 |
| Seeds | `[42, 43, 44]`；eval harness seed **恒 42** |
| Evaluation | 六维 scan（与 E1 `evaluation.capability` 对齐）；**禁止 SST-2 proxy** |
| Recovery | None |
| GPU | **双卡多进程**：GPU0=`42,43`，GPU1=`44`（非模型并行） |
| 断点续传 | `e2_checkpoint.json`：cell `(seed,target,method)` + 维级 partial；默认 `--resume` |
| Config / Runner | [`e2_iterative_vs_oneshot.yaml`](../../configs/stage_a/e2_iterative_vs_oneshot.yaml) / [`run_e2_iterative_vs_oneshot.py`](../../experiments/stage_a/run_e2_iterative_vs_oneshot.py) |
| Launch / Merge | [`launch_e2_dual.sh`](../../scripts/launch_e2_dual.sh) / [`e2_dual_merge_when_done.sh`](../../scripts/e2_dual_merge_when_done.sh) |
| Gate A | 主四维 ≥3/4 且 ≥7/9 cells iterative 胜 |

### E2 开跑命令

```bash
source venv/bin/activate && source scripts/env_llm.sh
export PYTHONPATH=/root/Model-Compression HF_ALLOW_CODE_EVAL=1
# smoke（单卡）
python experiments/stage_a/run_e2_iterative_vs_oneshot.py \
  --config configs/stage_a/e2_iterative_vs_oneshot.yaml --smoke
# formal 双卡（默认 --resume）
bash scripts/launch_e2_dual.sh
bash scripts/e2_dual_merge_when_done.sh
# 强制清空续跑：E2_FRESH=1 bash scripts/launch_e2_dual.sh
```

产物：`/mnt/data2/results/E2_iterative_vs_oneshot/` + `_shard_gpu0/` + `_shard_gpu1/`。

---

## 进度

| E | 状态 | 产物 |
|---|------|------|
| E0 | **done** | `/mnt/data2/results/E0_dense_baseline/` |
| E1 | **done** | `/mnt/data2/results/E1_oneshot_sparsity_curve/` |
| E2 | **paused（待 Reasoning 1024 patch→resume）** | `/mnt/data2/results/E2_iterative_vs_oneshot/` |
| E3+ | pending | — |
