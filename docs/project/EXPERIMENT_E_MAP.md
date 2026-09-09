# Experiment E map（唯一标准 · PDF §31）

强制顺序：**一次一个 E**。  
核对：[../process/PDF_SETUP_CHECKLIST.md](../process/PDF_SETUP_CHECKLIST.md)

**模型**：E0 = **Qwen2.5-3B-Instruct**；E1/E2 = **Qwen2.5-3B**（base，非 Instruct）。二者不得混用。

| E | PDF 目的 | Model（PDF） | 状态 | 产物 |
|---|----------|--------------|------|------|
| E0 | Dense baseline | 3B-Instruct | **done** | `/mnt/data2/results/E0_dense_baseline/` |
| E1 | One-shot curve | 3B base | **done** | `/mnt/data2/results/E1_oneshot_sparsity_curve/` |
| E2 | Iterative vs one-shot | 3B base | **done (Gate A FAIL — 暂缓 Agent)** | `/mnt/data2/results/E2_iterative_vs_oneshot/` |
| E3 | Compression Gap | | pending | — |
| E8 | Random recovery | | pending（E3 后） | — |
| E9 | High-Gap recovery | | pending | — |
| E4–E7 / E10–E14 | | | pending | 禁止插队 |

## Gates（§32）

| Gate | 依赖 | 失败动作 |
|------|------|----------|
| A | E1/E2 | 暂缓 Agent |
| B | E3/E9 | 放弃 Frontier 数据主线 |

## E1 对齐摘要

- **Sparsity**：10%–70%，步长 10%
- **Method**：**Wanda**（MLP intermediate；`||W_gate|| × mean(|h_i|)`；校准 SST-2 train LM）
- **Evaluation**：六维 scan（同 E0）；Delta vs **base 3B dense**（非 E0 Instruct）；Reasoning **max_gen_toks=1024**；eval batch=16（OOM 16→8→4→2→1）
- **Runner**：`experiments/stage_a/run_e1_oneshot_curve.py`
- **Config**：`configs/stage_a/e1_oneshot_curve.yaml`
- **开跑前**：下载 `/mnt/data/models/Qwen2.5-3B`

## E2 对齐摘要

- **Targets**：40% / 50% / 60%；**5% incremental**（累计 +5pp）vs oneshot；Recovery None
- **Seeds**：42 / 43 / 44；seed=42 oneshot **从 E1 导入**；eval harness seed 固定 42
- **Evaluation**：六维 scan（同 E1，含 Reasoning 1024）；Delta vs base 3B dense；eval batch=16
- **GPU**：双卡多进程均衡 — GPU0=`42,44`，GPU1=`43`；`launch_e2_dual_balanced.sh` + `e2_dual_merge_when_done.sh`（旧 `launch_e2_dual.sh` 为 42+43/44）
- **断点续传**：`e2_checkpoint.json`（cell + 维级 partial）；默认 `--resume`
- **Runner / Config**：`experiments/stage_a/run_e2_iterative_vs_oneshot.py` / `configs/stage_a/e2_iterative_vs_oneshot.yaml`

## Soft-eval 旁路（导师菜单）

- **状态**：`_easy`+Sentiment / `_soft` / `_lite` 均 **done**
- **不进 Gate A**；正式 E1/E2 不变；**禁止**把旁路分写回正式六维向量
- **菜单**：[MENTOR_EVAL_MENU.md](MENTOR_EVAL_MENU.md)
- **总报告**：[E1_eval_ladder_all_banks.md](../results/E1_eval_ladder_all_banks.md)
- **Runner**：`experiments/stage_a/run_soft_eval_side.py`
- **产物**：`/mnt/data2/results/E1_soft_eval_side/` · `…/E1_soft_eval_soft/` · `…/E1_soft_eval_lite/`

## 当前步

**E2 已结束**：Gate A passed=False wins=1/9。旁路 soft-eval 供老师选尺子；PDF 下一步 E3 仍受 Gate A 约束。**E0 Instruct 永不进入 Gate A**。
