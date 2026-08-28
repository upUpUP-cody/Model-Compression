# Experiment E map（唯一标准 · PDF §31）

强制顺序：**一次一个 E**。  
核对：[../process/PDF_SETUP_CHECKLIST.md](../process/PDF_SETUP_CHECKLIST.md)

**模型**：E0 = **Qwen2.5-3B-Instruct**；E1/E2 = **Qwen2.5-3B**（base，非 Instruct）。二者不得混用。

| E | PDF 目的 | Model（PDF） | 状态 | 产物 |
|---|----------|--------------|------|------|
| E0 | Dense baseline | 3B-Instruct | **done** | `/mnt/data2/results/E0_dense_baseline/` |
| E1 | One-shot curve | 3B base | **done** | `/mnt/data2/results/E1_oneshot_sparsity_curve/` |
| E2 | Iterative vs one-shot | 3B base | **pending patch→resume（1024）** | `/mnt/data2/results/E2_iterative_vs_oneshot/` |
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
- **Evaluation**：六维 scan（同 E0）；Delta vs **base 3B dense**（非 E0 Instruct）；Reasoning **max_gen_toks=1024**；eval batch=4
- **Runner**：`experiments/stage_a/run_e1_oneshot_curve.py`
- **Config**：`configs/stage_a/e1_oneshot_curve.yaml`
- **开跑前**：下载 `/mnt/data/models/Qwen2.5-3B`

## E2 对齐摘要

- **Targets**：40% / 50% / 60%；**5% incremental**（累计 +5pp）vs oneshot；Recovery None
- **Seeds**：42 / 43 / 44；seed=42 oneshot **从 E1 导入**；eval harness seed 固定 42
- **Evaluation**：六维 scan（同 E1，含 Reasoning 1024）；Delta vs base 3B dense
- **GPU**：双卡多进程 — GPU0=`42,43`，GPU1=`44`；`launch_e2_dual.sh` + `e2_dual_merge_when_done.sh`
- **断点续传**：`e2_checkpoint.json`（cell + 维级 partial）；默认 `--resume`
- **Runner / Config**：`experiments/stage_a/run_e2_iterative_vs_oneshot.py` / `configs/stage_a/e2_iterative_vs_oneshot.yaml`

## 当前步

**E1 Reasoning 1024 已齐**（`bbh1024_e1_complete.flag`；正式表禁止 2048）。下一步：E2 Reasoning sync（`run_e2_reasoning_patch`）→ `bbh1024_protocol_locked.flag` → `launch_e2_dual.sh` resume。**E0 Instruct 永不进入 Gate A**。
