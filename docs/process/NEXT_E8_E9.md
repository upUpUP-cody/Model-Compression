# 下一步：E8 → E9（执行入口 · 可暂停续跑）

> 规格：`proxy_1.5B` · 单卡 · 对照 [PDF_E_REQUIREMENTS_31_34.md](../project/PDF_E_REQUIREMENTS_31_34.md) §34
> 完成后必写：[GATE_RESULTS.md](../project/GATE_RESULTS.md)（尚不存在；跑完再建）
> **进度文件**：`/mnt/data2/results/E8_random_recovery/RUN_PROGRESS.md`、`E9_.../RUN_PROGRESS.md`；镜像 `docs/process/E8_RUN_PROGRESS.md` / `docs/process/E9_RUN_PROGRESS.md`

## 随时暂停 / 下次接着跑

每个 cell 完成后写入 `cells/<strategy>_n{n}_seed{s}_steps{steps}.json`。再次运行**同一命令**会自动 `[SKIP]` 已完成 cell（冒烟 steps=30 与正式 steps=300 **互不覆盖**）。

```bash
# 冒烟：E8 仅 n=256 seed=42 max_steps=30
python experiments/stage_c/run_e8_random_recovery.py \
  --config configs/stage_c/e8_random_recovery.yaml \
  --sizes 256 --seeds 42 --max-steps 30

# 正式 E8 矩阵（可中断后续跑）
python experiments/stage_c/run_e8_random_recovery.py \
  --config configs/stage_c/e8_random_recovery.yaml

# E9（建议 E8 冒烟后再开）
python experiments/stage_c/run_e9_high_gap_recovery.py \
  --config configs/stage_c/e9_high_gap_recovery.yaml
```

强制重跑：删对应 `cells/*.json`，或加 `--no-resume`。

## 目标

1. 实现并跑通 **E8 Random recovery**、**E9 High-Gap / Hard 数据策略**
2. 落盘 `/mnt/data2/results/E8_*`、`E9_*` + `docs/results/`
3. 回答 **Q1 / Q2**，并写 **Gate A–E** 说明文件

## 锁定参数

| 项 | 值 |
|----|-----|
| 模型 | Qwen2.5-1.5B-Instruct（标 `proxy_1.5B`） |
| Child | 50% MLP magnitude 剪枝（与 E3 同代理） |
| LoRA | `max_steps=300`，r=8，α=16，lr=1e-4 |
| E8 | Random；n∈{256,512,1024}；seeds 42/43/44 |
| E9 | 六策略；n=512；steps=300；同 seeds |
| 数据/评测 | SST-2 LM pack；RecoveryGain = Perf(recover)−Perf(compressed) |

## 实现清单

1. [√] `src/recovery/qwen_lora_recovery.py`：`max_steps`
2. [√] `src/experiments/stage_c_common.py`：池上算 G + 六策略 + cell resume
3. [√] `configs/stage_c/e8_*.yaml` + `experiments/stage_c/run_e8_*.py`
4. [√] `configs/stage_c/e9_*.yaml` + `experiments/stage_c/run_e9_*.py`
5. [ ] 单测选样 → 全量 pytest
6. [ ] 串行跑 E8 再 E9
7. [ ] 更新 E-MAP / WORK_LOG / MENTOR_DELIVERY
8. [ ] 撰写 `docs/GATE_RESULTS.md` + Q1/Q2 结论

## Q1 / Q2 作答规则（提醒）

| Q | 主要证据 | 规则 |
|---|----------|------|
| Q1 Frontier？ | E0–E3 | 无可信 cliff → No/Inconclusive；不得因跑了 E8 改口 Yes |
| Q2 diagnose+repair？ | E3+E8/E9 | HighGap≫Random/LowGap 且 Gain>0 → 倾向 Yes；否则 No |

## 参考

- Stage C 公共：`src/experiments/stage_c_common.py`
- LoRA：`src/recovery/qwen_lora_recovery.py`
