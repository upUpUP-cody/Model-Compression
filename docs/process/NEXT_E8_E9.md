# 下一步：E8 → E9（formal_3B）

> **正式模型**：Qwen2.5-3B-Instruct（`spec: formal_3B`）  
> 旧 1.5B 权重与 proxy E 产物已清除。  
> 对照：[PDF_E_REQUIREMENTS_31_34.md](../project/PDF_E_REQUIREMENTS_31_34.md) §34  
> 完成后写：[GATE_RESULTS.md](../project/GATE_RESULTS.md)

## 一键重跑（E0–E3 + E8）

```bash
# 已在后台可用；手动续跑：
bash scripts/rerun_formal_3b_e0_e8.sh
# 或单独：
python experiments/stage_a/run_e0_dense.py --config configs/stage_a/e0_dense.yaml
python experiments/stage_c/run_e8_random_recovery.py --config configs/stage_c/e8_random_recovery.yaml
python experiments/stage_c/run_e9_high_gap_recovery.py --config configs/stage_c/e9_high_gap_recovery.yaml
```

E8/E9 按 cell 续跑：已完成的 `cells/*_steps300.json` 会 `[SKIP]`。

## 日志

- 总控：`/mnt/data2/results/rerun_formal_3b_master.log`
- 分实验：`rerun_e0.log` … `rerun_e8.log`
- E8 进度：`docs/process/E8_RUN_PROGRESS.md`

## 参数（正式）

| 项 | 值 |
|----|-----|
| 模型 | `/mnt/data/models/Qwen2.5-3B-Instruct` |
| E8 | Random；n∈{256,512,1024}；seeds 42/43/44；LoRA 300 steps |
| E9 | 六策略；n=512；同 seeds / steps |
