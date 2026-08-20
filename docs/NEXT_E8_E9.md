# 下一步：E8 → E9（明早执行入口）

> 规格：`proxy_1.5B` · 单卡 · 对照 [PDF_E_REQUIREMENTS_31_34.md](PDF_E_REQUIREMENTS_31_34.md) §34
> 完成后必写：[GATE_RESULTS.md](GATE_RESULTS.md)（尚不存在；跑完再建）

## 目标

1. 实现并跑通 **E8 Random recovery**、**E9 High-Gap / Hard 数据策略**
2. 落盘 `/mnt/data2/results/E8_*`、`E9_*` + `docs/e_reports/`
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

1. `src/recovery/qwen_lora_recovery.py`：加 `max_steps`
2. `src/experiments/stage_c_common.py`：池上算 G + 六策略选样
3. `configs/stage_c/e8_*.yaml` + `experiments/stage_c/run_e8_*.py`
4. `configs/stage_c/e9_*.yaml` + `experiments/stage_c/run_e9_*.py`
5. 单测选样 / max_steps → 全量 pytest
6. 串行跑 E8 再 E9（预计约 3–6 h；仍单卡即可）
7. 更新 E-MAP / WORK_LOG / MENTOR_DELIVERY
8. 撰写 `docs/GATE_RESULTS.md` + Q1/Q2 结论

## Q1 / Q2 作答规则（提醒）

| Q | 主要证据 | 规则 |
|---|----------|------|
| Q1 Frontier？ | E0–E3 | 无可信 cliff / capability-specific / iterative 优势 → No/Inconclusive；**不得**因跑了 E8 改口 Yes |
| Q2 diagnose+repair？ | E3+E8/E9 | HighGap≫Random/LowGap 且 Gain>0 → 倾向 Yes；否则 No |

两问都是 Yes 才值得投入 AdaptiveTicketSearch → Synthetic → SelfController。

## 参考

- 详细 Cursor Plan：本机 `.cursor/plans/e8_e9_与双问_*.plan.md`（若有）
- Stage A 公共代码：`src/experiments/stage_a_common.py`
- LoRA：`src/recovery/qwen_lora_recovery.py`
- E3 Gap：`experiments/stage_a/run_e3_compression_gap.py`
