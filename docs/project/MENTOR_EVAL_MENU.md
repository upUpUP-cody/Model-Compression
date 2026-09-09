# 导师评测菜单（旁路 soft-eval · 不进 Gate A）

> 模型不变：`Qwen2.5-3B` base。正式 E1/E2 六维结果保留。
> 本表请老师勾选以后正式采用哪些任务档；**勾选前不改 Gate / 不重写历史结论**。
> 难度策略 = **混合**：先原库降难（`_easy`）；不够再换简单新题库（`_soft` / `_lite`），**维名不变**。

## 一页结论

- 正式六维上，Math / Reasoning 约在 **30–40%** 稀疏后接近地板（E1）。
- **Sentiment** 在 0–70% 仍有梯度，作简单难度锚点。
- **`_easy`**：Math/Code 仍贴地；**Reasoning_easy 已修协议**（可读缓降）；Knowledge 无梯度。
- **`_soft`**：Knowledge/Reasoning 有梯度；Math/Instruction/Code 仍偏难。
- **`_lite`**：Math→arithmetic_2da、Instruction→RTE、Code→单行补全（已测）。
- 总表：[`docs/results/E1_eval_ladder_all_banks.md`](../results/E1_eval_ladder_all_banks.md)

## 原版六维（E1 已测 · 不重跑）

| 维 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 请勾选 |
|----|-------|-----|-----|-----|-----|-----|-----|------|--------|
| PPL | 11.6777 | 15.0183 | 23.0596 | 29.9586 | 54.6195 | 76.7580 | 129.9627 | 229.9322 | [ ] |
| Math | 0.7031 | 0.4531 | 0.2188 | 0.0312 | 0.0312 | 0.0000 | 0.0156 | 0.0312 | [ ] |
| Knowledge | 0.3357 | 0.2838 | 0.2776 | 0.2551 | 0.2602 | 0.2515 | 0.2529 | 0.2526 | [ ] |
| Reasoning | 0.5220 | 0.3374 | 0.2587 | 0.0556 | 0.0220 | 0.0000 | 0.0069 | 0.0041 | [ ] |
| Instruction | 0.2500 | 0.2031 | 0.2031 | 0.1250 | 0.1406 | 0.0781 | 0.1094 | 0.1562 | [ ] |
| Code | 0.6875 | 0.7188 | 0.5625 | 0.3750 | 0.1562 | 0.0312 | 0.0000 | 0.0000 | [ ] |

## 旁路：Sentiment + `_easy`（已测）

| 维 | 含义 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 请勾选 |
|----|------|-------|-----|-----|-----|-----|-----|-----|------|--------|
| Sentiment | SST-2 **分类 acc** | 0.8281 | 0.8281 | 0.8711 | 0.7734 | 0.6680 | 0.6562 | 0.5625 | 0.4336 | [ ] |
| Math_easy | GSM8K limit32, 3-shot | 0.6250 | 0.3438 | 0.0938 | 0.0625 | 0.0312 | 0.0312 | 0.0000 | 0.0000 | [ ] |
| Reasoning_easy | BBH boolean easy (no-chat gen32 flex)@32 | 0.8438 | 0.8125 | 0.6562 | 0.5000 | 0.5312 | 0.4062 | 0.5625 | 0.3750 | [ ] |
| Knowledge_easy | MMLU psychology 1-shot@64 | 0.2969 | 0.2969 | 0.3125 | 0.2969 | 0.2969 | 0.2656 | 0.3125 | 0.2656 | [ ] |
| Instruction_easy | IFEval loose@32 | 0.3125 | 0.2188 | 0.1250 | 0.0625 | 0.1250 | 0.0938 | 0.0625 | 0.1562 | [ ] |
| Code_easy | HumanEval limit16 | 0.6875 | 0.7500 | 0.3750 | 0.3750 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | [ ] |

## 旁路备选：`_soft`（已测）

| 维 | 题库 / 规格 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 请勾选 |
|----|-------------|-------|-----|-----|-----|-----|-----|-----|------|--------|
| PPL_soft | WikiText max_length=128 | 19.5322 | 26.6179 | 36.9532 | 54.2662 | 83.8635 | 136.5746 | 233.8574 | 552.0293 | [ ] |
| Math_soft | **asdiv** limit32 0-shot | 0.0625 | 0.0625 | 0.0625 | 0.0000 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | [ ] |
| Knowledge_soft | **arc_easy** limit64 0-shot | 0.7344 | 0.7188 | 0.6875 | 0.5781 | 0.5312 | 0.4844 | 0.4062 | 0.3594 | [ ] |
| Reasoning_soft | **boolq** limit64 0-shot | 0.7031 | 0.7344 | 0.7188 | 0.6875 | 0.7500 | 0.7500 | 0.6719 | 0.7031 | [ ] |
| Instruction_soft | IFEval loose limit16 | 0.3750 | 0.3125 | 0.2500 | 0.1250 | 0.1875 | 0.1875 | 0.0625 | 0.1875 | [ ] |
| Code_soft | **mbpp** limit16 | 0.6875 | 0.4375 | 0.3125 | 0.0625 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | [ ] |

## 旁路备选：`_lite`（Math / Instruction / Code · 已测）

| 维 | 题库 / 规格 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 请勾选 |
|----|-------------|-------|-----|-----|-----|-----|-----|-----|------|--------|
| Math_lite | **arithmetic_2da** limit64 0-shot | 0.9844 | 0.8750 | 0.5625 | 0.1250 | 0.0938 | 0.0312 | 0.0156 | 0.0469 | [ ] |
| Instruction_lite | **GLUE RTE** 分类 acc | 0.8514 | 0.7390 | 0.6225 | 0.5703 | 0.2169 | 0.4056 | 0.4016 | 0.4016 | [ ] |
| Code_lite | **humaneval_single_line_infilling**@16 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0625 | 0.1250 | 0.1250 | [ ] |

跑数命令：

```bash
source venv/bin/activate && source scripts/env_llm.sh
export PYTHONPATH=/root/Model-Compression
python experiments/stage_a/run_soft_eval_side.py --config configs/stage_a/e1_soft_eval_side.yaml
python experiments/stage_a/run_soft_eval_side.py --config configs/stage_a/e1_dim_soft_alts.yaml
python experiments/stage_a/run_soft_eval_side.py --config configs/stage_a/e1_dim_lite_alts.yaml
```

## 请老师三选一（或自组合勾选）

1. [ ] **简单阶梯**：Sentiment + Knowledge_soft + Reasoning_soft + Instruction_lite + Math_lite
2. [ ] **混合**：部分原版六维 + 上述旁路
3. [ ] **仍用正式 PDF 全六维**（接受高稀疏贴地）

## 六维协议卫生

见 [`docs/process/E1_SIXDIM_HYGIENE.md`](../process/E1_SIXDIM_HYGIENE.md)。`_easy` / `_soft` / `_lite` 单独成行，不冒充原维。
