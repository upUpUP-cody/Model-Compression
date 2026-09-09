# 经典 NLP 评测任务调研（支撑导师菜单）

> 目的：说明各任务测什么、难度、与压缩的关系，以及本仓库 **原版六维** vs **`_easy` / `_soft` / `_lite` / Sentiment** 的差异。
> 正式 Gate / E1–E2 仍以 PDF 六维为准，直至老师在 [`MENTOR_EVAL_MENU.md`](../project/MENTOR_EVAL_MENU.md) 勾选新协议。

## 一页结论：混合难度控制

**维名不变**（仍是 PPL / Math / Knowledge / Reasoning / Instruction / Code），难度分档：

| 档 | 题源策略 | 状态 |
|----|----------|------|
| 正式 | 原题库全协议 | E1 已测；进 Gate / 正式报告 |
| `_easy` | **原题库降难**（limit / shot / 单子集） | soft-eval 已测 |
| `_soft` | **混合**：能降再降；不够则 **换更简单新题库** | 已测；Knowledge/Reasoning 有梯度 |
| `_lite` | Math/Instruction/Code 再降一档（算术 / RTE / 单行补全） | 已测 |

触发换库（实测）：Math / Reasoning / Code 的 `_easy`/`_soft` 仍偏难；`_lite` 后 Math/Instruction 可读。Sentiment（SST-2）作难度锚点。

**禁止**：把 `_easy` / `_soft` / `_lite` 分数写进旧 E1/E2 正式表或 Gate A。

## 总览

| 任务族 | 代表数据集 | 典型指标 | 难度 | 压缩后行为（E1 / soft-eval） |
|--------|------------|----------|------|------------------------------|
| 语言建模 | WikiText | word PPL | 中 | PPL 随稀疏单调变差 |
| 数学推理 | GSM8K → ASDiv → **arithmetic_2da**（`_lite`） | EM / acc | 难→易 | `_lite` dense 0.98 → 20% 0.56 |
| 知识问答 | MMLU → ARC-Easy（`_soft`） | acc | 中–难→易 | MMLU 掉幅缓；`_soft` 有梯度 |
| 综合推理 | BBH → BoolQ（`_soft`） | EM / acc | 难→易 | BoolQ 全程可读 |
| 指令遵循 | IFEval → **GLUE RTE**（`_lite`） | strict/loose / 分类 | 中–难→易 | RTE dense 0.85 → 70% 0.40 |
| 代码 | HumanEval → MBPP → **单行补全**（`_lite`） | pass@1 | 难→易 | `_lite` 低分但非零 |
| 情感分类 | GLUE SST-2 | **分类 acc** | 易 | dense 0.83 → 40% 0.67 → 70% 0.43 |

## 原版六维（PDF / E0–E2）

| 维 | 任务 | 指标 | limit（scan） | 备注 |
|----|------|------|--------------|------|
| PPL | wikitext | word_perplexity | 4 | 越低越好 |
| Math | gsm8k | EM flexible-extract | 64 | 5-shot |
| Knowledge | mmlu | acc | 128 | 5-shot |
| Reasoning | bbh | EM get-answer | 64 | max_gen_toks=1024 |
| Instruction | ifeval | prompt_level_strict_acc | 64 | |
| Code | humaneval | pass@1 | 32 | 补全协议 |

实现：`src/evaluation/capability.py`。

## `_easy` 变体（原库降难 · 已测）

| soft 维 | 相对原版如何变简单 | 指标 | dense→40%→70%（摘要） |
|---------|-------------------|------|------------------------|
| Math_easy | limit **32**，few-shot **3** | EM | 0.63→0.03→0.00 |
| Reasoning_easy | 单子集 `bbh_fewshot_boolean_expressions`；max_gen_toks **256**；limit **32** | EM | 0.75→0.00→0.00 |
| Knowledge_easy | 单科 `mmlu_high_school_psychology`；1-shot；limit **64** | acc | ~0.30 全程平 |
| Instruction_easy | 优先 **loose**；limit **32** | loose/strict | 0.31→0.13→0.16 |
| Code_easy | limit **16** | pass@1 | 0.69→0.13→0.00 |

配置：`configs/stage_a/e1_soft_eval_side.yaml` · 逻辑：`src/experiments/soft_eval_common.py`。

## `_soft` 备选（混合 · SST-2 级目标）

| soft 维 | 题源策略 | 任务 | 规格 | 指标 |
|---------|----------|------|------|------|
| PPL_soft | 原库再降 | wikitext | `max_length=128`；limit **4** | word PPL |
| Math_soft | **新题库** | **asdiv**（计划 SVAMP；本环境无 svamp 任务，用 asdiv） | limit **32**；0-shot | EM / acc |
| Knowledge_soft | **新题库** | **arc_easy** | limit **64**；0-shot | acc |
| Reasoning_soft | **新题库** | **boolq** | limit **64**；0-shot | acc |
| Instruction_soft | 原库再降 | ifeval | loose 优先；limit **16** | loose acc |
| Code_soft | **新题库** | **mbpp** | limit **16**；unsafe code 确认 | pass@1 |

配置：`configs/stage_a/e1_dim_soft_alts.yaml` · 与 `_easy` 分 digest / 分 checkpoint，避免冲掉已测旁路分。

Fallback（任务不可用时）：Math→asdiv（已用）；Reasoning→hellaswag；不静默退回 GSM8K/BBH 全难协议。

## `_lite` 备选（Math / Instruction / Code · 已测）

| soft 维 | 题源策略 | 任务 | 规格 | 指标 | dense→40%→70% |
|---------|----------|------|------|------|----------------|
| Math_lite | **新题库** | **arithmetic_2da_local**（两位数加法） | limit **64**；0-shot | acc | 0.98→0.09→0.05 |
| Instruction_lite | **新题库** | **GLUE RTE** 分类 | max_samples **256** | classification_acc | 0.85→0.22→0.40 |
| Code_lite | **新题库** | **humaneval_single_line_infilling_local** | limit **16** | pass@1 | 0.25→0.25→0.13 |

实现要点：

- 本地 JSONL：`/mnt/data/datasets/lite_banks/` + `configs/lm_eval_tasks/lite/`（规避 datasets≥5 禁用 HF loading script）。
- `capability` 需 `task_include_path`；Wanda 校准 `dataset.task` 仍为 **sst2**，Instruction_lite 评测时强制 `glue_task=rte`。
- 配置：`configs/stage_a/e1_dim_lite_alts.yaml` · 独立 digest / checkpoint。

## Sentiment（难度锚点，非六维替换）

| 项 | 定义 |
|----|------|
| 数据 | GLUE SST-2 validation |
| 指标 | **分类准确率**（positive/negative verbalizer） |
| 实现 | `evaluate_glue_split`（`src/utils/qwen_glue_eval.py`） |
| 不是 | Wanda 校准用的 SST-2 **LM NLL** |

## 与「文本生成」

老师提到可试生成任务。当前优先：Sentiment 锚点 + `_easy` / `_soft` / `_lite` 备选。短续写 PPL 可作为后续附录，不另开第七维。

## 推荐阅读路径

1. GLUE：https://gluebenchmark.com/
2. GSM8K / MMLU / BBH / IFEval / HumanEval / ASDiv / ARC / BoolQ / MBPP / arithmetic / RTE：与配置中 task 名一致
3. 总报告（全部题集 × 剪枝率）：[`docs/results/E1_eval_ladder_all_banks.md`](../results/E1_eval_ladder_all_banks.md)
4. 本仓库历史 Non-E GLUE 过渡：`docs/project/MENTOR_DELIVERY.md`
