# E1 评测阶梯总表：全部题集 × 剪枝率

> **不覆盖 Gate A / 正式 E1 结论。** 旁路 `_easy` / Sentiment / `_soft` / `_lite` 仅供导师选尺子。
>
> 模型：Qwen2.5-3B base · 剪枝：Wanda oneshot · Recovery：None · 仍单卡即可。

## 策略摘要

| 档 | 含义 |
|----|------|
| 正式六维 | PDF / E1 原题库全协议 |
| Sentiment + `_easy` | 原库降难 + SST-2 分类锚点（已测） |
| `_soft` | 混合换库；Knowledge/Reasoning 有梯度；Math/Instruction/Code 仍偏难 |
| `_lite` | Math→arithmetic_2da、Instruction→RTE、Code→单行补全（已测） |

说明：`arithmetic_2da` / `humaneval_single_line_infilling` 使用本地 JSONL 任务（`configs/lm_eval_tasks/lite/`），规避 datasets≥5 禁用 loading script。

### 一、正式六维（E1 已测）

| 题集 | 题库 / 规格 | 指标 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|------|------------|------|------|------|------|------|------|------|------|------|
| PPL | wikitext | word_perplexity | 11.6777 | 15.0183 | 23.0596 | 29.9586 | 54.6195 | 76.7580 | 129.9627 | 229.9322 |
| Math | gsm8k | exact_match | 0.7031 | 0.4531 | 0.2188 | 0.0312 | 0.0312 | 0.0000 | 0.0156 | 0.0312 |
| Knowledge | mmlu | acc | 0.3357 | 0.2838 | 0.2776 | 0.2551 | 0.2602 | 0.2515 | 0.2529 | 0.2526 |
| Reasoning | bbh | exact_match | 0.5220 | 0.3374 | 0.2587 | 0.0556 | 0.0220 | 0.0000 | 0.0069 | 0.0041 |
| Instruction | ifeval | prompt_level_strict_acc | 0.2500 | 0.2031 | 0.2031 | 0.1250 | 0.1406 | 0.0781 | 0.1094 | 0.1562 |
| Code | humaneval | pass@1 | 0.6875 | 0.7188 | 0.5625 | 0.3750 | 0.1562 | 0.0312 | 0.0000 | 0.0000 |

### 二、旁路 Sentiment + `_easy`（已测）

| 题集 | 题库 / 规格 | 指标 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|------|------------|------|------|------|------|------|------|------|------|------|
| Sentiment | GLUE SST-2 | classification_acc | 0.8281 | 0.8281 | 0.8711 | 0.7734 | 0.6680 | 0.6562 | 0.5625 | 0.4336 |
| Math_easy | gsm8k limit32 3-shot | exact_match | 0.6250 | 0.3438 | 0.0938 | 0.0625 | 0.0312 | 0.0312 | 0.0000 | 0.0000 |
| Reasoning_easy | bbh boolean easy (no-chat, gen32, flexible-extract)@32 | exact_match | 0.8438 | 0.8125 | 0.6562 | 0.5000 | 0.5312 | 0.4062 | 0.5625 | 0.3750 |
| Knowledge_easy | mmlu_high_school_psychology 1-shot@64 | acc | 0.2969 | 0.2969 | 0.3125 | 0.2969 | 0.2969 | 0.2656 | 0.3125 | 0.2656 |
| Instruction_easy | ifeval loose@32 | loose_acc | 0.3125 | 0.2188 | 0.1250 | 0.0625 | 0.1250 | 0.0938 | 0.0625 | 0.1562 |
| Code_easy | humaneval@16 | pass@1 | 0.6875 | 0.7500 | 0.3750 | 0.3750 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |

### 三、备选 `_soft`（已测）

| 题集 | 题库 / 规格 | 指标 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|------|------------|------|------|------|------|------|------|------|------|------|
| PPL_soft | wikitext max_length=128 | word_perplexity | 19.5322 | 26.6179 | 36.9532 | 54.2662 | 83.8635 | 136.5746 | 233.8574 | 552.0293 |
| Math_soft | asdiv limit32 0-shot | acc | 0.0625 | 0.0625 | 0.0625 | 0.0000 | 0.0312 | 0.0000 | 0.0000 | 0.0000 |
| Knowledge_soft | arc_easy limit64 0-shot | acc | 0.7344 | 0.7188 | 0.6875 | 0.5781 | 0.5312 | 0.4844 | 0.4062 | 0.3594 |
| Reasoning_soft | boolq limit64 0-shot | acc | 0.7031 | 0.7344 | 0.7188 | 0.6875 | 0.7500 | 0.7500 | 0.6719 | 0.7031 |
| Instruction_soft | ifeval loose@16 | loose_acc | 0.3750 | 0.3125 | 0.2500 | 0.1250 | 0.1875 | 0.1875 | 0.0625 | 0.1875 |
| Code_soft | mbpp limit16 | pass@1 | 0.6875 | 0.4375 | 0.3125 | 0.0625 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |

### 四、备选 `_lite`（已测）

| 题集 | 题库 / 规格 | 指标 | dense | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|------|------------|------|------|------|------|------|------|------|------|------|
| Math_lite | arithmetic_2da_local (2-digit add) limit64 | acc | 0.9844 | 0.8750 | 0.5625 | 0.1250 | 0.0938 | 0.0312 | 0.0156 | 0.0469 |
| Instruction_lite | GLUE RTE classification | classification_acc | 0.8514 | 0.7390 | 0.6225 | 0.5703 | 0.2169 | 0.4056 | 0.4016 | 0.4016 |
| Code_lite | humaneval_single_line_infilling_local@16 | pass@1 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0625 | 0.1250 | 0.1250 |

## 简读

### 尺子与梯度

- **Math_lite**：dense **0.98**，20% 仍 **0.56**，明显优于 Math_soft/GSM8K；30% 后仍掉得快。
- **Instruction_lite（RTE）**：dense **0.85** → 70% **0.40**，全程可读，接近 Sentiment 难度带；但 **40% 掉到 0.22 再回升到 ~0.40** 属分类塌缩（见下 C），勿当成能力恢复。
- **Code_lite（单行补全）**：dense～40% 约 **0.25**（N=16），高稀疏仍有非零分，优于 MBPP 归零，但梯度偏平。
- 推荐沟通组合：`Sentiment + Reasoning_soft + Knowledge_soft + Instruction_lite + Math_lite`。

### 相邻稀疏「几十百分点」怎么读（勿一律归因小样本）

| 类 | 形态 | 主因 | 表内典型 |
|----|------|------|----------|
| **A 真能力悬崖** | 大体**单调**骤降、几十 pp | Oneshot Wanda、无 recovery；难生成任务真实断崖。小 N 只放大台阶，**不是主因** | Formal Math 10%→30%（约 -42pp）；Math_easy dense→20%；Code_easy 10%→20%（-37.5pp，6/16 题） |
| **B 格式 / exact_match 全有全无** | **非单调**大跳、可整段归零 | 输出不再匹配答案抽取（格式乱、截断、拒答） | **已修**：Reasoning_easy 改为 no-chat + max_gen_toks=32 + until + flexible-extract；旧曲线 0→0.375→0 **作废**。现曲线 dense 0.84→70% 0.38（大体可读；N=32 仍有小抖动） |
| **C 分类塌缩** | 先低于 chance，再钉在多数类附近 | 二分类表示崩坏后乱猜→塌到基线；**回升≠恢复** | Instruction_lite：30% 0.57 → 40% **0.22** → 50%～70% **~0.40** |
| **小噪声** | 仅 **±1～8pp** | 整数题数抖动（1～5 题） | Formal Code dense→10%（+1 题）；Math 地板 0→0.0156；Formal Instruction 50%→70%（+7.8pp） |

**Instruction_lite 的「回升」仍是 C 类分类塌缩误读，不要当单调压缩曲线读。** Reasoning_easy **B 类协议已修**（见下），推荐沟通组合仍优先 `Reasoning_soft`。

**Reasoning_easy 协议修复（B）**：关 chat 模板、`max_gen_toks=32`、加强 until、本地任务 `bbh_fewshot_boolean_expressions_easy`（flexible-extract）。旧旁路假零分作废。抽查 dense/10/20/30@8：短答、无 emoji/长循环；全稀疏复测见上表。产物：`/mnt/data2/results/E1_soft_eval_side/reasoning_easy_gen_spotcheck.md`。

## 产物路径

| 项 | 路径 |
|----|------|
| 本报告 | `docs/results/E1_eval_ladder_all_banks.md` |
| Reasoning_easy 生成抽查 | `/mnt/data2/results/E1_soft_eval_side/reasoning_easy_gen_spotcheck.md` |
| JSON 总表 | `/mnt/data2/results/E1_soft_eval_side/all_banks_sparsity_table.json` |
| `_easy` | `/mnt/data2/results/E1_soft_eval_side/` |
| `_soft` | `/mnt/data2/results/E1_soft_eval_soft/` |
| `_lite` | `/mnt/data2/results/E1_soft_eval_lite/` |
| 菜单 | `docs/project/MENTOR_EVAL_MENU.md` |
