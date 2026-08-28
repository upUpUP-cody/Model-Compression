# E0. Dense Model Baseline

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E0 |
| 目的 | 建立所有后续实验的 dense baseline |
| Model | formal_3B (PDF: Qwen2.5-3B-Instruct) |
| Method / Compression | None |
| Evaluation | 标准小样本六维（lm_eval, mode=scan, seed=42） |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| status | done |

## 六维协议（冻结）

| 维 | 任务 | limit | 指标 |
|----|------|-------|------|
| PPL | wikitext | 4.0 | word_perplexity,none |
| Math | gsm8k | 64.0 | exact_match,flexible-extract |
| Knowledge | mmlu | 128.0 | acc,none |
| Reasoning | bbh | 64.0 | exact_match,get-answer |
| Instruction | ifeval | 64.0 | prompt_level_strict_acc,none |
| Code | humaneval | 32.0 | pass@1,create_test |

## 输出

Dense performance vector P(M0) 与资源行（PDF §E0）。

## 记录表

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| 9.4696 | 0.5938 | 0.6761 | 0.0133 | 0.6406 | 0.6875 |

| GPU memory (GB peak) | latency_sec_eval | parameter_count | model path |
|----------------------|------------------|-----------------|------------|
| 10.690512657165527 | 3334.8 | 3085938688 | `/mnt/data/models/Qwen2.5-3B-Instruct` |

## 结论（对照成功条件）

- **满足**：六维 lm_eval pipeline 已落盘；小样本协议主看后续相对 Delta 与 capability-specific cliff。
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E；**E0 Instruct 永不进入 Gate A 数值**。
- 旧 SST-2 部分向量产物作废；以本报告为准。
- Code 维已用修复后的 HumanEval 补全协议重评（gate score=0.6875）。

- Reasoning 维已用 BBH max_gen_toks=1024 重评（Instruct；old=0.0133 new=0.0133）。**不可与 E1 base 对比**。
