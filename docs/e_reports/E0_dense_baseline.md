# E0. Dense Model Baseline

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E0 |
| 目的 | 建立所有后续实验的 dense baseline |
| Model | proxy_1.5B (PDF: Qwen2.5-3B-Instruct) |
| Method / Compression | None |
| Evaluation | SST-2 LM proxy PPL + Instruction proxy_acc；Math/Knowledge/Reasoning/Code = n/a（待扩） |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| status | done_proxy |

## 输出

Dense performance vector P(M0) 与资源行（PDF §E0）。

## 记录表

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| 1.9851 | n/a | n/a | n/a | 59.32 | n/a |

| GPU memory (GB peak) | latency_sec_eval | parameter_count | model path |
|----------------------|------------------|-----------------|------------|
| 3.0132555961608887 | 17.5 | 1543714304 | `/mnt/data/models/Qwen2.5-1.5B-Instruct` |

## 结论（对照成功条件）

- **部分满足**：pipeline 可复现（load + SST-2 LM eval）；完整六维能力向量未齐。
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E。
- 规格：`spec=proxy_1.5B`（PDF 要 3B）。
