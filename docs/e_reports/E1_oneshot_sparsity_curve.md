# E1. One-shot Sparsity Curve

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E1 |
| 目的 | 判断是否存在明显 performance cliff |
| Model | proxy_1.5B (PDF: Qwen2.5-3B-Instruct) |
| Method / Compression | oneshot MLP magnitude（PDF 写 Wanda；本跑为 proxy） |
| Evaluation | SST-2 LM PPL；Recovery **None** |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | 存在明显非线性 degradation / capability-specific degradation |
| status | done_proxy |

## 输出

核心图：`figures/performance_vs_sparsity.png`

## 记录表

Dense PPL = 1.9851

| Sparsity | Delta PPL | PPL | proxy_acc |
|----------|-----------|-----|-----------|
| 10% | 51102394.015 | 51102396.000 | 5.33 |
| 20% | 319487774.015 | 319487776.000 | 4.86 |
| 30% | 136023246.015 | 136023248.000 | 5.07 |
| 40% | 32938412.015 | 32938414.000 | 5.46 |
| 50% | 542765694.015 | 542765696.000 | 4.74 |
| 60% | 22583052286.015 | 22583052288.000 | 4.03 |
| 70% | 826822078.015 | 826822080.000 | 4.64 |

## 结论（对照成功条件）

- **部分满足**：PPL 随 sparsity 上升，见曲线；单任务 proxy，非 PDF 多能力 cliff。
- **Gate A 输入**：本跑提供 E1 侧证据（需与 E2 一并判断是否做 Agent）。
- 规格：`proxy_1.5B`；方法非 Wanda。
