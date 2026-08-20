# E2. Iterative vs One-shot Compression

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E2 |
| 目的 | 验证逐步压缩是否优于一次性压缩 |
| Model | proxy_1.5B (PDF: Qwen2.5-3B-Instruct) |
| Method / Compression | Baseline A One-shot；Baseline B 4-step incremental；Recovery **None** |
| Evaluation | SST-2 LM PPL（Final performance proxy） |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | iterative 稳定优于 one-shot |
| status | done_proxy |

## 输出

核心指标：Final performance（PPL，越低越好）。Dense PPL = 1.9851

## 记录表

| Target sparsity | One-shot PPL | Iterative PPL | Winner |
|-----------------|--------------|---------------|--------|
| 40% | 32938414.000 | 39611152.000 | oneshot |
| 50% | 542765696.000 | 492089728.000 | iterative |

## 结论（对照成功条件）

- **不满足/未稳定**：iterative 胜 1/2 档。
- **Gate A**：与 E1 一并判断是否存在 iterative advantage；本跑 不支持「必须做 Agent」的前提之一。
- 规格：`proxy_1.5B`；PDF 目标 40/50/60%×3 seed 未全开。
