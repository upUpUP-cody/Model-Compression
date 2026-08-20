# Experiment E map（唯一标准 · PDF §31）

强制顺序：**E0 → E1 → E2 → E3**，再第一批 **E8 → E9**，然后按 Gate 进入 Stage B / D。

- 排版：[E_REPORT_TEMPLATE.md](E_REPORT_TEMPLATE.md)
- 结构：[STRUCTURE.md](STRUCTURE.md)
- **§31–34**：[PDF_E_REQUIREMENTS_31_34.md](PDF_E_REQUIREMENTS_31_34.md)
- 日志：[WORK_LOG.md](WORK_LOG.md)（按 E ID）

PDF 默认模型：**Qwen2.5-3B-Instruct**。当前磁盘仅 **1.5B** → Stage A 标注 `spec: proxy_1.5B`。

| E | PDF 目的 | Stage | 状态 | 产物目录 |
|---|----------|-------|------|----------|
| E0 | Dense baseline | A | done_proxy | `/mnt/data2/results/E0_dense_baseline/` · [results](../results/) |
| E1 | One-shot sparsity curve | A | done_proxy | `/mnt/data2/results/E1_oneshot_sparsity_curve/` |
| E2 | Iterative vs one-shot | A | done_proxy | `/mnt/data2/results/E2_iterative_vs_oneshot/` |
| E3 | Compression Gap | A | done_proxy | `/mnt/data2/results/E3_compression_gap/` |
| E4–E7 | Adaptive search | B | pending | — |
| E8 | Random recovery | C | **partial_smoke**（n=256/seed42/steps30） | `/mnt/data2/results/E8_random_recovery/` |
| E9 | High-Gap recovery | C | pending | — |
| E10–E11 | Frontier data | C | pending | — |
| E12–E14 | Self-compression | D | pending | **禁止插队** |

## Gates（§32）

| Gate | 依赖 | 失败动作 |
|------|------|----------|
| A | E1/E2 | 暂缓 Agent |
| B | E3/E9 | 放弃 Frontier 数据主线 |
| C | E5–E7 | 可不做复杂 controller |
| D | E11 | synthetic 仅 optional |
| E | E13 | 不声称完全 self-compressing |

## Non-E（勿迁入 E* 目录）

| 工作 | 路径 |
|------|------|
| CIFAR formal100 | `results/cifar_p12_*`；详表见 `archive/docs/EVIDENCE_PACK.md` |
| GLUE 过渡 | `/mnt/data2/results/qwen_glue_*` |
| SQuAD LoRA | `/mnt/data2/results/qwen_k6_*` |

## 下一步

审阅 E0–E3 → 执行入口 [`../process/NEXT_E8_E9.md`](../process/NEXT_E8_E9.md)；不启动 E13。
