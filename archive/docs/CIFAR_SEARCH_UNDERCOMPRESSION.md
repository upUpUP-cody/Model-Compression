# CIFAR 自主搜索高压缩欠压分析（Phase I.A）

## 现象（sweep_v2）

在目标 ≥4x 时，`autonomous_search` 的实测压缩常明显低于目标；`iterative_structured_level1` 能稳定打到目标附近。

| seed | 目标 | search 最终压缩 | 关键事件摘要 |
|------|------|-----------------|--------------|
| 42 | 10x | **3.58x** | it0: `regrow` / `capability_gap_exceeded` @ 10.16x → it1: `accept` @ 3.58x |
| 44 | 10x | **1.00x** | it0: `regrow` @ 10.16x → it1: `regrow` @ 3.58x（耗尽后停在基线） |

证据目录：`results/cifar_p12_comparison_gpu_sweep_v2/ratio_10_seed_{42,44}/`

## 根因

1. 第一轮按全目标均匀剪枝（约 10.16x）。
2. Level-1 恢复后 validation 掉点超过 `max_accuracy_drop_points: 2.0` → 触发 `capability_gap_exceeded` → `regrow`。
3. 第二轮只能接受更温和剪枝，或继续失败停在 1.00x。
4. **不对称**：iterative **没有** 2 点能力门禁，可直接接受掉 ~4–5 点的 10x 模型；search 在硬门禁下无法同压缩对齐。

## 修复策略（已实现）

保留 2 点门禁的科学含义，改为 **多轮增量逼近目标**：

- `search.max_step_compression`（默认 1.75）：每轮只朝目标迈一小步，避免一轮直接 10x。
- `search.max_iterations` 提高到 12，接受后以新模型为 baseline 继续剪。
- 仍用 `target_compression_reached` 止损；`regrow` 后用更小步长继续。

### I.A′ 边界加固（sweep_v3 outlier 后）

| 案例 | 现象 | 根因 / 修复 |
|------|------|-------------|
| 4x seed43 → 6.32x | 过冲 | 近目标时 `_round_candidate_ratios` 回退全目标 `configured_ratios`；现禁止该 fallback，并在接受前过滤 `> target*1.15` |
| 10x seed42 → 8.43x | 欠压 | 近目标多次门禁失败耗尽迭代；`max_iterations` 提到 12 |

相关代码：`src/autonomous_search.py`（`_round_candidate_ratios`、过冲过滤）、`src/experiments/compression_targets.py`（`derive_uniform_prune_ratio`）。
