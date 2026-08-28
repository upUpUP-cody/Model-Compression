# E1. One-shot Sparsity Curve

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E1 |
| 目的 | 判断是否存在明显 performance cliff（capability-specific） |
| Model | formal_3B_base (PDF: Qwen2.5-3B) |
| Method / Compression | oneshot Wanda MLP structured prune |
| Sparsity | 10%, 20%, 30%, 40%, 50%, 60%, 70% |
| Recovery | None |
| Evaluation | 六维 scan（与 E0 相同 limit/seed）；Delta 相对 **base 3B dense**（非 E0 Instruct）；Reasoning **max_gen_toks=1024** |
| skip_dimensions | （无） |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | 明显非线性 / capability-specific degradation |
| status | done |

## Dense 参照（base 3B 未剪枝）

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| 11.6777 | 0.7031 | 0.3357 | 0.5220 | 0.2500 | 0.6875 |

## 各档 P(s) 六维向量

| Sparsity | PPL | Math | Knowledge | Reasoning | Instruction | Code |
|----------|------|------|------|------|------|------|
| 10% | 15.0183 | 0.4531 | 0.2838 | 0.3374 | 0.2031 | 0.7188 |
| 20% | 23.0596 | 0.2188 | 0.2776 | 0.2587 | 0.2031 | 0.5625 |
| 30% | 29.9586 | 0.0312 | 0.2551 | 0.0556 | 0.1250 | 0.3750 |
| 40% | 54.6195 | 0.0312 | 0.2602 | 0.0220 | 0.1406 | 0.1562 |
| 50% | 76.7580 | 0.0000 | 0.2515 | 0.0000 | 0.0781 | 0.0312 |
| 60% | 129.9627 | 0.0156 | 0.2529 | 0.0069 | 0.1094 | 0.0000 |
| 70% | 229.9322 | 0.0312 | 0.2526 | 0.0041 | 0.1562 | 0.0000 |

## 各档 Delta vs dense

| Sparsity | Delta PPL | Delta Math | Delta Knowledge | Delta Reasoning | Delta Instruction | Delta Code |
|----------|------|------|------|------|------|------|
| 10% | 3.3407 | -0.2500 | -0.0518 | -0.1846 | -0.0469 | 0.0312 |
| 20% | 11.3820 | -0.4844 | -0.0581 | -0.2633 | -0.0469 | -0.1250 |
| 30% | 18.2810 | -0.6719 | -0.0806 | -0.4664 | -0.1250 | -0.3125 |
| 40% | 42.9418 | -0.6719 | -0.0755 | -0.5000 | -0.1094 | -0.5312 |
| 50% | 65.0804 | -0.7031 | -0.0842 | -0.5220 | -0.1719 | -0.6562 |
| 60% | 118.2850 | -0.6875 | -0.0828 | -0.5150 | -0.1406 | -0.6875 |
| 70% | 218.2545 | -0.6719 | -0.0830 | -0.5179 | -0.0938 | -0.6875 |

## Cliff 判定（启发式）

- capability-specific 信号：是（cliff 维：PPL, Math, Reasoning, Instruction, Code）
- Gate A 输入：与 E2 一并判断是否暂缓 Agent
- 缺失 / skip 维不参与 cliff（报告为 n/a）

## 输出图

- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_ppl.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_math.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_knowledge.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_reasoning.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_instruction.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_code.png`
- `/mnt/data2/results/E1_oneshot_sparsity_curve/figures/performance_vs_sparsity_capability_specific.png`

## 结论

- Dense 参照为 **base Qwen2.5-3B** 自身未剪枝向量；**不得**与 E0 Instruct 混比。
- 方法为 **Wanda**（gate 权重 × MLP 中间激活校准）；校准集见 config `pruning.calibration`。
- Reasoning 正式协议为 **BBH max_gen_toks=1024**（`do_sample=false`）；`pre_bbh1024_*` 归档中的 2048 分数**作废**，不得进 Gate A。
- 墙钟（本报告生成时）：495989.8s
