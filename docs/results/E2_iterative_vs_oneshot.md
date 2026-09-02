# E2. Iterative vs One-shot Compression

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E2 |
| 目的 | 验证逐步压缩是否优于一次性压缩 |
| Model | formal_3B_base (PDF: Qwen2.5-3B) |
| Method / Compression | Wanda；Baseline A One-shot；Baseline B 5% incremental；Recovery **None** |
| Target sparsity | 40%, 50%, 60% |
| Evaluation | 六维 scan（与 E1 相同 limit/seed）；Delta vs **base 3B dense** |
| Seeds | [42, 43, 44] |
| GPU | dual multi-process by seed (see launch_e2_dual.sh) |
| 优先级 | P0 |
| 成功条件 | iterative 稳定优于 one-shot（主四维 ≥3/4 且 ≥7/9 cells） |
| status | done |

## Dense 参照（base 3B，来自 E1）

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| 11.6777 | 0.7031 | 0.3357 | 0.5220 | 0.2500 | 0.6875 |

## Cell 对照（PPL + Gate）

**Cell (主四维)** 只看 PPL / Math / Knowledge / Reasoning（**不含** Instruction / Code）：

| 标签 | 含义 |
|------|------|
| `iterative` | 主四维上 iterative 至少赢 3 维（Gate 计为 cell 胜） |
| `oneshot` | 主四维上 oneshot 至少赢 3 维 |
| `tie` | 两边都未满 3 维（例如 2–2，或夹杂单维平局） |

**单维** `tie`（见下方六维表）= 该维分数完全相等，与 **Cell `tie`** 不是同一概念。  
**Six wins** 为全六维对照，**不改变** Gate A。

| Seed | Target | One-shot PPL | Iterative PPL | Main iter | Main oneshot | Six wins (iter) | Cell (主四维) |
|------|--------|--------------|---------------|-----------|--------------|-----------------|--------------|
| 42 | 40% | 54.6195 | 52.1421 | 3/4 | 1/4 | 4/6 | iterative |
| 42 | 50% | 76.7580 | 86.5116 | 2/4 | 2/4 | 4/6 | tie |
| 42 | 60% | 129.9627 | 168.3847 | 1/4 | 3/4 | 3/6 | oneshot |
| 43 | 40% | 53.9279 | 56.8807 | 2/4 | 1/4 | 4/6 | tie |
| 43 | 50% | 85.9510 | 88.6062 | 1/4 | 2/4 | 3/6 | tie |
| 43 | 60% | 127.1546 | 138.0806 | 2/4 | 2/4 | 3/6 | tie |
| 44 | 40% | 53.6900 | 51.5690 | 2/4 | 2/4 | 3/6 | tie |
| 44 | 50% | 79.9151 | 94.9769 | 2/4 | 2/4 | 3/6 | tie |
| 44 | 60% | 117.5813 | 230.8308 | 1/4 | 3/4 | 2/6 | oneshot |

## 六维分数对照（逐 cell）

PPL 越低越好；其余维越高越好。  
**Winner（单维）**：`iterative`（该维更好）/ `oneshot`（该维更好）/ `tie`（该维分数相等）。

### Seed 42 · Target 40%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 54.6195 | 52.1421 | iterative |
| Math | 0.0312 | 0.0469 | iterative |
| Knowledge | 0.2602 | 0.2539 | oneshot |
| Reasoning | 0.0220 | 0.0694 | iterative |
| Instruction | 0.1406 | 0.1406 | tie |
| Code | 0.1562 | 0.2812 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **3/4**，oneshot **1/4**（单维 tie 两边都不计）→ **Cell (主四维) = iterative**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**4/6**（仅对照，不进 Gate）

### Seed 42 · Target 50%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 76.7580 | 86.5116 | oneshot |
| Math | 0.0000 | 0.0156 | iterative |
| Knowledge | 0.2515 | 0.2490 | oneshot |
| Reasoning | 0.0000 | 0.0035 | iterative |
| Instruction | 0.0781 | 0.0938 | iterative |
| Code | 0.0312 | 0.0625 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **2/4**，oneshot **2/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**4/6**（仅对照，不进 Gate）

### Seed 42 · Target 60%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 129.9627 | 168.3847 | oneshot |
| Math | 0.0156 | 0.0000 | oneshot |
| Knowledge | 0.2529 | 0.2541 | iterative |
| Reasoning | 0.0069 | 0.0023 | oneshot |
| Instruction | 0.1094 | 0.1250 | iterative |
| Code | 0.0000 | 0.0312 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **1/4**，oneshot **3/4**（单维 tie 两边都不计）→ **Cell (主四维) = oneshot**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**3/6**（仅对照，不进 Gate）

### Seed 43 · Target 40%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 53.9279 | 56.8807 | oneshot |
| Math | 0.0312 | 0.0312 | tie |
| Knowledge | 0.2612 | 0.2639 | iterative |
| Reasoning | 0.0116 | 0.0133 | iterative |
| Instruction | 0.0938 | 0.1250 | iterative |
| Code | 0.1875 | 0.2500 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **2/4**，oneshot **1/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**4/6**（仅对照，不进 Gate）

### Seed 43 · Target 50%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 85.9510 | 88.6062 | oneshot |
| Math | 0.0156 | 0.0156 | tie |
| Knowledge | 0.2528 | 0.2465 | oneshot |
| Reasoning | 0.0000 | 0.0064 | iterative |
| Instruction | 0.0625 | 0.1094 | iterative |
| Code | 0.0000 | 0.0938 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **1/4**，oneshot **2/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**3/6**（仅对照，不进 Gate）

### Seed 43 · Target 60%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 127.1546 | 138.0806 | oneshot |
| Math | 0.0000 | 0.0156 | iterative |
| Knowledge | 0.2503 | 0.2575 | iterative |
| Reasoning | 0.0127 | 0.0000 | oneshot |
| Instruction | 0.1250 | 0.1094 | oneshot |
| Code | 0.0000 | 0.0625 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **2/4**，oneshot **2/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**3/6**（仅对照，不进 Gate）

### Seed 44 · Target 40%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 53.6900 | 51.5690 | iterative |
| Math | 0.0469 | 0.0156 | oneshot |
| Knowledge | 0.2607 | 0.2731 | iterative |
| Reasoning | 0.0237 | 0.0035 | oneshot |
| Instruction | 0.1250 | 0.0781 | oneshot |
| Code | 0.1250 | 0.2812 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **2/4**，oneshot **2/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**3/6**（仅对照，不进 Gate）

### Seed 44 · Target 50%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 79.9151 | 94.9769 | oneshot |
| Math | 0.0156 | 0.0312 | iterative |
| Knowledge | 0.2509 | 0.2494 | oneshot |
| Reasoning | 0.0023 | 0.0162 | iterative |
| Instruction | 0.1875 | 0.1250 | oneshot |
| Code | 0.0312 | 0.0625 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **2/4**，oneshot **2/4**（单维 tie 两边都不计）→ **Cell (主四维) = tie**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**3/6**（仅对照，不进 Gate）

### Seed 44 · Target 60%

| Dim | One-shot | Iterative | Winner（单维） |
|-----|----------|-----------|----------------|
| PPL | 117.5813 | 230.8308 | oneshot |
| Math | 0.0156 | 0.0312 | iterative |
| Knowledge | 0.2539 | 0.2425 | oneshot |
| Reasoning | 0.0064 | 0.0000 | oneshot |
| Instruction | 0.1094 | 0.0938 | oneshot |
| Code | 0.0000 | 0.0312 | iterative |

- 主四维（PPL/Math/Knowledge/Reasoning；Gate 只用这四维）： iterative **1/4**，oneshot **3/4**（单维 tie 两边都不计）→ **Cell (主四维) = oneshot**（≥3/4 才判胜；否则为 tie）
- 全六维 iterative 胜（含 Instruction/Code；单维 tie 不计）：**2/6**（仅对照，不进 Gate）


## 逐维 winner 总表（单维：iterative / oneshot / tie）

| Seed | Target | PPL | Math | Knowledge | Reasoning | Instruction | Code | Six wins |
|------|--------|------|------|------|------|------|------|----------|
| 42 | 40% | iterative | iterative | oneshot | iterative | tie | iterative | 4/6 |
| 42 | 50% | oneshot | iterative | oneshot | iterative | iterative | iterative | 4/6 |
| 42 | 60% | oneshot | oneshot | iterative | oneshot | iterative | iterative | 3/6 |
| 43 | 40% | oneshot | tie | iterative | iterative | iterative | iterative | 4/6 |
| 43 | 50% | oneshot | tie | oneshot | iterative | iterative | iterative | 3/6 |
| 43 | 60% | oneshot | iterative | iterative | oneshot | oneshot | iterative | 3/6 |
| 44 | 40% | iterative | oneshot | iterative | oneshot | oneshot | iterative | 3/6 |
| 44 | 50% | oneshot | iterative | oneshot | iterative | oneshot | iterative | 3/6 |
| 44 | 60% | oneshot | iterative | oneshot | oneshot | oneshot | iterative | 2/6 |

## Gate A（E2 半）

- iterative cell wins: **1/9**（门槛 ≥7；按主四维 ≥3/4）
- **未通过**：不支持「必须做 Agent」的 iterative 前提。
- 与 E1（capability-specific frontier）合判 Gate A。

## 输出图

- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_ppl.png`
- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_math.png`
- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_knowledge.png`
- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_reasoning.png`
- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_instruction.png`
- `/mnt/data2/results/E2_iterative_vs_oneshot/figures/iterative_vs_oneshot_code.png`

## 结论

- Dense / seed=42 oneshot 来自 E1 checkpoint；seed 43/44 oneshot 在 E2 内重剪。
- Incremental：累计 +5pp 绝对 sparsity；remaining-relative Wanda；无 recovery。
- Cell 栏与 Gate 只看主四维；报告已区分单维 `tie` 与 Cell 级 `oneshot` / `tie`。
