# PDF 实验执行表 §31–34 要求记录

> 来源：[Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf](../refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf)
> 文内页码 **31–34**（PDF 文件约第 **16–20** 页）；规格细节交叉引用同 PDF 前部 E0–E3 / E8–E9（约第 1–4、9–10 页）
> 记录日期：2026-08-20 · 执行对照：[EXPERIMENT_E_MAP.md](EXPERIMENT_E_MAP.md)

---

## 总原则（PDF 文首 · 与 §31–34 一体）

| Stage | 核心问题 | 实验 |
|-------|----------|------|
| A | Compressibility Frontier 是否真实存在？ | E0–E3 |
| B | Adaptive Ticket Search 是否优于固定剪枝？ | E4–E7 |
| C | Compression Frontier 数据是否能帮助恢复？ | E8–E11 |
| D | 模型能否形成长期 Autonomous Self-Compression？ | E12–E14 |

**硬约束**：前一阶段没有稳定信号，则暂缓进入下一阶段。最终系统要发现的是 **Empirical Compressibility Frontier**。

---

## §31 整体实验清单（必须按 ID 归档成果）

| ID | 实验 | 核心结论（要回答的问题） | 优先级 |
|----|------|--------------------------|--------|
| E0 | Dense baseline | 建立评价基准 | P0 |
| E1 | One-shot sparsity curve | Frontier 是否存在 | P0 |
| E2 | One-shot vs iterative | 逐步搜索是否必要 | P0 |
| E3 | Compression Gap | Gap 是否预测损伤 | P0 |
| E4 | Fixed iterative | Autonomous baseline | P0 |
| E5 | Adaptive ratio | 自适应步长是否有效 | P1 |
| E6 | Adaptive region | Capability feedback 是否有效 | P1 |
| E7 | Rewind + regrow | 可逆搜索是否有效 | P1 |
| E8 | Random recovery | Recovery baseline | P0 |
| E9 | High-gap recovery | Compression-specific data | P0 |
| E10 | Cross-compression | Policy-specific frontier | P1 |
| E11 | Synthetic frontier | BigBang-style generator | P1 |
| E12 | Controller comparison | LLM controller 是否必要 | P1 |
| E13 | Self vs external | Self-governance frontier | P1 |
| E14 | Full autonomous run | Autonomous Lottery Ticket | P2 |

**本仓库要求**：每个已跑 ID 的产物目录为 `/mnt/data2/results/E{n}_{slug}/`，含 `e{n}_report.md` + `e{n}_summary.json`；人类索引在 `docs/results/`。禁止把非 E 过渡实验改贴成已完成 En。

---

## §32 最关键的 Go / No-Go Gates（写进报告结论栏）

| Gate | 依赖 | 失败则 |
|------|------|--------|
| **A** | E1 / E2 | 若不存在 performance cliff、capability-specific frontier、iterative advantage → **暂时不要做 Agent** |
| **B** | E3 / E9 | 若既不能预测 degradation、也不能指导 recovery → **放弃 Compression Frontier 数据主线** |
| **C** | E5 / E6 / E7 | 若 adaptive search 已够（相对 fixed schedule）→ **不用做复杂 controller** |
| **D** | E11 | 若 SyntheticFrontier ≤ HighGapReal → synthetic **仅 optional**，不作论文核心 |
| **E** | E13 | 若 compressed child 无法可靠控制下一轮 → **不要声称完全 self-compressing**；可改为 PersistentMetaController + AutonomousTicketSearch，并把 Self-Governance Frontier 作为分析结论 |

---

## §33 GPU 执行建议（资源配置）

| Phase | 实验 | 建议硬件 | PDF 备注 |
|-------|------|----------|----------|
| 1 | E0–E3 | **1× RTX 4090 24GB** | 完全够；主要跑 Qwen **3B** + **2k–5k** profiling pool |
| 2 | E4–E11 | 2–4× 4090 | ticket / seed 可并行 |
| 3 | E12–E14 | 4× 4090 或 2–4× A100/L40S | 需求在 throughput（多 ticket / recovery / bench），不是单模装不下 |

**本机基线**：仍默认单卡；仅当可拆 ≥2 独立 cell 且墙钟很长时再加第 2 卡（见 CLAUDE.md）。当前权重为 1.5B 时，E 报告必须标 `spec: proxy_1.5B`。

---

## §34 最先应该启动的六个实验（第一批）

**不要一口气实现 E0–E14。** 第一批只启动：

1. **E0** Dense baseline
2. **E1** sparsity curve
3. **E2** one-shot vs iterative
4. **E3** Compression Gap correlation
5. **E8** random recovery
6. **E9** High-Gap vs Student-Hard vs Random

回答两个核心科学问题（**只有两个都是 Yes，才值得继续 Agent 线**）：

| Q | 问题 | 否 → |
|---|------|------|
| **Q1** | Is there a meaningful Compressibility Frontier? | 暂停 Agent |
| **Q2** | Can the model diagnose and repair compression-specific failures? | 不投入 AdaptiveTicketSearch → SyntheticGeneration → SelfController |

PDF 原话要点：第一阶段甚至**完全不需要 Agent**。

**排期锁定（本仓库）**：第一批前半 **E0→E1→E2→E3** 已跑 proxy；下一步审阅后开 **E8→E9**；E13 属 Stage D，禁止插队。

---

## 第一批规格摘要（供执行核对 · 摘自 PDF 各 E 节）

### E0 Dense baseline（P0）

| 项 | PDF 要求 |
|----|----------|
| Model | Qwen2.5-3B-Instruct |
| Compression | None |
| Evaluation | PPL + Math + Knowledge + Reasoning + Instruction + Code |
| Seeds / GPU | 1 / 1×24GB |
| 输出 | Dense performance vector P(M0)；GPU memory；latency；parameter count；model size |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| 用途 | 后续所有 acceptance threshold 以 E0 为基准 |

### E1 One-shot sparsity curve（P0）

| 项 | PDF 要求 |
|----|----------|
| Method | Wanda |
| Sparsity | 10%…70%（步长 10%） |
| Recovery | None |
| 核心图 | performance vs sparsity |
| 成功条件 | 明显非线性 degradation / **capability-specific** degradation（不同能力不同 cliff） |

### E2 Iterative vs One-shot（P0）

| 项 | PDF 要求 |
|----|----------|
| Target sparsity | 40%, 50%, 60% |
| Baseline A / B | One-shot / 5% incremental |
| Recovery | None |
| Seeds | 3 |
| 成功条件 | iterative **稳定**优于 one-shot |
| 若无差异 | 「逐代 lineage」可能无价值，Lottery Ticket 叙事需重审 |

### E3 Compression Gap（P0）

| 项 | PDF 要求 |
|----|----------|
| Parent / Children | Dense / 30–60% sparse |
| Pool | **2k–5k** prompts |
| Score | G(x)=L_S(y\|x) − L_T(y\|x)；corr(G, Failure)；对照 parent/child NLL、length、random |
| 成功条件 | Gap **显著**预测 downstream degradation |

### E8 Random recovery（P0 · 第一批后半）

| 项 | PDF 要求 |
|----|----------|
| Child | 50% sparse |
| Recovery | LoRA 200–500 steps |
| Data | Random；size 256 / 512 / 1k |
| Seeds | 3 |
| 用途 | 后续数据方法 baseline |

### E9 Hard / Gap data（P0 · 第一批后半 · Frontier 最重要）

| 策略 | 定义要点 |
|------|----------|
| Random / Teacher Hard / Student Hard / Low Gap / High Gap / HighGap+Random 80/20 | 对照难度来源 |
| 固定 | 相同 #examples、total token budget、尽量 matched length、相同 recovery steps |
| 期望排序 | HighGap > StudentHard > Random > LowGap（关键：HighGap > StudentHard） |
| 指标 | RecoveryGain = Perf(M_recover) − Perf(M_compressed) |

---

## 本趟对照（proxy_1.5B · 2026-08-20）

| ID | 归档目录 | 相对 PDF | Gate 影响 |
|----|----------|----------|-----------|
| E0 | `/mnt/data2/results/E0_dense_baseline/` | 仅 SST-2 LM proxy；六维能力不全；模型 1.5B 非 3B | 基准可用但不完整 |
| E1 | `…/E1_oneshot_sparsity_curve/` + 核心图 | magnitude MLP 代理 Wanda；单能力；剪后 PPL 爆炸 | **Gate A 暂不可判**（方法过糙，非真 cliff） |
| E2 | `…/E2_iterative_vs_oneshot/` | 仅 40/50%；单 seed；无稳定 iterative 优势 | **Gate A 暂不可判** |
| E3 | `…/E3_compression_gap/` | pool n=64；corr=nan | **Gate B 待 E9**；本 proxy 信号弱 |
| E8–E9 | — | 未开 | 审阅 E0–E3 后再开 |

人类镜像：`docs/results/E{0–3}_*.md` · 索引：`/mnt/data2/results/README_E_INDEX.md`。

---

## 成果文件夹约定（落实 §31）

```text
/mnt/data2/results/E0_dense_baseline/
/mnt/data2/results/E1_oneshot_sparsity_curve/
/mnt/data2/results/E2_iterative_vs_oneshot/
/mnt/data2/results/E3_compression_gap/
…（E4+ 同理）

docs/results/E0_dense_baseline.md   # 链到或摘录报告
```

非 E 历史产物（`qwen_glue_*` / `qwen_k6_*`）保留原路径，仅在 E-MAP「Non-E / proxy」登记，**不**迁入 `E*` 目录冒充。
