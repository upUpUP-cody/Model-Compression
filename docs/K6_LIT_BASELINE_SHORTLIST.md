# K6-lit — 外部压缩 baseline 调研短表（文献质量门禁）

> 日期：2026-08-16
> 依据：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) §8；挂接 [PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md) K6-lit
> 图例：`[√]` 入短名单 · `[~]` Related Work 一句 · `[×]` 不当主 baseline

## 研究问题

同压缩预算（及尽可能同恢复预算）下，`autonomous_search` 是否优于 **人工设计** 压缩方案（内部 oneshot/iterative + 外部代表方法）。

## 短名单（过质量门禁）

| 状态 | 工作 | Venue / 依据 | 方法类型 | 与本课题可比点 | 本阶段用法 |
|------|------|--------------|----------|----------------|------------|
| `[√]` | Lottery Ticket Hypothesis (Frankle & Carbin) | ICLR；领域奠基 | 迭代剪枝 / 重风初始化 | 「人工设计 iterative」谱系 | Related Work + 叙事对照；不直接复现 1.5B |
| `[√]` | Wanda (Sun et al.) | 高引用 LLM 剪枝；项目已有接口 | 幅值×激活一刀剪 | 人工设计 oneshot 侧 | Related Work；CIFAR/视觉侧已接触；LLM 结构化设定下 **不硬比数字** |
| `[√]` | SparseGPT (Frantar & Alistarh) | ICML | 一刀近似重建剪枝 | 人工设计 oneshot 侧 | Related Work；非物理改形状，差异须声明 |
| `[√]` | LLM-Pruner / SliceGPT 等结构化 LLM 剪枝 | 顶会结构化路线 | 结构化 / 深度或宽度裁剪 | 与「物理改形状」更可比 | Related Work 优先；能对齐压缩带再考虑复现 |

## 明确不进主 baseline

| 状态 | 说明 |
|------|------|
| `[×]` | 无出处博客数字、私有不可核验结果 |
| `[×]` | 与物理结构化剪枝完全不可比却硬贴「我们更好」的数字 |
| `[×]` | 未过门禁的边缘预印本当主表对照 |

## 与内部方法的角色分工

| 角色 | 方法 |
|------|------|
| 内部人工设计 | `oneshot`、`iterative_level1` |
| 内部自动搜索 | `autonomous_search`（当前单候选冒烟/小扫路径） |
| 外部人工设计 | 上表短名单（先文献，后视复现成本） |

## 与本仓库恢复差距（已用 R1 验证）

| 维度 | 旧 Level-1 SGD | 文献 / R1 LoRA |
|------|----------------|----------------|
| 数据 | 128–512 | LLM-Pruner ~50k；R1 用 SQuAD **8192** |
| 优化 | 全参 SGD | LoRA + AdamW（对齐 LLM-Pruner） |
| 1.5x oneshot F1 | ≈0.24（加深 SGD） | **≈30.6**（R1；n=64） |

结论：SQuAD 塌点主要是恢复配方，不是协议坏了。P² Law 的 0.5B-token 级持续预训练仍超出本阶段预算。

## 当前实验约束

- KG.5 GLUE：过渡可读；非正式主表。
- K6 SQuAD：LoRA 后 1.5x 四方法 F1 可读（iterative 38.9 / oneshot 30.6 / search 30.0；n=64）→ 可 Informal 对照；**仍不扩 2x**；不开 frozen test。
- 外部方法能复现则挂 `/mnt/data2/results/qwen_*` 同协议；否则仅 Related Work。

## 下一步（lit）

1. 论文 Related Work 按上表 4 条展开（各 3–5 句 + 声明差异；强调 LLM-Pruner LoRA 恢复）。
2. 评估是否复现 Wanda/结构化剪枝作同预算外对照（可选；非挡收口）。
