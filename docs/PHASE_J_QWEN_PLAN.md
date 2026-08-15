# Phase J — Qwen / SQuAD 规划（仅文档，不实现）

> 状态：规划占位 · **不下载权重 · 不写 Transformer 剪枝代码**  
> 视觉域主结论见 [EVIDENCE_PACK.md](EVIDENCE_PACK.md) / [WORK_LOG.md](WORK_LOG.md)  
> 总路线：[PROJECT_PLAN.md](../PROJECT_PLAN.md) Phase J/K · 执行入口：[P2_EXECUTION_PLAN.md](P2_EXECUTION_PLAN.md) §P2.9

---

## 1. 目的与边界

把自主结构化剪枝 + 搜索协议从 CIFAR ResNet **迁移规划**到小型 LLM 阅读理解设定。本阶段只回答「怎么做、量什么、门禁是什么」；**Phase K 才开代码**。

| 做 | 不做 |
|----|------|
| 剪枝单元 / 指标 / 对照矩阵草图 | 下载 Qwen 或其它 LLM 权重 |
| 数据与磁盘/显存门禁清单 | 实现 head/FFN 物理剪枝 |
| 与视觉域叙事的衔接说明 | 跑 SQuAD 正式表 |

---

## 2. 任务设定（草图）

| 项 | 规划默认 |
|----|----------|
| 模型 | 小型 Qwen 指令/基座变体（具体 checkpoint 在 Phase K 开代码前再锁定） |
| 任务 | SQuAD 2.0（或同协议子集）；主指标 **F1 / EM** |
| 剪枝单元 | attention **head**；FFN **中间维**（物理缩小，非掩码稀疏） |
| 数据协议 | train / validation / test 严格隔离；选择只看 validation；test 仅 freeze 后一次 |
| 对照方法 | dense / oneshot / iterative+recovery / autonomous_search |
| 压缩目标 | 先 1.5x–4x 小矩阵冒烟，再扩展；是否出现 regime-dependent crossover **待验** |

---

## 3. 与视觉域叙事的衔接

CIFAR 已定稿为 **regime-dependent**（≤4x iterative 略稳；≥8x search 更强；机制随压缩率变）。LLM 上：

- **不预设** search 系统全面优于 iterative
- 先问：同压缩预算 + 同恢复预算下，是否仍出现 crossover / 门禁贡献
- 视觉域的 2pt 能力门禁、增量步长、过冲硬顶是否迁移，作为 Phase K 假设，需单独消融

---

## 4. 硬件与磁盘门禁（Phase K 启动前）

| 门禁 | 要求 |
|------|------|
| VRAM | 建议 ≥16GB（本机 4090 满足算力，但仍须按具体模型核对） |
| 磁盘 | 权重 + HF 缓存 + 多次 run：**建议空闲 ≥30G**；不足则先扩盘 |
| 证据 | Phase H/I 文档审查通过；CIFAR 主表与机制消融可读 |
| 批准 | 用户确认扩盘并同意开始下载后，才进入 Phase K |

**当前**：只维护本文档；**禁止** `huggingface-cli download` / 拉取大模型。

---

## 5. Phase K 实现清单（预告，本阶段不执行）

1. 数据：SQuAD 划分与 validation 选择器  
2. 后端：Transformer 物理剪枝（head / FFN 中间维）+ 参数量审计  
3. 搜索：复用 controller / frontier / history 协议；适配新候选空间  
4. 对照：E1/E2 级 dense / oneshot / iterative / search  
5. 产物：manifest / fingerprint / frozen test report（与 P1/P2 同构）

---

## 6. 验收（Phase J）

- [√] 本文可独立阅读，含单元、指标、对照、门禁  
- [√] PROJECT_PLAN / P2 已链到本文  
- [×] 仓库内出现新增大模型权重或缓存 — **不允许**
