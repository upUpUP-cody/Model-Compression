# Phase J — Qwen / SQuAD 规划（仅文档，不实现）

> 状态：规划已完成 · **Phase K 冒烟已跑通（2026-08-15）**
> **Phase K 执行计划**：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 模型锁定：`Qwen/Qwen2.5-1.5B-Instruct`；大文件根目录：`/mnt/data`（软链 `llm_data/`）
> 视觉域主结论见 [EVIDENCE_PACK.md](EVIDENCE_PACK.md) / [WORK_LOG.md](WORK_LOG.md)
> 总路线：[PROJECT_PLAN.md](../PROJECT_PLAN.md) Phase J/K · 执行入口：[P2_EXECUTION_PLAN.md](P2_EXECUTION_PLAN.md) §P2.9
> 冒烟入口：`source scripts/env_llm.sh && python experiments/run_qwen_squad_eval.py --config configs/qwen_squad_smoke.yaml`

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
| 模型 | **已锁定** `Qwen/Qwen2.5-1.5B-Instruct`（本地 `/mnt/data/models/Qwen2.5-1.5B-Instruct`） |
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

**当前**：磁盘门禁已满足（`/mnt/data`）；Phase K 已开代码。权重与缓存**只写** `/mnt/data`，不进 git。

---

## 5. Phase K 实现清单（进行中）

1. 数据：SQuAD 划分与 validation 选择器
2. 后端：Transformer 物理剪枝（head / FFN 中间维）+ 参数量审计
3. 搜索：复用 controller / frontier / history 协议；适配新候选空间（本轮冒烟之后）
4. 对照：先 dense / oneshot 冒烟；iterative / search 接线留后续
5. 产物：manifest / fingerprint / frozen test report（与 P1/P2 同构）

环境入口：`source scripts/env_llm.sh`

---

## 6. 验收（Phase J）

- [√] 本文可独立阅读，含单元、指标、对照、门禁
- [√] PROJECT_PLAN / P2 已链到本文
- [√] Phase K 启动后权重落在 `/mnt/data`，不提交进仓库
