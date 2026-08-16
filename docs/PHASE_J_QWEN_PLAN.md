# Phase J — Qwen 规划（仅文档，不实现）

> 状态：规划已完成 · **Phase K 冒烟已跑通（2026-08-15/16）**
> **执行顺序更新（2026-08-16）**：**先 GLUE，再 SQuAD 小矩阵** — 以 [PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md) §KG 为准
> **Phase K 执行计划**：[PHASE_K_QWEN_PLAN.md](PHASE_K_QWEN_PLAN.md)
> 模型锁定：`Qwen/Qwen2.5-1.5B-Instruct`；大文件根目录：`/mnt/data`（软链 `llm_data/`）
> 视觉域主结论见 [EVIDENCE_PACK.md](EVIDENCE_PACK.md) / [WORK_LOG.md](WORK_LOG.md)
> 总路线：[PROJECT_PLAN.md](../PROJECT_PLAN.md) Phase J/K · 执行入口：[P2_EXECUTION_PLAN.md](P2_EXECUTION_PLAN.md) §P2.9

---

## 1. 目的与边界

把自主结构化剪枝 + 搜索协议从 CIFAR ResNet **迁移规划**到小型 LLM。本阶段只回答「怎么做、量什么、门禁是什么」；**Phase K 才开代码**。

| 做 | 不做 |
|----|------|
| 剪枝单元 / 指标 / 对照矩阵草图 | 下载 Qwen 或其它 LLM 权重（当时） |
| 数据与磁盘/显存门禁清单 | 实现 head/FFN 物理剪枝（当时） |
| 与视觉域叙事的衔接说明 | 跑正式大表 |

---

## 2. 任务设定（草图；执行以 PHASE_K 为准）

| 项 | 规划默认 |
|----|----------|
| 模型 | **已锁定** `Qwen/Qwen2.5-1.5B-Instruct` |
| **任务顺序** | **先 GLUE（SST-2 → RTE/QNLI）看效果，再 SQuAD 2.0** |
| GLUE 指标 | Accuracy |
| SQuAD 指标 | F1 / EM |
| 剪枝单元 | attention **head**；FFN **中间维**（物理缩小，非掩码稀疏） |
| 数据协议 | train / validation / test 严格隔离 |
| 对照方法 | dense / oneshot / iterative+recovery / autonomous_search |
| 压缩目标 | 先小矩阵冒烟，再扩展；regime-dependent crossover **待验** |

---

## 3. 与视觉域叙事的衔接

CIFAR 已定稿为 **regime-dependent**。LLM 上：

- **不预设** search 系统全面优于 iterative
- 先在 GLUE 问：同压缩预算下是否仍有 crossover / 门禁贡献；再在 SQuAD 验证
- 视觉域 2pt 门禁等迁移假设须单独消融

---

## 4. 硬件与磁盘门禁（Phase K 启动前）

| 门禁 | 要求 |
|------|------|
| VRAM | 建议 ≥16GB（本机 4090 满足算力，但仍须按具体模型核对） |
| 磁盘 | 权重 + HF 缓存 + 多次 run：**建议空闲 ≥30G** |
| 证据 | Phase H/I 文档审查通过 |
| 批准 | 用户确认扩盘并同意开始下载后，才进入 Phase K |

**当前**：Phase K 已开代码；权重与缓存只写 `/mnt/data`；运行产物写 `/mnt/data2`。

---

## 5. Phase K 实现清单（进行中；细节见 PHASE_K）

1. [√] SQuAD 协议与管线冒烟（K0–K5）
2. [ ] **GLUE 协议与冒烟（KG）— 当前优先**
3. [ ] SQuAD 小矩阵（K6，依赖 KG）
4. [ ] 外部 baseline 调研（K6-lit）
5. 产物：manifest / fingerprint / frozen test report（与 P1/P2 同构）

环境入口：`source scripts/env_llm.sh`

---

## 6. 验收（Phase J）

- [√] 本文可独立阅读，含单元、指标、对照、门禁
- [√] PROJECT_PLAN / P2 已链到本文
- [√] Phase K 启动后权重落在 `/mnt/data`，不提交进仓库
- [√] 执行顺序已更新为 **GLUE → SQuAD**（与导师要求对齐）
