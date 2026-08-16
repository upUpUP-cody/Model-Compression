# Phase K — Qwen / GLUE → SQuAD 实现计划

> 状态：**K0–K5 `[√]`** · **KG.5 `[√]`** · **K6 小扫 + SGD 加深负对照 `[√]`** · **K6 LoRA@1.5x 四方法 Informal `[√]`** · **K6-lit 短表 `[√]`** · 下一档 = **论文口径收口**（Informal 附录；不扩 2x / 不开 frozen test）
> 导师调整（2026-08-16）：**在继续 SQuAD 正式/小矩阵前，先做 GLUE 看压缩效果**
> 任务锁定（2026-08-16）：**GLUE 正式标准 = SST-2 + RTE + QNLI**；KG.5 必做门禁；SQuAD = 生成式主考卷（见 §1.1）
>
> **当前结论（2026-08-16）**：短/中 SGD 恢复撑不起 SQuAD（负对照）；对齐文献的 **LoRA + 更大恢复预算** 后 1.5x 四方法 F1 可读（iterative ≥ oneshot ≈ search）；**非正式主表**；主贡献仍 CIFAR + GLUE 过渡。
> 图例：`[√]` 已完成 · `[ ]` 未做 · `[×]` 证据不支持 / 不做
> 规划前身：[PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) · 视觉结论：[EVIDENCE_PACK.md](EVIDENCE_PACK.md) / [WORK_LOG.md](WORK_LOG.md)
> 总路线：[PROJECT_PLAN.md](../PROJECT_PLAN.md) · 主机：RTX 4090 · 权重/数据：`/mnt/data` · 运行产物：`/mnt/data2`

---

## 1. 目标与边界

把 CIFAR 上已验证的协议迁到小型 LLM：

- **物理结构化剪枝**（改模块形状，非掩码稀疏）
- **train / validation / test 严格隔离**（选择只看 validation；test 冻结后一次）
- **同压缩预算对照**：dense / oneshot / iterative+recovery / autonomous_search

核心问题（与视觉域同一句式）：在 LLM 上，search vs iterative 是否仍呈 **regime-dependent**，还是塌成别的形态。

**任务顺序（锁定）**：

1. **KG — GLUE**：短文本闭集 NLU（正式子集 **SST-2 + RTE + QNLI**），先看剪枝+恢复是否出信号（当前优先）
2. **K6 — SQuAD**：长文阅读理解小矩阵（**KG.5 门禁通过后再开**）
3. **K6-lit**：外部压缩 baseline 调研（可与 KG 并行文档工作）

| 做 | 不做（本阶段默认） |
|----|-------------------|
| 1.5B 级 Qwen + **先 GLUE、后 SQuAD** | 7B / 14B；跳过 GLUE 直接开 SQuAD 小矩阵 |
| head / FFN 物理剪枝 + 参数量审计 | 把权重或 HF 缓存提交进 Git |
| dense / oneshot 冒烟 → **三任务正式小扫（KG.5）** | 重跑 CIFAR formal100；全 GLUE 九任务大表 |
| 复用搜索协议（门禁 / frontier / history） | 预设「search 全面更优」 |
| manifest / fingerprint / frozen test 报告 | 放宽视觉域 2pt 门禁叙事到 LLM（须单独消融） |

### 1.1 任务区分：GLUE vs SQuAD（锁定）

**一句话**：GLUE 测短文本闭集判断；SQuAD 测长文开放抽答——二者不可互相替代。

| 标准 | 任务 | 考察面 | 输出 / 指标 | 在课题中的角色 |
|------|------|--------|-------------|----------------|
| GLUE | **SST-2** | 单句情感 | verbalizer → Accuracy | 冒烟（KG.4）+ 正式三任务之一 |
| GLUE | **RTE** | 短文蕴含（二分类） | verbalizer → Accuracy | 正式 GLUE 标准 |
| GLUE | **QNLI** | 问句–句对是否可答 | verbalizer → Accuracy | 正式 GLUE 标准 |
| SQuAD | **SQuAD 2.0** | 长上下文阅读理解 / span + abstain | 生成 → F1 / EM | LLM **生成式主标准**（K6） |

**协议（写死）**：

- GLUE 一律 `prompt + verbalizer`（**不用**分类头）；数据划分与视觉同构（官方 validation = frozen test）
- Verbalizer 标签（固定，禁止中途混比）：
  - SST-2：`positive` / `negative`
  - RTE：`entailment` / `not_entailment`
  - QNLI：`yes` / `no`（句对是否包含答案）

**论文写法约束**：

- SST-2 冒烟 accuracy **不得**写成 LLM 主结论（须标冒烟）
- 三任务对照表可作 LLM **过渡 / 短 NLU** 证据
- 生成式主证据仍写 **SQuAD**（F1 / EM）

---

## 2. 锁定设定

| 项 | 锁定值 |
|----|--------|
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 本地权重 | `/mnt/data/models/Qwen2.5-1.5B-Instruct` |
| **当前优先任务** | **GLUE**（见 §KG） |
| GLUE 数据缓存 | `/mnt/data/datasets/glue` |
| GLUE 冒烟子集 | **SST-2**（管线打通；指标非正式） |
| **GLUE 正式标准** | **SST-2 + RTE + QNLI**（三任务；不做全 GLUE 九任务） |
| GLUE 主指标 | **Accuracy**（百分制；任务特殊指标若有则附记） |
| **SQuAD（生成式主标准）** | SQuAD 2.0（`rajpurkar/squad_v2`）；主指标 **F1 / EM** |
| SQuAD 数据缓存 | `/mnt/data/datasets/squad`（K1 已下载） |
| 剪枝单元 | attention **KV-group**；FFN **intermediate** |
| 设备 | CUDA（4090 24GB）；环境入口 `source scripts/env_llm.sh` |
| 结果根 | GLUE → `/mnt/data2/results/qwen_glue_*`；SQuAD → `/mnt/data2/results/qwen_*` / 既有 `qwen_squad_*` |
| **开 K6 门禁** | **KG.5 `[√]`**（2026-08-16；`/mnt/data2/results/qwen_glue_kg5/`） |

### 数据协议（与视觉同构）

```text
官方 train  --(split_seed, validation_fraction)-->  train' + validation'
官方 validation / 官方 test 约定 -------------------->  frozen test（选模/搜索禁止使用）
```

- SQuAD：[`src/utils/squad_protocol.py`](../src/utils/squad_protocol.py)（已有）
- GLUE：[`src/utils/glue_protocol.py`](../src/utils/glue_protocol.py)；断言 test 不进 selection path

### 存储约定

```bash
source scripts/env_llm.sh
# HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE -> /mnt/data/...
# HF_ENDPOINT 默认 https://hf-mirror.com
```

视觉域产物已迁到块存储并软链回仓库路径：

- `results` → `/mnt/data/results/vision`
- `checkpoints` → `/mnt/data/checkpoints/vision`

**磁盘建议**：K5 起结果写 `/mnt/data2`（约 69G）；权重/SQuAD/HF 仍在 `/mnt/data`。GLUE 缓存另占 `/mnt/data/datasets/glue`。

---

## 3. 阶段划分

```mermaid
flowchart LR
  k0[K0_Env]
  k1[K1_Download]
  k2[K2_Split_Eval]
  k3[K3_Prune_Backend]
  k4[K4_Dense_Oneshot]
  k5[K5_Iter_Search]
  kg[KG_GLUE]
  k6[K6_SQuAD_Matrix]
  k0 --> k1 --> k2 --> k3 --> k4 --> k5 --> kg --> k6
```

> K0–K5 已在 **SQuAD** 上完成管线冒烟；按导师要求，**正式小矩阵前插入 KG（GLUE）**，再回到 K6（SQuAD）。

### K0 — 环境 `[√]`

- [√] `/mnt/data/{hf,datasets,models,results}`
- [√] [`scripts/env_llm.sh`](../scripts/env_llm.sh)
- [√] 依赖：`transformers` / `datasets` / `accelerate` / `evaluate`
- [√] 系统盘清理：迁 `results`/`checkpoints`；移除未用 miniconda；`venv` 改指系统 Python 3.10

### K1 — 下载 `[√]`（SQuAD）；GLUE 三任务待齐

- [√] Qwen2.5-1.5B-Instruct → `/mnt/data/models/...`（约 2.9G）
- [√] SQuAD → `/mnt/data/datasets/squad`（train 130319 / official val 11873）
- [√] GLUE → `/mnt/data/datasets/glue`（**正式标准三任务：SST-2 + RTE + QNLI**）
- [√] 系统盘不承载大权重（验收：`df -h /` 不明显上涨）

下载脚本：[`scripts/download_qwen_squad.sh`](../scripts/download_qwen_squad.sh)；GLUE：[`scripts/download_qwen_glue.sh`](../scripts/download_qwen_glue.sh)（KG.5 前须覆盖三任务）。

### K2 — 协议 + 评估骨架 `[√]`（SQuAD）

- [√] train/val/test 划分与元数据
- [√] 配置 [`configs/qwen_squad_smoke.yaml`](../configs/qwen_squad_smoke.yaml)
- [√] 评估 [`experiments/run_qwen_squad_eval.py`](../experiments/run_qwen_squad_eval.py) + [`src/utils/qwen_squad_eval.py`](../src/utils/qwen_squad_eval.py)
- [√] 单测：划分可复现；test 不进 selection path

### K3 — Transformer 物理剪枝 `[√]`

- [√] [`src/pruning/transformer_structured_pruning.py`](../src/pruning/transformer_structured_pruning.py)
- [√] 接入 [`src/pruning/pruning_backend.py`](../src/pruning/pruning_backend.py)（`model_type=qwen|transformer|qwen2`）
- [√] 单测：tiny Qwen2 上 MLP/head 剪后 forward 正常、参数量下降（`tests/test_transformer_structured_pruning.py`）

要点：GQA 下 head 剪枝按 **整 KV-group** 保留，保证 `num_key_value_groups` 一致。

### K4 — dense + oneshot 冒烟 `[√]`（SQuAD）

| 模式 | 说明 | 冒烟数字（64 条 carved val，零样本生成，**非正式主表**） |
|------|------|----------------------------------------------------------|
| dense | 未剪枝 | F1/EM ≈ 18.75 |
| oneshot | 全层 MLP intermediate 剪 25%；无恢复 | F1/EM ≈ 18.75；参数 1.54B → 1.25B（约 **1.23x**） |

产物：`/mnt/data/results/qwen_squad_smoke/`
**解读约束**：仅证明管线可跑；**不得**写入论文主结论。

### K5 — iterative + autonomous_search 接线 `[√]`（SQuAD）

| 子项 | 状态 | 内容 | 验收 |
|------|------|------|------|
| K5.1–K5.5 | `[√]` | 四方法冒烟接线 | `/mnt/data2/results/qwen_k5_smoke/` |

| 方法 | 压缩比 | F1/EM（16 条 carved val，非正式） |
|------|--------|----------------------------------|
| dense | 1.00x | 6.25 / 6.25 |
| oneshot | 1.23x | 6.25 / 6.25 |
| iterative_level1 | 1.44x | 6.25 / 6.25 |
| autonomous_search | 1.01x（单层候选） | 6.25 / 6.25 |

说明：生成式 F1 仅证明管线；正式对照改走 **KG（GLUE）** 再回 SQuAD。

复跑（历史冒烟，非当前优先）：

```bash
source scripts/env_llm.sh
python experiments/run_qwen_k5_smoke.py --config configs/qwen_k5_smoke.yaml
```

### KG — GLUE 先看效果 `[√]` 冒烟已跑通；**KG.5 必做**（指标非正式）

> 导师：在 SQuAD 前先做 GLUE。目的：用更短、更稳的分类式 NLU 信号，验证剪枝 + 恢复 + 四方法对照是否「有效果」，再投入 SQuAD 生成式评测成本。
> 任务角色见 **§1.1**。**评测策略（写死）**：`prompt + verbalizer`（标签见 §1.1）；**不用**分类头。

| 子项 | 状态 | 内容 | 验收 |
|------|------|------|------|
| KG.0 | `[√]` | 下载脚本 `scripts/download_qwen_glue.sh`；缓存 `/mnt/data/datasets/glue` | 见 env `LLM_GLUE_DIR` |
| KG.1 | `[√]` | `glue_protocol` + prompt 评估骨架 + 单测 | test 不进 selection |
| KG.2 | `[√]` | `configs/qwen_glue_smoke.yaml`、`experiments/run_qwen_glue_smoke.py` | dense/oneshot 可跑 |
| KG.3 | `[√]` | Level-1 短恢复接到 GLUE LM pack | 只看 carved val |
| KG.4 | `[√]` | 四方法冒烟（**仅 SST-2**）+ chat/SDPA 重跑 | dense acc≈84.4（32 条 carved val）；`/mnt/data2/results/qwen_glue_smoke/` |
| KG.5 | `[√]` | **必做**：SST-2+RTE+QNLI × 1.5x/2x 四方法小扫 | 可读压缩信号已有；产物 `/mnt/data2/results/qwen_glue_kg5/`（~82 min；32 条 carved val；非正式主表） |

复跑（KG.5）：

```bash
source scripts/env_llm.sh
bash scripts/download_qwen_glue.sh
python experiments/run_qwen_glue_kg5.py --config configs/qwen_glue_kg5.yaml
```

复跑（SST-2 冒烟）：

```bash
source scripts/env_llm.sh
bash scripts/download_qwen_glue.sh
python experiments/run_qwen_glue_smoke.py --config configs/qwen_glue_smoke.yaml
```

**门禁**：**KG.5 `[√]`**（三任务可读压缩信号）已通过 → 可开 **K6 SQuAD 小矩阵**。KG.4 仅证明管线；K6-lit 调研文档可与 K6 并行。
**明确不做（KG）**：全 GLUE 九任务大表；把 **SST-2 冒烟 / KG.5 小扫** accuracy 写成论文 LLM 主结论（须标非正式）。三任务表可作过渡证据，生成式主结论仍属 SQuAD。

**结果解读（2026-08-16）**（详情：[WORK_LOG.md](WORK_LOG.md) §4 第 11 条 / [WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md)）：

1. 门禁通过：评测可用；剪枝有代价；恢复有效。
2. Oneshot 无恢复几乎全崩 → 负结果，Level-1 恢复必要。
3. 1.5x 档 iterative 更稳；2.0x 档掉点加大。
4. search vs iterative 已 crossover（禁止「全面更优」）；与 CIFAR regime-dependent 兼容，LLM 侧仅过渡证据。
5. 读表修正：iterative 实测超标（~1.88 / ~2.81）；CE ≠ 任务 acc；n=32。K6 须同预算对齐。

### K6 — SQuAD 小矩阵与报告 `[√]` 小扫已跑（非正式；frozen test 未开）

| 子项 | 状态 | 默认 |
|------|------|------|
| 前置 | `[√]` | **KG.5 通过**（三任务可读信号，2026-08-16） |
| 压缩目标 | `[√]` | **1.5x / 2x / 4x**（MLP-only 名义 4x 实测上限约 3.1–3.6x） |
| seed | `[√]` | 1 seed（小扫） |
| 恢复预算 | `[√]` | 相对 dense 累计压缩已修（`baseline_parameter_count`）；1.5x/2.0x 对齐；禁止 KG.5 叠乘 |
| 冻结 test | `[ ]` | 官方 validation **只评一次**（尚未开） |
| 产物 | `[√]` | `/mnt/data2/results/qwen_k6/` + `k6_summary.json` + WORK_LOG |
| 叙事 | `[√]` | 小扫允许失败；**禁止**「search 全面更优」；F1 主表须标 n=16 carved val |

**结果解读（2026-08-16；含后续 LoRA）**：

1. **预算对齐成功**：1.5x/2.0x 实测 ≈ 目标；不再出现 KG.5 式 ~2.8x 超剪。
2. **dense 可读**：chat+SDPA 下 F1≈30.6（16 条）/ ≈25.5（64 条）；CE finite。
3. **短 SGD / 加深 SGD 不足（负对照）**：1 epoch 或 512×4 SGD 后剪枝 F1≈0（CE 仍有限）→ 弱恢复撑不起生成式 QA。
4. **LoRA@1.5x Informal 可读**（8192×2；n=64；产物 `qwen_k6_recover_lora_1p5x*`）：dense 25.5 / oneshot 30.6 / iterative **38.9** / search 30.0；均 1.50x。本格 **iterative ≥ oneshot ≈ search**。
5. **4x 名义不可达（MLP-only）**：oneshot/search≈3.07x，iterative≈3.62x；须记上限或扩剪枝单元。
6. **不说明**：非正式主表；不能证 search 优于 iterative；不能替代多 seed / frozen test；不扩 2x。

复跑：

```bash
source scripts/env_llm.sh
python experiments/run_qwen_k6.py --config configs/qwen_k6.yaml
# LoRA Informal（dense+oneshot）：
python experiments/run_qwen_k6.py --config configs/qwen_k6_recover_lora_1p5x.yaml
# LoRA Informal（iterative+search）：
python experiments/run_qwen_k6.py --config configs/qwen_k6_recover_lora_1p5x_methods.yaml
```

### K6-lit — 外部压缩 baseline（自动搜索 vs 人工设计）`[~]` 文献短表已写

> 导师要求：调研他人模型压缩工作作 baseline，看自动搜索是否优于人工设计。
> 论文位置：[PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) §8；短表：[K6_LIT_BASELINE_SHORTLIST.md](K6_LIT_BASELINE_SHORTLIST.md)。

| 子项 | 状态 | 内容 |
|------|------|------|
| 调研 | `[√]` | 短表已按质量门禁勾选 LTH / Wanda / SparseGPT / 结构化 LLM |
| 选型 | `[√]` | 先 Related Work；复现挂 SQuAD/GLUE 同预算（F1 已 Informal 可读） |
| 对照问题 | `[~]` | 内部 Informal：本格 iterative ≥ search；外部 lit 复现仍可选 |
| 验收 | `[~]` | 短表已标注 venue/依据；外部复现实验未开 |

**顺序**：调研可与 KG 并行；外部复现可选，非挡论文收口。

---

## 4. 与视觉域叙事的衔接

CIFAR 定稿：**regime-dependent**（≤4x iterative 略稳；≥8x search 更高；机制随压缩率变）。

LLM 上（更新后口径）：

1. GLUE KG.5：过渡信号 + crossover 片段（非正式）
2. SQuAD：弱恢复失败（负对照）；LoRA 后 1.5x Informal 可读，**本格未支持 search 优于 iterative**
3. 不把 LLM Informal 升级为与 CIFAR 对等的主表

论文位置：视觉为主结果；LLM 为 **第二域迁移 / 讨论**（GLUE 短 NLU 过渡 → SQuAD Informal），见 [PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) 与本文 §1.1。

---

## 5. 代码与配置索引

| 角色 | 路径 |
|------|------|
| 环境 | `scripts/env_llm.sh`（结果 → `/mnt/data2`） |
| SQuAD 下载 | `scripts/download_qwen_squad.sh` |
| K4 SQuAD 冒烟 | `configs/qwen_squad_smoke.yaml` / `experiments/run_qwen_squad_eval.py` |
| K5 SQuAD 四方法 | `configs/qwen_k5_smoke.yaml` / `experiments/run_qwen_k5_smoke.py` |
| SQuAD 协议 / 评估 / LM pack | `src/utils/squad_protocol.py` / `qwen_squad_eval.py` / `qwen_train_data.py` |
| LM Level-1 恢复 | `src/recovery/qwen_lm_recovery.py`（含 `run_configured_recovery`） |
| LM LoRA 恢复 | `src/recovery/qwen_lora_recovery.py` |
| 剪枝后端 | `src/pruning/transformer_structured_pruning.py` |
| **KG GLUE** | `configs/qwen_glue_smoke.yaml`、`experiments/run_qwen_glue_smoke.py`、`src/utils/glue_protocol.py`、`qwen_glue_eval.py`、`qwen_glue_train_data.py`、`src/experiments/qwen_glue_comparison.py` |
| K4/K5 产物 | `/mnt/data/results/qwen_squad_smoke/`、`/mnt/data2/results/qwen_k5_smoke/` |
| KG 产物 | `/mnt/data2/results/qwen_glue_smoke/`、`/mnt/data2/results/qwen_glue_kg5/` |
| K6 LoRA Informal | `/mnt/data2/results/qwen_k6_recover_lora_1p5x/`、`..._methods/` |

---

## 6. 验收清单

### 已验收（K0–K5，SQuAD 管线）`[√]`

- [√] 大文件仅在 `/mnt/data`（+ K5 产物 `/mnt/data2`）；仓库无权重提交
- [√] SQuAD 划分可复现；test 不进 selection
- [√] 物理剪枝 + oneshot/iterative/search 冒烟可跑
- [√] 相关单测通过（含 `tests/test_qwen_k5_recovery.py`）

### 待验收（KG → K6）`[ ]`

- [√] **KG.0–KG.4**：GLUE SST-2 四方法冒烟管线 + 协议单测
- [√] **评测稳定**：chat template + 去掉 eager attn；dense acc>0 且 CE finite（重跑 2026-08-16）
- [√] **KG.5 必做**：SST-2+RTE+QNLI × 1.5x/2x；dense acc>0 / CE finite；oneshot 无恢复崩、iterative/search 部分恢复（`/mnt/data2/results/qwen_glue_kg5/`）；**过渡证据结论已记入 WORK_LOG**
- [√] **开 K6 小扫**：SQuAD 1.5x–4x；预算对齐已修；产物 `qwen_k6`（短恢复 F1 塌 = 负对照）
- [ ] frozen test 报告与 fingerprint / manifest（延后）
- [√] WORK_LOG 已记 KG.5 / K6 / LoRA Informal 含义；冒烟与 Informal 指标不进主结论
- [√] K6-lit：外部压缩调研短表（过质量门禁；标注 venue/依据）
- [√] LoRA 恢复后 Informal 可比 search vs iterative（本格 iterative ≥ search；非正式）

---

## 7. 风险与已知债

| 项 | 说明 | 处理 |
|----|------|------|
| Instruct 零样本乱码 | 曾因 `attn_implementation=eager` + 无 chat template | **已修**：SDPA + chat template；dense 冒烟 acc≈84% |
| SQuAD 零样本 F1 偏低 | 生成式评测噪声大、成本高 | **先 GLUE 三任务看信号**；再回 SQuAD 优化模板/恢复 |
| Instruct + GLUE | 分类任务需固定 prompt+verbalizer | §1.1 写死三任务标签；禁止中途混比 |
| 显存 | 1.5B 全参 Adam 紧；LoRA 可训 | 默认 LoRA 恢复；全参 SGD 仅作负对照 |
| 盘余量 | `/mnt/data` 偏紧 | GLUE 缓存写 `/mnt/data/datasets/glue`；结果写 `/mnt/data2` |
| 评估速度 | SQuAD 全量生成慢 | Informal 用 n=64；正因如此 KG 优先 |
| **GPU / 加卡** | 本机基线 **1×4090**；LoRA Informal 显存约 10–13GB/24GB | **仍单卡即可**。仅当单卡预估 ≥约 3h 且可按 target 拆时才请用户加第 2 卡。规则见 [CLAUDE.md](../CLAUDE.md) |

**本趟资源判定（2026-08-16）**：仍单卡即可。

---

## 8. 下一步（立即）

1. [√] **稳定 GLUE 评测**：chat template + SDPA + CE float32（SST-2 已重跑，dense acc≈84%）
2. [√] **下载补齐 RTE / QNLI**（与 SST-2 同缓存协议）
3. [√] **KG.5（必做）**：SST-2+RTE+QNLI × 1.5x/2x 四方法小扫；可读信号已有（`qwen_glue_kg5`）
4. [√] **K6（SQuAD）小矩阵**：预算对齐 + chat；`/mnt/data2/results/qwen_k6/`（短恢复 F1 塌）
5. [√] **加深 SGD 1.5x**：负对照（`qwen_k6_recover_1p5x`）
6. [√] **LoRA Informal 1.5x**：R1 + iterative/search（`qwen_k6_recover_lora_1p5x*`）
7. [√] **K6-lit**：短表 [K6_LIT_BASELINE_SHORTLIST.md](K6_LIT_BASELINE_SHORTLIST.md)
8. [ ] 论文口径收口：PAPER_RESULTS_OUTLINE / EVIDENCE_PACK 同步；**不扩 2x**；frozen test 延后

---

## 9. 相关文档

- [PHASE_J_QWEN_PLAN.md](PHASE_J_QWEN_PLAN.md) — 规划占位 `[√]`（执行顺序以本文为准：先 GLUE）
- [EVIDENCE_PACK.md](EVIDENCE_PACK.md) — 视觉域证据包
- [PAPER_RESULTS_OUTLINE.md](PAPER_RESULTS_OUTLINE.md) — 论文提纲
- [P2_EXECUTION_PLAN.md](P2_EXECUTION_PLAN.md) §P2.9
- [WORK_LOG.md](WORK_LOG.md) / [WORK_LOG_BRIEF.md](WORK_LOG_BRIEF.md)
- [CLAUDE.md](../CLAUDE.md) — 长实验规范；**LLM / GPU 与配置优化告知**
