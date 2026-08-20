# E3. Compression-Induced Capability Gap

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E3 |
| 目的 | 验证 Compression Gap 是否能描述真实能力损失 |
| Model | proxy_1.5B (PDF: Qwen2.5-3B-Instruct) |
| Method / Compression | Parent Dense；Child oneshot MLP sparsity=50%；Recovery **None** |
| Evaluation | pool n=64；G=NLL_child−NLL_parent；Failure=G>0.05 |
| Seeds | 42 |
| GPU | cuda:0 |
| 优先级 | P0 |
| 成功条件 | Gap 显著预测 degradation（相对 teacher/student NLL、length、random） |
| status | done_proxy |

## 输出

核心指标：correlation /（本跑未算完整 AUROC）。

## 记录表

| Score | corr with Failure |
|-------|-------------------|
| Compression Gap G | nan |
| Parent NLL | nan |
| Child NLL | nan |
| Prompt length | nan |
| Random | nan |

## 结论（对照成功条件）

- **No-Go / 信号弱（本 proxy）**：Gap 未明显高于对照。
- **Gate B 输入**：需与后续 **E9** 一并判断是否放弃 Frontier 数据主线；仅 E3 不足以下最终结论。
- 规格：`proxy_1.5B`；PDF 要 2k–5k pool + 多 child sparsity。
