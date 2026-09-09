# Compression Harness 架构

## 闭环

```text
Goal Spec → Planner → Recipe Compiler → Executor → Deploy → Evaluator → Diagnoser → Memory
                                                                              ↓
                                                                         Planner
```

## 分层

| 层 | 职责 | v0.1 |
|----|------|------|
| Goal Spec | model / hardware / 近无损 SLO / 预算 | YAML + JSON Schema |
| Action Space | 有限 primitive，禁止自由写压缩脚本 | quantize / serve / evaluate / diagnose / compare / profile |
| Adapters | 不重写算法 | llm_compressor / autoround / torchao（接口 stub） |
| Diagnoser | 为何超浮动、层敏感 | stub → 建议更温和 recipe |
| Memory | run 树、hypothesis | JSON 文件树（后期可加 SQLite） |
| Planner | 预算内下一步；近无损合格即停 | 规则 stub |
| Skills | Agent 调用面 | `.cursor/skills` + `harness/skills` |

## 近无损门控（本仓裁剪）

成功条件：**任务指标相对 dense 在浮动带内**，不要求达到某压缩比或 INT4。

自动搜索偏好 `milder_first`：先保守 recipe（如 global INT8），合格即 `Accept_and_stop`。

## 包布局

- `compression_harness.controller` — 编排 auto 循环
- `compression_harness.planner` — 选下一 recipe / 停机
- `compression_harness.executor` — 调 Adapter 执行 Recipe
- `compression_harness.evaluator` — 相对 dense 算 drop、判 near_lossless
- `compression_harness.diagnoser` — 超阈值归因（stub）
- `compression_harness.memory` — 读写 `experiments/run_*`
- `compression_harness.actions` — primitive 声明
- `compression_harness.adapters` — 后端接口
