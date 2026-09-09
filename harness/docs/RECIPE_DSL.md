# Recipe DSL

Recipe 描述「怎么压」，Goal 描述「要什么」。Agent 只能改 Recipe 字段，不能内联任意 Python。

## 最小结构

```yaml
recipe_id: global_int8_torchao
backend: torchao          # llm_compressor | autoround | torchao
method: int8_weight_only
layers:
  default:
    weight_bits: 8
    activation_bits: 16
exclude:
  - lm_head
calibration:
  dataset: pile
  samples: 128
```

## 字段约定

| 字段 | 含义 |
|------|------|
| `backend` | 哪个 Adapter |
| `method` | 该后端上的算法名（如 `int8_weight_only`、`gptq`、`autoround`） |
| `layers.default` | 全局默认 bit / group_size |
| `layers."<lo>-<hi>"` | 层段覆盖（mixed） |
| `exclude` | 不量化模块名 |
| `calibration` | 校准数据与样本数 |

## 温和优先示例（本仓）

- `recipes/global_int8.yaml` — 默认首选
- `recipes/global_int8_llmcompressor.yaml` — 另一 backend
- `recipes/conservative_mixed.yaml` — 首尾层更高精度（仍偏保守）

无「必须 4bit」硬目标；INT4 示例可后加，不作为近无损默认路径。

## 与 Goal 的关系

Goal 含 `quality.mode: near_lossless` 与 `compression.ratio_required: false` 时，
Planner 在任一 Recipe 评测进入浮动带后停止，不再为更高压缩比继续搜索。
