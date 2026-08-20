# E report template (copy per experiment)

Match the experiment-plan PDF **项目 / 设置** block. Fill every row; do not free-form replace this table.

```markdown
# E{n}. {PDF title}

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E{n} |
| 目的 | |
| Model | |
| Method / Compression | |
| Evaluation | |
| Seeds | |
| GPU | |
| 优先级 | P0 / P1 / P2 |
| 成功条件 | (from PDF) |
| status | pending / running / done / proxy |

## 输出

(PDF: 输出 / 得到 / 核心图 / 核心指标)

## 记录表

(main numeric table)

## 结论（对照成功条件）

- 满足 / 不满足 / 部分满足:
- Gate (if any):
```

## Machine file `e{n}_summary.json`

Required keys:

`experiment_id`, `title`, `purpose`, `model`, `method`, `compression_or_sparsity`,
`evaluation`, `seeds`, `gpu`, `priority`, `success_criteria`, `status`, `outputs`,
`records`, `notes_proxy`

Proxy runs: keep `experiment_id` as `E{n}`, set `"spec": "proxy_1.5B"` and `"pdf_model": "Qwen2.5-3B-Instruct"`.
