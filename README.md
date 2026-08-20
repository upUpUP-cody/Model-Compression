# Model Compression / Autonomous Lottery Ticket Discovery

**唯一实验编号标准**：[`docs/refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf`](docs/refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf) §31（**E0–E14**）。进度见 [`docs/EXPERIMENT_E_MAP.md`](docs/EXPERIMENT_E_MAP.md)。

| Doc | Role |
|-----|------|
| [docs/EXPERIMENT_E_MAP.md](docs/EXPERIMENT_E_MAP.md) | E0–E14 状态 |
| [docs/NEXT_E8_E9.md](docs/NEXT_E8_E9.md) | **明早执行入口**（E8→E9 + Gate 说明） |
| [docs/WORK_LOG.md](docs/WORK_LOG.md) | 按 E ID 工作日志 |
| [docs/PDF_E_REQUIREMENTS_31_34.md](docs/PDF_E_REQUIREMENTS_31_34.md) | §31–34 要求记录 |
| [docs/E_REPORT_TEMPLATE.md](docs/E_REPORT_TEMPLATE.md) | PDF 式报告排版 |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | PDF 代码映射 |
| [docs/MENTOR_DELIVERY.md](docs/MENTOR_DELIVERY.md) | 导师交付 |
| [archive/README.md](archive/README.md) | 已归档旧计划 / Phase* / EVIDENCE_PACK |

## Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

LLM runs: `source scripts/env_llm.sh`（权重/数据 `/mnt/data`，结果 `/mnt/data2`）。

## Tests

```bash
python -m pytest tests -q
```

## Stage A entry

```bash
bash scripts/run_stage_a.sh
# or:
python experiments/stage_a/run_e0_dense.py --config configs/stage_a/e0_dense.yaml
```

Legacy runners：`archive/experiments_legacy/`（`experiments/run_*.py` 薄 shim 保测试）。
