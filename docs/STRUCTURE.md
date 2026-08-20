# Repository structure (PDF §23 mapping)

Program PDFs: [refs/Autonomous_Lottery_Ticket_Discovery.pdf](refs/Autonomous_Lottery_Ticket_Discovery.pdf),
[refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf](refs/Autonomous_Lottery_Ticket_Discovery_experiment_plan.pdf).

Experiment progress: [EXPERIMENT_E_MAP.md](EXPERIMENT_E_MAP.md)（**唯一 ID 标准**：E0–E14）。
Report layout: [E_REPORT_TEMPLATE.md](E_REPORT_TEMPLATE.md).
Work log: [WORK_LOG.md](WORK_LOG.md)（按 E ID）。

## PDF `autonomous_lottery/` -> this repo

| PDF path | This repo |
|----------|-----------|
| `compression/wanda\|sparsegpt\|structured\|regrowth` | `src/pruning/` (structured exists; wanda/sparsegpt/regrowth stubs) |
| `profiling/*` | `src/frontier/` + `src/profiling/` (stubs for gap/capability/governance) |
| `ticket/*` | `src/autonomous_search.py` + `src/ticket/` stubs |
| `recovery/*` | `src/recovery/` (+ Stage C stubs) |
| `controller/*` | `src/controller/` (+ Stage D stubs) |
| `evaluation/*` | `src/evaluation/` (+ capability/governance/cost stubs) |
| `state/*` | `src/state/` stubs |
| `scripts/run_stage_*.sh` | `scripts/run_stage_*.sh` |

Python package root remains **`src/`** (not renamed) to keep imports stable.

## Main tree vs archive

- **Main:** Stage A–D configs/experiments, core `src/`, `tests/`, E docs（E-MAP / WORK_LOG / e_reports）、交付页。
- **Archive:** [../archive/README.md](../archive/README.md) — 旧 Phase*/EVIDENCE_PACK、旧 yaml、旧 runners。

## Results layout (not in Git)

```text
/mnt/data2/results/E{n}_{slug}/
  e{n}_report.md
  e{n}_summary.json
  figures/
```

Human copies under `docs/e_reports/`.
