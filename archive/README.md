# Archive index

Historical / non-essential files kept for **active references** or **CIFAR/Non-E reproduction**.
Main numbering is PDF **E0–E14** only (`docs/project/EXPERIMENT_E_MAP.md`).
Restore with `git` if needed. Orphan Qwen K/GLUE runners, early MNIST `exp_*`, and superseded `EXECUTION_PLAN` were removed.

## Docs (`archive/docs/`)

| File | Why kept |
|------|----------|
| `EVIDENCE_PACK.md` | CIFAR formal100 详表；主树多处引用 |
| `PAPER_RESULTS_OUTLINE.md` | `PROJECT_PLAN` 引用 |
| `K6_LIT_BASELINE_SHORTLIST.md` | `PROJECT_PLAN` 引用 |
| `GPU_WORKFLOW.md` | `WORKFLOW` / `ENVIRONMENT` 引用 |
| `P2_EXECUTION_PLAN.md` | 历史 P2 手册；`PROJECT_PLAN` 引用 |
| `CIFAR_SEARCH_UNDERCOMPRESSION.md` | 欠压根因笔记（挂接 EVIDENCE_PACK） |
| `PHASE_J_QWEN_PLAN.md` / `PHASE_K_QWEN_PLAN.md` | 迁移规划史料；命令路径可能过时 |

## Experiments (`archive/experiments_legacy/`)

Shim targets under `experiments/run_*.py`, plus CIFAR baseline reproduction:

- `run_p12_comparison.py` / `run_p12_multiseed.py`
- `run_cifar_p12_comparison.py` / `run_cifar_p12_multiseed.py`
- `run_cifar_recovery_ablation.py` / `run_cifar_crossover_path_ablation.py`
- `exp_cifar_baseline.py`

## Configs (`archive/configs_legacy/`)

- All remaining **CIFAR** yaml (formal100 / ia_prime / crossover / budget_match / lowcomp / baseline / recovery / study-smoke-sweep)
- **MNIST** p12 + `mnist_mlp_autonomous_cpu.yaml` (tests / GPU_WORKFLOW / runner defaults)

**Still needed for tests:** thin shims under `experiments/run_*.py` load from `experiments_legacy/`.
Config paths in tests point at `archive/configs_legacy/`.

Program PDFs live in **`docs/refs/`** (not archived).
