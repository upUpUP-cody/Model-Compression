# Archive index

Historical / non-essential files. **Main numbering is PDF E0–E14 only** (`docs/EXPERIMENT_E_MAP.md`).
Nothing here is deleted; restore with `git mv` if needed.

| Archived path | Original | Reason |
|---------------|----------|--------|
| `docs/EVIDENCE_PACK.md` | `docs/` | 旧「Phase H」证据包；CIFAR 详表仍可查，不作主编号 |
| `docs/PHASE_J_QWEN_PLAN.md` | `docs/` | 旧 Phase J |
| `docs/PHASE_K_QWEN_PLAN.md` | `docs/` | 旧 Phase K / KG.* |
| `docs/P2_EXECUTION_PLAN.md` | `docs/` | 旧 P2 |
| `docs/CIFAR_SEARCH_UNDERCOMPRESSION.md` | `docs/` | 历史根因笔记 |
| `docs/GPU_WORKFLOW.md` | `docs/` | 旧 GPU 交接 |
| `docs/K6_LIT_BASELINE_SHORTLIST.md` | `docs/` | 文献短表 |
| `docs/PAPER_RESULTS_OUTLINE.md` | `docs/` | 旧论文提纲 |
| `root/EXECUTION_PLAN.md` | repo root | 由 E-MAP 取代 |
| `configs_legacy/*.yaml` | `configs/` | 预 Stage 配置 |
| `experiments_legacy/*.py` | `experiments/` | 预 Stage runners |

**Still needed for tests:** thin shims under `experiments/run_*.py` load from `experiments_legacy/`.
Config paths in tests point at `archive/configs_legacy/`.

Program PDFs live in **`docs/refs/`** (not archived).
