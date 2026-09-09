# E1_soft_eval_lite Soft-Eval Ladder (NOT Gate A)

## Setup

| Item | Value |
|------|-------|
| Model | Qwen2.5-3B (/mnt/data/models/Qwen2.5-3B) |
| Weights | Wanda oneshot re-applied (same pruning as E1); Recovery None |
| Soft dims | Math_lite, Instruction_lite, Code_lite |
| Sparsity | 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70% |
| status | done |
| Gate | **Does not enter Gate A**; formal E1/E2 unchanged |

## Scores

| Sparsity | Math_lite | Instruction_lite | Code_lite |
|------|------|------|------|
| dense | 0.9844 | 0.8514 | 0.2500 |
| 10% | 0.8750 | 0.7390 | 0.2500 |
| 20% | 0.5625 | 0.6225 | 0.2500 |
| 30% | 0.1250 | 0.5703 | 0.2500 |
| 40% | 0.0938 | 0.2169 | 0.2500 |
| 50% | 0.0312 | 0.4056 | 0.0625 |
| 60% | 0.0156 | 0.4016 | 0.1250 |
| 70% | 0.0469 | 0.4016 | 0.1250 |

## Notes

- Sentiment = SST-2 **classification accuracy** (not Wanda LM calibration NLL).
- `*_easy` = same-bank softer limits; `*_soft` / `*_lite` = hybrid easier tasks (维名不变).
- See `docs/process/NLP_EVAL_TASK_SURVEY.md` and `docs/results/E1_eval_ladder_all_banks.md`.
- Resume file: `soft_eval_checkpoint.json` (cell = sparsity|dim).
