# E1_soft_eval_soft Soft-Eval Ladder (NOT Gate A)

## Setup

| Item | Value |
|------|-------|
| Model | Qwen2.5-3B (/mnt/data/models/Qwen2.5-3B) |
| Weights | Wanda oneshot re-applied (same pruning as E1); Recovery None |
| Soft dims | PPL_soft, Math_soft, Knowledge_soft, Reasoning_soft, Instruction_soft, Code_soft |
| Sparsity | 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70% |
| status | done |
| Gate | **Does not enter Gate A**; formal E1/E2 unchanged |

## Scores

| Sparsity | PPL_soft | Math_soft | Knowledge_soft | Reasoning_soft | Instruction_soft | Code_soft |
|------|------|------|------|------|------|------|
| dense | 19.5322 | 0.0625 | 0.7344 | 0.7031 | 0.3750 | 0.6875 |
| 10% | 26.6179 | 0.0625 | 0.7188 | 0.7344 | 0.3125 | 0.4375 |
| 20% | 36.9532 | 0.0625 | 0.6875 | 0.7188 | 0.2500 | 0.3125 |
| 30% | 54.2662 | 0.0000 | 0.5781 | 0.6875 | 0.1250 | 0.0625 |
| 40% | 83.8635 | 0.0312 | 0.5312 | 0.7500 | 0.1875 | 0.1250 |
| 50% | 136.5746 | 0.0000 | 0.4844 | 0.7500 | 0.1875 | 0.0000 |
| 60% | 233.8574 | 0.0000 | 0.4062 | 0.6719 | 0.0625 | 0.0000 |
| 70% | 552.0293 | 0.0000 | 0.3594 | 0.7031 | 0.1875 | 0.0000 |

## Notes

- Sentiment = SST-2 **classification accuracy** (not Wanda LM calibration NLL).
- `*_easy` = same-bank softer limits; `*_soft` = hybrid (same-bank or new easier tasks).
- See `docs/process/NLP_EVAL_TASK_SURVEY.md` and `docs/results/E1_eval_ladder_all_banks.md`.
- Resume file: `soft_eval_checkpoint.json` (cell = sparsity|dim).
