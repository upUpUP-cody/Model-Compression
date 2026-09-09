# E1_soft_eval_side Soft-Eval Ladder (NOT Gate A)

## Setup

| Item | Value |
|------|-------|
| Model | Qwen2.5-3B (/mnt/data/models/Qwen2.5-3B) |
| Weights | Wanda oneshot re-applied (same pruning as E1); Recovery None |
| Soft dims | Sentiment, Math_easy, Reasoning_easy, Knowledge_easy, Instruction_easy, Code_easy |
| Sparsity | 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70% |
| status | done |
| Gate | **Does not enter Gate A**; formal E1/E2 unchanged |

## Scores

| Sparsity | Sentiment | Math_easy | Reasoning_easy | Knowledge_easy | Instruction_easy | Code_easy |
|------|------|------|------|------|------|------|
| dense | 0.8281 | 0.6250 | 0.8438 | 0.2969 | 0.3125 | 0.6875 |
| 10% | 0.8281 | 0.3438 | 0.8125 | 0.2969 | 0.2188 | 0.7500 |
| 20% | 0.8711 | 0.0938 | 0.6562 | 0.3125 | 0.1250 | 0.3750 |
| 30% | 0.7734 | 0.0625 | 0.5000 | 0.2969 | 0.0625 | 0.3750 |
| 40% | 0.6680 | 0.0312 | 0.5312 | 0.2969 | 0.1250 | 0.1250 |
| 50% | 0.6562 | 0.0312 | 0.4062 | 0.2656 | 0.0938 | 0.0000 |
| 60% | 0.5625 | 0.0000 | 0.5625 | 0.3125 | 0.0625 | 0.0000 |
| 70% | 0.4336 | 0.0000 | 0.3750 | 0.2656 | 0.1562 | 0.0000 |

## Notes

- Sentiment = SST-2 **classification accuracy** (not Wanda LM calibration NLL).
- `*_easy` = same-bank softer limits; `*_soft` / `*_lite` = hybrid easier tasks (维名不变).
- **Reasoning_easy protocol fix**: `apply_chat_template=false`, `max_gen_toks=32`, stop `until`, task `bbh_fewshot_boolean_expressions_easy` (flexible-extract). Prior gen256+chat zeros **void**.
- See `docs/process/NLP_EVAL_TASK_SURVEY.md` and `docs/results/E1_eval_ladder_all_banks.md`.
- Resume file: `soft_eval_checkpoint.json` (cell = sparsity|dim).
