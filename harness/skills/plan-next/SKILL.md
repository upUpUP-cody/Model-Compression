---
name: harness-plan-next
description: >-
  Plans the next Compression Harness recipe under milder_first and near-lossless
  stop rules, or runs the auto loop (real torchao INT8 by default). Use when the user
  asks what experiment to run next, to auto-compress, or to stop once quality is
  within the float band.
---

# Harness Plan-Next

## Instructions

1. Stop if the last run was near-lossless OK — do **not** chase higher compression.
2. Suggest next recipe:

```bash
PYTHONPATH=harness/src python -m compression_harness.cli plan-next
PYTHONPATH=harness/src python -m compression_harness.cli plan-next --last-ok
```

3. Full auto (real, ~20-45 min on 4090):

```bash
source venv/bin/activate && source scripts/env_llm.sh
PYTHONPATH=harness/src:src python -m compression_harness.cli auto \
  --goal harness/recipes/goal_near_lossless_small.yaml
```

4. `--dry-run` keeps Phase A stubs. Respect `search.max_trials` and `ratio_required: false`.

## Examples

- "自动近无损压缩一下" → `auto`
- "下一步做什么" → `plan-next`
