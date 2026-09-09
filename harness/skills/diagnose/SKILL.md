---
name: harness-diagnose
description: >-
  Diagnoses why a Compression Harness run exceeded the near-lossless float band and
  suggests a milder next recipe. Use when quality drops, layer sensitivity is asked,
  or plan-next needs a failure hypothesis.
---

# Harness Diagnose

## Instructions

1. Require a `run_id` with `EXPERIMENTS_DIR/<run_id>/artifact.json` (symlink `harness/experiments/`; create via `auto` or compress+evaluate flow first).
2. Run:

```bash
PYTHONPATH=harness/src python -m compression_harness.cli diagnose --run-id run_001_global_int8_torchao
```

3. Surface `hypothesis` and `next_recipe_suggestion`. Do not invent new backends outside adapters.
4. Stub diagnoser only in v0.1; still keep suggestions milder_first.

## Examples

- "为什么这次压缩掉点超了？" → diagnose skill
