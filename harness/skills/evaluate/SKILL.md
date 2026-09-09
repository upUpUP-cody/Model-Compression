---
name: harness-evaluate
description: >-
  Evaluates a compressed run against a near-lossless Goal (hellaswag@64 acc_norm vs dense).
  Use when the user asks to evaluate a harness recipe/run, check quality float band,
  or compare compressed score to baseline for Compression Harness.
---

# Harness Evaluate

## Instructions

1. Load Goal (`quality.mode: near_lossless`, `max_relative_drop`, `evaluation.limit: 64`).
2. Run:

```bash
source venv/bin/activate && source scripts/env_llm.sh
PYTHONPATH=harness/src:src python -m compression_harness.cli evaluate \
  --goal harness/recipes/goal_near_lossless_small.yaml \
  --model /mnt/data/models/Qwen2.5-3B
# or against a compressed dir:
PYTHONPATH=harness/src:src python -m compression_harness.cli evaluate \
  --goal harness/recipes/goal_near_lossless_small.yaml \
  --compressed-dir /mnt/data2/results/harness_experiments/run_001_global_int8_torchao/compressed
# or via symlink: harness/experiments/run_001_global_int8_torchao/compressed
```

3. Report `baseline_score`, `score`, `relative_drop`, `near_lossless_ok`.
4. `--dry-run` uses stub scores (no lm-eval).

## Examples

- "这个 INT8 recipe 还在近无损带内吗？" → evaluate skill
