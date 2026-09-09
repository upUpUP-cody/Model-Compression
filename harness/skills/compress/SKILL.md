---
name: harness-compress
description: >-
  Runs Compression Harness compress skill against a Recipe DSL YAML via registered
  adapters (torchao INT8 real in Phase B; llm_compressor / autoround stubs). Use when
  the user asks to compress a model, quantize to INT8, run a harness recipe, or apply
  milder_first quantization without writing custom compression scripts.
---

# Harness Compress

## Instructions

1. Do **not** invent ad-hoc quantization scripts. Use Recipe YAML under `harness/recipes/`.
2. Prefer `global_int8.yaml` (torchao INT8 weight-only) for near-lossless goals.
3. Run from repo root (real compress):

```bash
source venv/bin/activate && source scripts/env_llm.sh
PYTHONPATH=harness/src:src python -m compression_harness.cli compress \
  --recipe harness/recipes/global_int8.yaml \
  --goal harness/recipes/goal_near_lossless_small.yaml
```

4. Add `--stub` only for dry-run without weight changes.
5. Write outcomes under `EXPERIMENTS_DIR` (`/mnt/data2/results/harness_experiments/`, symlink `harness/experiments/`). Do not alter E1–E9 Gate results.

## Examples

- "用 harness compress 把小模型压成 INT8" → `global_int8.yaml`
