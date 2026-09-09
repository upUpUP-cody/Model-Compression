---
name: harness-iterate
description: >-
  Runs Compression Harness persistent short-step iterative compression (layer-wise
  prune/quantize plugins) under lossless (<=1% drop) or lossy_bounded (<=5% drop)
  goals. Use when the user asks to iterate compress, resume an iter_ run, short-step
  prune then quantize, or stop and revert when quality drops past the threshold.
---

# Harness Iterate

## Instructions

1. Prefer iterative goals:
   - `harness/recipes/goal_lossless_iter.yaml` — max drop 1%
   - `harness/recipes/goal_lossy5_iter.yaml` — max drop 5%
2. Default is **stub** plugins (bookkeeping + heuristic score). With `--real` (Phase II):
   `prune_wanda` / `quantize_torchao_layers` / `evaluate_real` + disk checkpoint revert.
3. Commands:

```bash
source venv/bin/activate && source scripts/env_llm.sh
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossless_iter.yaml
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossy5_iter.yaml --real
PYTHONPATH=harness/src:src python -m compression_harness.cli iterate \
  --goal harness/recipes/goal_lossy5_iter.yaml --real --resume iter_YYYYMMDDThhmmssZ
```

4. Behavior: propose short layer step → apply prune or quantize plugin → evaluate → if drop exceeds threshold, **revert to last_good and stop**; otherwise continue until max_rounds or search exhausted.
5. Artifacts: `/mnt/data2/results/harness_experiments/iter_*/`（或 symlink `harness/experiments/iter_*/`）含 `state.json`、`ckpt/rXXX/`、`rounds/rXXX.json`、`report.md`。
6. Also update/report via `harness-report` skill if needed.

## Examples

- "按无损档短步迭代压缩" → lossless goal iterate
- "掉点超过 5% 就停并回退" → lossy5 goal (built-in gate)
- "真跑 Wanda/INT8 短步" → add `--real`
