---
name: harness-report
description: >-
  Builds Compression Harness post-run reports with per-step timing, bit step size,
  final compression ratio, and multi-dimension score comparison tables. Use when a
  harness auto/compress task finishes, or the user asks for a compression report,
  ratio summary, step timings, or score comparison markdown/json.
---

# Harness Report

## Instructions

1. Prefer generating from existing `EXPERIMENTS_DIR/run_*/artifact.json` (no re-eval). Symlink: `harness/experiments/`.
2. After `auto`, report paths are also in the CLI JSON under `report_paths`.
3. Commands:

```bash
source venv/bin/activate && source scripts/env_llm.sh
# All runs under experiments/
PYTHONPATH=harness/src:src python -m compression_harness.cli report
# One run
PYTHONPATH=harness/src:src python -m compression_harness.cli report \
  --run-id run_001_global_int8_torchao
```

4. Outputs: `EXPERIMENTS_DIR/reports/report_*.json|.md` (default `/mnt/data2/results/harness_experiments/reports/`), plus copies `report.json` / `report.md` in the accepted run dir when present.
5. Report sections: meta, steps (bits / durations / size), final compression ratio, `scores_comparison` (six capability dims + `primary`; unset dims are null).
6. Historical artifacts without `metrics.timing` keep `duration_sec` as null — say so in the reply.
7. Do not invent E1 Gate scores; do not re-run GPU eval unless the user asks.

## Examples

- "任务做完了，出个压缩报告" → `cli report`
- "只汇总 run_001" → `--run-id run_001_global_int8_torchao`
