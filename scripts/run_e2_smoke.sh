#!/usr/bin/env bash
# E2 single-GPU smoke (limit_override=2, seed=42, target=40%).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1
exec python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml" \
  --smoke \
  --output-root /mnt/data2/results/E2_iterative_vs_oneshot_smoke \
  "$@"
