#!/usr/bin/env bash
# Re-run Stage A E0-E3 then Stage C E8 (formal_3B). Resumable E8 cells.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_llm.sh
export HF_HUB_DISABLE_XET=1
PY="${ROOT}/venv/bin/python"
LOG_ROOT=/mnt/data2/results
mkdir -p "$LOG_ROOT"

echo "[INFO] formal_3B rerun starting $(date -Is)"
echo "[INFO] model=$LLM_MODEL_DIR"

"$PY" experiments/stage_a/run_e0_dense.py --config configs/stage_a/e0_dense.yaml \
  2>&1 | tee "$LOG_ROOT/rerun_e0.log"
"$PY" experiments/stage_a/run_e1_oneshot_curve.py --config configs/stage_a/e1_oneshot_curve.yaml \
  2>&1 | tee "$LOG_ROOT/rerun_e1.log"
"$PY" experiments/stage_a/run_e2_iterative_vs_oneshot.py --config configs/stage_a/e2_iterative_vs_oneshot.yaml \
  2>&1 | tee "$LOG_ROOT/rerun_e2.log"
"$PY" experiments/stage_a/run_e3_compression_gap.py --config configs/stage_a/e3_compression_gap.yaml \
  2>&1 | tee "$LOG_ROOT/rerun_e3.log"
"$PY" experiments/stage_c/run_e8_random_recovery.py --config configs/stage_c/e8_random_recovery.yaml \
  2>&1 | tee "$LOG_ROOT/rerun_e8.log"

echo "[OK] E0-E3 + E8 finished $(date -Is)"
echo "[NEXT] E9: python experiments/stage_c/run_e9_high_gap_recovery.py --config configs/stage_c/e9_high_gap_recovery.yaml"
