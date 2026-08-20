#!/usr/bin/env bash
# Stage A: E0 -> E1 -> E2 -> E3 (PDF section 34 first-batch front half)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/env_llm.sh 2>/dev/null || true
PY="${ROOT}/venv/bin/python"
echo "[INFO] Stage A E0"
"$PY" experiments/stage_a/run_e0_dense.py --config configs/stage_a/e0_dense.yaml
echo "[INFO] Stage A E1"
"$PY" experiments/stage_a/run_e1_oneshot_curve.py --config configs/stage_a/e1_oneshot_curve.yaml
echo "[INFO] Stage A E2"
"$PY" experiments/stage_a/run_e2_iterative_vs_oneshot.py --config configs/stage_a/e2_iterative_vs_oneshot.yaml
echo "[INFO] Stage A E3"
"$PY" experiments/stage_a/run_e3_compression_gap.py --config configs/stage_a/e3_compression_gap.yaml
echo "[OK] Stage A E0-E3 finished"
