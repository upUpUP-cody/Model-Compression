#!/usr/bin/env bash
# E2 Phase A: finish in-place partials on old seed digests.
# GPU0: seeds 42,43 --targets 0.5  (resume 42 iterative 50; may start 43@0.5)
# GPU1: seeds 44 --targets 0.4     (resume 44 iterative 40)
# Does NOT use --fresh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
SHARD0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
SHARD1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
CFG="$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml"
LOCK="/mnt/data2/results/bbh_maxtok1024_calib/bbh1024_protocol_locked.flag"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

if [[ ! -f "$LOCK" ]]; then
  echo "[ERROR] missing $LOCK; run protocol audit first" >&2
  exit 1
fi

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${gpu_count:-0}" -lt 2 ]]; then
  echo "[ERROR] need >= 2 GPUs, found ${gpu_count:-0}" >&2
  exit 1
fi

mkdir -p "$MAIN_OUT" "$SHARD0" "$SHARD1"

if pgrep -f "run_e2_iterative_vs_oneshot.py" >/dev/null; then
  echo "[ERROR] E2 process already running; refuse Phase A launch" >&2
  pgrep -af "run_e2_iterative_vs_oneshot.py" || true
  exit 1
fi

echo "[INFO] Phase A GPU0: seeds 42,43 targets 0.5 -> $SHARD0"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$CFG" \
  --seeds 42,43 \
  --targets 0.5 \
  --resume \
  --output-root "$SHARD0" \
  > "$MAIN_OUT/e2_phase_a_gpu0.log" 2>&1 &
echo "GPU0_PID=$!"

echo "[INFO] Phase A GPU1: seeds 44 targets 0.4 -> $SHARD1"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$CFG" \
  --seeds 44 \
  --targets 0.4 \
  --resume \
  --output-root "$SHARD1" \
  > "$MAIN_OUT/e2_phase_a_gpu1.log" 2>&1 &
echo "GPU1_PID=$!"

echo "[OK] Phase A launched. Logs: $MAIN_OUT/e2_phase_a_gpu0.log $MAIN_OUT/e2_phase_a_gpu1.log"
