#!/usr/bin/env bash
# E2 dual-GPU launch: GPU0 seeds 42,43; GPU1 seed 44.
# Default --resume. Set E2_FRESH=1 to wipe shard checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
SHARD0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
SHARD1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
CFG="$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${gpu_count:-0}" -lt 2 ]]; then
  echo "[ERROR] need >= 2 GPUs, found ${gpu_count:-0}" >&2
  exit 1
fi

mkdir -p "$MAIN_OUT" "$SHARD0" "$SHARD1"

FRESH_ARGS=()
if [[ "${E2_FRESH:-0}" == "1" ]]; then
  echo "[INFO] E2_FRESH=1: removing shard checkpoints"
  rm -f "$SHARD0/e2_checkpoint.json" "$SHARD1/e2_checkpoint.json"
  FRESH_ARGS=(--fresh)
fi

# Do not kill running E2 unless fresh relaunch requested
if [[ "${E2_FRESH:-0}" == "1" ]]; then
  PIDS=$(pgrep -f "run_e2_iterative_vs_oneshot.py" || true)
  if [[ -n "${PIDS:-}" ]]; then
    echo "[INFO] SIGTERM existing E2 processes: $PIDS"
    kill -TERM $PIDS || true
    sleep 5
  fi
fi

echo "[INFO] GPU0: seeds 42,43 -> $SHARD0"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$CFG" \
  --seeds 42,43 \
  --resume \
  "${FRESH_ARGS[@]}" \
  --output-root "$SHARD0" \
  > "$MAIN_OUT/e2_gpu0.log" 2>&1 &
echo "GPU0_PID=$!"

echo "[INFO] GPU1: seeds 44 -> $SHARD1"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$CFG" \
  --seeds 44 \
  --resume \
  "${FRESH_ARGS[@]}" \
  --output-root "$SHARD1" \
  > "$MAIN_OUT/e2_gpu1.log" 2>&1 &
echo "GPU1_PID=$!"

echo "[OK] E2 dual launched. Logs: $MAIN_OUT/e2_gpu0.log $MAIN_OUT/e2_gpu1.log"
echo "[INFO] Start merge watcher: bash scripts/e2_dual_merge_when_done.sh"
