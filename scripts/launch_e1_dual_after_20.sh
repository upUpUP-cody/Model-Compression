#!/usr/bin/env bash
# E1 dual-GPU 3+3 launch: GPU0 20/30/40%, GPU1 50/60/70%.
# Prerequisites: second GPU available; no conflicting E1 processes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
CKPT="$MAIN_OUT/e1_checkpoint.json"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] missing checkpoint: $CKPT" >&2
  exit 1
fi

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${gpu_count:-0}" -lt 2 ]]; then
  echo "[ERROR] need >= 2 GPUs, found ${gpu_count:-0}" >&2
  exit 1
fi

# Stop any legacy E1 runner processes (fresh launch only)
LEGACY_PIDS=$(pgrep -f "run_e1_oneshot_curve.py" || true)
if [[ -n "${LEGACY_PIDS:-}" ]]; then
  echo "[INFO] sending SIGTERM to E1 processes: $LEGACY_PIDS"
  kill -TERM $LEGACY_PIDS || true
  sleep 10
fi

if [[ -f "$MAIN_OUT/e1_checkpoint.pre_dual.json" ]]; then
  cp -a "$CKPT" "$MAIN_OUT/e1_checkpoint.pre_launch_$(date +%Y%m%d_%H%M%S).json"
else
  cp -a "$CKPT" "$MAIN_OUT/e1_checkpoint.pre_dual.json"
fi
mkdir -p "$MAIN_OUT" "$SHARD_OUT"

echo "[INFO] GPU0: sparsities 20/30/40%"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --resume \
  --sparsities 0.2,0.3,0.4 \
  > "$MAIN_OUT/e1_gpu0.log" 2>&1 &
echo "GPU0_PID=$!"

GPU1_SEED_ARGS=()
SHARD_CKPT="$SHARD_OUT/e1_checkpoint.json"
if [[ ! -f "$SHARD_CKPT" ]]; then
  echo "[INFO] GPU1: seed from main checkpoint (cold shard)"
  GPU1_SEED_ARGS=(--seed-checkpoint "$CKPT")
else
  echo "[INFO] GPU1: shard checkpoint exists; resume without re-seed"
fi

echo "[INFO] GPU1: sparsities 50/60/70%"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
  --resume \
  "${GPU1_SEED_ARGS[@]}" \
  --sparsities 0.5,0.6,0.7 \
  > "$SHARD_OUT/e1_gpu1.log" 2>&1 &
echo "GPU1_PID=$!"

if ! pgrep -f "e1_dual_merge_when_done.sh" >/dev/null; then
  nohup "$ROOT/scripts/e1_dual_merge_when_done.sh" >> "$MAIN_OUT/e1_merge_watcher_nohup.log" 2>&1 &
  echo "MERGE_WATCHER_PID=$!"
fi

echo "[OK] dual E1 3+3 launched; merge when both finish:"
echo "  python experiments/stage_a/run_e1_oneshot_curve.py \\"
echo "    --config configs/stage_a/e1_oneshot_curve.yaml \\"
echo "    --merge-from $SHARD_OUT --resume"
