#!/usr/bin/env bash
# Resume E1 dual-GPU 3+3 from checkpoints (no --fresh).
# GPU0: main output dir checkpoint (20/30/40%).
# GPU1: shard checkpoint if present; otherwise seed dense+10% from main once.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
MAIN_CKPT="$MAIN_OUT/e1_checkpoint.json"
SHARD_CKPT="$SHARD_OUT/e1_checkpoint.json"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

if [[ ! -f "$MAIN_CKPT" ]]; then
  echo "[ERROR] missing main checkpoint: $MAIN_CKPT" >&2
  exit 1
fi

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${gpu_count:-0}" -lt 2 ]]; then
  echo "[ERROR] need >= 2 GPUs, found ${gpu_count:-0}" >&2
  exit 1
fi

if pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
  echo "[ERROR] E1 already running; stop it first or wait for completion" >&2
  pgrep -af "run_e1_oneshot_curve.py" || true
  exit 1
fi

mkdir -p "$MAIN_OUT" "$SHARD_OUT"

echo "[INFO] GPU0 resume: sparsities 20/30/40%"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --resume \
  --sparsities 0.2,0.3,0.4 \
  >> "$MAIN_OUT/e1_gpu0.log" 2>&1 &
echo "GPU0_PID=$!"

GPU1_SEED_ARGS=()
if [[ ! -f "$SHARD_CKPT" ]]; then
  echo "[INFO] GPU1: no shard checkpoint; seed once from main"
  GPU1_SEED_ARGS=(--seed-checkpoint "$MAIN_CKPT")
else
  echo "[INFO] GPU1: resume from shard checkpoint"
fi

echo "[INFO] GPU1 resume: sparsities 50/60/70%"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
  --resume \
  "${GPU1_SEED_ARGS[@]}" \
  --sparsities 0.5,0.6,0.7 \
  >> "$SHARD_OUT/e1_gpu1.log" 2>&1 &
echo "GPU1_PID=$!"

if ! pgrep -f "e1_dual_merge_when_done.sh" >/dev/null; then
  nohup "$ROOT/scripts/e1_dual_merge_when_done.sh" >> "$MAIN_OUT/e1_merge_watcher_nohup.log" 2>&1 &
  echo "MERGE_WATCHER_PID=$!"
fi

echo "[OK] dual E1 3+3 resumed"
echo "  logs: $MAIN_OUT/e1_gpu0.log , $SHARD_OUT/e1_gpu1.log"
echo "  checkpoints: $MAIN_CKPT , $SHARD_CKPT (shard written after each completed sparsity)"
