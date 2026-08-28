#!/usr/bin/env bash
# Resume E1 shard 70% on single GPU from partial checkpoint (PPL/Math/Knowledge may be done).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
SHARD_CKPT="$SHARD_OUT/e1_checkpoint.json"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

if [[ ! -f "$SHARD_CKPT" ]]; then
  echo "[ERROR] missing shard checkpoint: $SHARD_CKPT" >&2
  exit 1
fi

if pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
  echo "[ERROR] E1 curve already running" >&2
  pgrep -af "run_e1_oneshot_curve.py" || true
  exit 1
fi

mkdir -p "$SHARD_OUT"
echo "[INFO] single-GPU resume shard sparsity 0.70 from $SHARD_CKPT"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
  --resume \
  --sparsities 0.7 \
  >> "$SHARD_OUT/e1_gpu1.log" 2>&1 &
echo "E1_70_PID=$!"
echo "[OK] log: $SHARD_OUT/e1_gpu1.log"
