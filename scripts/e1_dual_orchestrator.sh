#!/usr/bin/env bash
# Wait for 20% checkpoint, then wait for 2 GPUs, then launch dual E1 and merge when done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
FLAG="$MAIN_OUT/e1_20pct_done.flag"
LOG="$MAIN_OUT/e1_dual_orchestrator.log"
GPU_WAIT_SEC="${E1_GPU_WAIT_SEC:-60}"

log() { echo "[orchestrator] $(date -Is) $*" | tee -a "$LOG"; }

log "waiting for 20% flag: $FLAG"
while [[ ! -f "$FLAG" ]]; do
  sleep 60
done
log "20% complete — waiting for second GPU (nvidia-smi -L count >= 2)"

while true; do
  n=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${n:-0}" -ge 2 ]]; then
    log "detected $n GPU(s); launching dual E1"
    break
  fi
  log "only $n GPU(s); recheck in ${GPU_WAIT_SEC}s (add second GPU when ready)"
  sleep "$GPU_WAIT_SEC"
done

"$ROOT/scripts/launch_e1_dual_after_20.sh" 2>&1 | tee -a "$LOG"

log "waiting for GPU0/GPU1 jobs to finish"
while pgrep -f "run_e1_oneshot_curve.py.*e1_oneshot_curve" >/dev/null; do
  sleep 120
done

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"

log "merging shard into main output"
python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --merge-from "$SHARD_OUT" \
  --resume 2>&1 | tee -a "$LOG"

log "orchestrator finished; check $MAIN_OUT/e1_summary.json"
