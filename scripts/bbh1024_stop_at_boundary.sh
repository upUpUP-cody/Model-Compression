#!/usr/bin/env bash
# Wait for main 30% + shard 50% Reasoning patch to finish, then stop Phase2 orchestrator
# before main 40% / shard 60%. Backup checkpoints and launch split orchestrator.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_LOG="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_reasoning_patch_main.log"
SHARD_LOG="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_reasoning_patch_shard.log"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
STOP_FLAG="$CALIB_OUT/bbh1024_boundary_stopped.flag"
LOG="$CALIB_OUT/bbh1024_stop_at_boundary.log"
POLL_SEC="${POLL_SEC:-60}"

log() { echo "[bbh1024-stop] $(date -Is) $*" | tee -a "$LOG"; }

main_done() {
  grep -q 'sparsity=0\.30 Reasoning=' "$MAIN_LOG" 2>/dev/null \
    && ! grep -q 'E1 Reasoning patch sparsity=0\.40' "$MAIN_LOG" 2>/dev/null
}

shard_done() {
  grep -q 'sparsity=0\.50 Reasoning=' "$SHARD_LOG" 2>/dev/null \
    && ! grep -q 'E1 Reasoning patch sparsity=0\.60' "$SHARD_LOG" 2>/dev/null
}

if [[ -f "$STOP_FLAG" ]]; then
  log "already stopped ($STOP_FLAG); skip"
  exit 0
fi

log "waiting for main 30% + shard 50% checkpoint writes (poll ${POLL_SEC}s)"
while true; do
  m_ok=0
  s_ok=0
  main_done && m_ok=1 || true
  shard_done && s_ok=1 || true
  log "main_30_done=$m_ok shard_50_done=$s_ok"
  if [[ "$m_ok" -eq 1 && "$s_ok" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SEC"
done

log "boundary reached; stopping orchestrator and patch runners"
ORCH_PIDS=$(pgrep -f "bbh1024_phase2_orchestrate.sh" || true)
if [[ -n "${ORCH_PIDS:-}" ]]; then
  log "SIGTERM orchestrator: $ORCH_PIDS"
  kill -TERM $ORCH_PIDS || true
  sleep 5
fi

PATCH_PIDS=$(pgrep -f "run_e1_reasoning_patch.py" || true)
if [[ -n "${PATCH_PIDS:-}" ]]; then
  log "SIGTERM patch: $PATCH_PIDS"
  kill -TERM $PATCH_PIDS || true
  sleep 15
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$CALIB_OUT/pre_split_${STAMP}"
mkdir -p "$BACKUP_DIR"
cp -a "$MAIN_OUT/e1_checkpoint.json" "$BACKUP_DIR/e1_checkpoint_main.json"
cp -a "$SHARD_OUT/e1_checkpoint.json" "$BACKUP_DIR/e1_checkpoint_shard.json"
log "backup -> $BACKUP_DIR"

source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT"
python "$ROOT/scripts/bbh1024_verify_e1_1024_checkpoint.py" --phase post-stop \
  | tee -a "$LOG"

date -Is > "$STOP_FLAG"
log "stop complete; launching split orchestrator"
bash "$ROOT/scripts/bbh1024_phase2_orchestrate_split.sh" 2>&1 | tee -a "$LOG"
