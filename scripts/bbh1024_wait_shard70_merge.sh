#!/usr/bin/env bash
# Wait for shard 70% Reasoning 1024, then merge into main and generate E1 report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARD_LOG="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_reasoning_patch_shard.log"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
LOG="$CALIB_OUT/bbh1024_wait_shard70_merge.log"
MERGE_LAUNCHED="$CALIB_OUT/bbh1024_shard70_merge_launched.flag"
POLL_SEC="${POLL_SEC:-60}"

log() { echo "[bbh1024-wait70] $(date -Is) $*" | tee -a "$LOG"; }

shard70_done() {
  grep -q 'sparsity=0\.70 Reasoning=' "$SHARD_LOG" 2>/dev/null
}

if [[ -f "$CALIB_OUT/bbh1024_e1_complete.flag" ]]; then
  log "E1 already complete"
  exit 0
fi

if [[ -f "$MERGE_LAUNCHED" ]]; then
  log "merge already launched ($MERGE_LAUNCHED)"
  exit 0
fi

log "waiting for shard 70% Reasoning 1024 (poll ${POLL_SEC}s); GPU0 main patch should be stopped"
while ! shard70_done; do
  if pgrep -f "run_e1_reasoning_patch.py.*--shard-only" >/dev/null; then
    tail -1 "$SHARD_LOG" 2>/dev/null | grep -o 'Running generate_until.*' | head -c 120 || true
    echo "" >> /dev/null
  else
    log "[WARNING] shard-only patch not running; checking log anyway"
  fi
  sleep "$POLL_SEC"
done

log "shard 70% written to log; verify checkpoint then merge"
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT"
python "$ROOT/scripts/bbh1024_verify_e1_1024_checkpoint.py" --phase post-chain \
  --main "/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json" \
  --shard "/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json" \
  | tee -a "$LOG"

date -Is > "$MERGE_LAUNCHED"
log "launch E1 merge + report"
bash "$ROOT/scripts/bbh1024_e1_merge_and_report.sh" 2>&1 | tee -a "$LOG"
log "[OK] E1 merge pipeline finished"
