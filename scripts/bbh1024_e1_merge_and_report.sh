#!/usr/bin/env bash
# E1-only tail: merge shard Reasoning 1024 into main, verify, regenerate report (no E2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"

MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
LOG="$CALIB_OUT/bbh1024_e1_merge_report.log"
MERGE_DONE="$CALIB_OUT/bbh1024_e1_merge_done.flag"
E1_DONE="$CALIB_OUT/bbh1024_e1_complete.flag"
MAIN_CFG="$ROOT/configs/stage_a/e1_oneshot_curve.yaml"

log() { echo "[bbh1024-e1] $(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$E1_DONE" ]]; then
  log "E1 already complete ($E1_DONE)"
  echo "report: $MAIN_OUT/e1_report.md"
  echo "docs:   docs/results/E1_oneshot_sparsity_curve.md"
  exit 0
fi

if [[ ! -f "$MERGE_DONE" ]]; then
  log "merge shard 0.5/0.6/0.7 overwrite into main + regenerate report/figures"
  python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
    --config "$MAIN_CFG" \
    --merge-from "$SHARD_OUT" \
    --merge-overwrite-sparsities 0.5,0.6,0.7 \
    --resume 2>&1 | tee -a "$LOG"

  python "$ROOT/scripts/bbh1024_verify_e1_1024_checkpoint.py" --phase final \
    --main "$MAIN_OUT/e1_checkpoint.json" \
    --shard "$SHARD_OUT/e1_checkpoint.json" \
    | tee -a "$LOG"

  date -Is > "$MERGE_DONE"
  log "E1 merge + verify OK"
fi

date -Is > "$E1_DONE"
log "[OK] E1 complete (Reasoning 1024, all sparsities)"
log "report: $MAIN_OUT/e1_report.md"
log "docs:   docs/results/E1_oneshot_sparsity_curve.md"
log "summary: $MAIN_OUT/e1_summary.json"
