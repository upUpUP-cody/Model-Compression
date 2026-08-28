#!/usr/bin/env bash
# BBH1024 Phase2 tail: merge shard Reasoning 1024 into main, verify, E2 patch, audit, resume.
# E1 Reasoning re-eval for 0.4 / 0.6 / 0.7 is done by bbh1024_chain_per_gpu.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
E2_MAIN="/mnt/data2/results/E2_iterative_vs_oneshot"
E2_S0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
E2_S1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
LOG="$CALIB_OUT/bbh1024_phase2_split_orch.log"
MERGE_DONE="$CALIB_OUT/bbh1024_e1_merge_done.flag"
MAIN_CFG="$ROOT/configs/stage_a/e1_oneshot_curve.yaml"

log() { echo "[bbh1024-split] $(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$CALIB_OUT/bbh1024_protocol_locked.flag" ]]; then
  log "already locked; launching E2 resume only"
  bash "$ROOT/scripts/launch_e2_dual.sh"
  exit 0
fi

if [[ ! -f "$MERGE_DONE" ]]; then
  log "merge shard 0.5/0.6/0.7 overwrite into main + E1 report"
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
else
  log "E1 merge already done ($MERGE_DONE)"
fi

log "E2 Reasoning patch dual"
mkdir -p "$E2_MAIN" "$E2_S0" "$E2_S1"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e2_reasoning_patch.py" \
  --config "$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml" \
  --checkpoint "$E2_S0/e2_checkpoint.json" \
  --seeds 42,43 --device cuda:0 \
  > "$E2_S0/e2_reasoning_patch.log" 2>&1 &
E2P0=$!
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e2_reasoning_patch.py" \
  --config "$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml" \
  --checkpoint "$E2_S1/e2_checkpoint.json" \
  --seeds 44 --device cuda:0 \
  > "$E2_S1/e2_reasoning_patch.log" 2>&1 &
E2P1=$!
wait "$E2P0" || { log "[ERROR] E2 gpu0 patch failed"; exit 1; }
wait "$E2P1" || { log "[ERROR] E2 gpu1 patch failed"; exit 1; }
log "E2 patches OK"

log "protocol audit J0-J5"
python "$ROOT/experiments/stage_a/run_bbh1024_protocol_audit.py" | tee -a "$LOG"
[[ -f "$CALIB_OUT/bbh1024_protocol_locked.flag" ]] || { log "[ERROR] lock flag missing"; exit 1; }

log "resume E2 dual"
bash "$ROOT/scripts/launch_e2_dual.sh"
if ! pgrep -f "e2_dual_merge_when_done.sh" >/dev/null; then
  nohup bash "$ROOT/scripts/e2_dual_merge_when_done.sh" >> "$E2_MAIN/e2_merge_watcher_nohup.log" 2>&1 &
  log "started e2 merge watcher"
fi
log "[OK] Phase2 tail finished; E2 resumed under 1024+batch4"
