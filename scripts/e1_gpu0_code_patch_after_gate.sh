#!/usr/bin/env bash
# After E0 Code gate passes: patch E1 Code on CUDA0 for dense + 10–60%.
# Does NOT patch 70% Code (defer); does NOT touch GPU1 E1 curve process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
E0_OUT="/mnt/data2/results/E0_dense_baseline"
GATE_FLAG="$E0_OUT/e0_code_gate_passed.flag"
LOG="$MAIN_OUT/e1_code_patch.log"
PHASE_A_DONE="$MAIN_OUT/e1_code_patch_phase_a.flag"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

mkdir -p "$MAIN_OUT" "$SHARD_OUT"
log() { echo "[code-patch] $(date -Is) $*" | tee -a "$LOG"; }

if [[ ! -f "$GATE_FLAG" ]]; then
  log "ERROR: missing E0 gate flag $GATE_FLAG"
  exit 1
fi

if [[ ! -f "$PHASE_A_DONE" ]]; then
  log "phase A: patch main (dense+10/20/30/40) and shard (dense+50/60); skip 70%"
  CUDA_VISIBLE_DEVICES=0 python "$ROOT/experiments/stage_a/run_e1_code_patch.py" \
    --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
    --shard-config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
    --sparsities 0.1,0.2,0.3,0.4,0.5,0.6 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG"
  date -Is > "$PHASE_A_DONE"
  log "phase A done (70% Code deferred)"
else
  log "phase A already done ($PHASE_A_DONE)"
fi

date -Is > "$MAIN_OUT/e1_code_patch_all_done.flag"
log "Code patch finished for dense+10-60; 70% Code not patched"
