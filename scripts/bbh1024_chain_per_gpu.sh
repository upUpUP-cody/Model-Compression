#!/usr/bin/env bash
# Per-GPU chain: main 30% -> 40% on GPU0; shard 50% -> 60%,70% on GPU1 (independent triggers).
# When both chains finish: merge overwrite + E2 tail via bbh1024_phase2_orchestrate_split.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_LOG="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_reasoning_patch_main.log"
SHARD_LOG="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_reasoning_patch_shard.log"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
LOG="$CALIB_OUT/bbh1024_chain_per_gpu.log"
CHAIN_DONE="$CALIB_OUT/bbh1024_chain_per_gpu_done.flag"
GPU0_STARTED="$CALIB_OUT/bbh1024_chain_gpu0_started.flag"
GPU1_STARTED="$CALIB_OUT/bbh1024_chain_gpu1_started.flag"
GPU0_DONE="$CALIB_OUT/bbh1024_chain_gpu0_done.flag"
GPU1_DONE="$CALIB_OUT/bbh1024_chain_gpu1_done.flag"
ORCH_STOPPED="$CALIB_OUT/bbh1024_orch_stopped.flag"
BACKUP_DONE="$CALIB_OUT/bbh1024_chain_backup_done.flag"
POLL_SEC="${POLL_SEC:-60}"

MAIN_CFG="$ROOT/configs/stage_a/e1_oneshot_curve.yaml"
SHARD_CFG="$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml"
PATCH="$ROOT/experiments/stage_a/run_e1_reasoning_patch.py"

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

log() { echo "[bbh1024-chain] $(date -Is) $*" | tee -a "$LOG"; }

main_boundary() {
  grep -q 'sparsity=0\.30 Reasoning=' "$MAIN_LOG" 2>/dev/null \
    && ! grep -q 'E1 Reasoning patch sparsity=0\.40' "$MAIN_LOG" 2>/dev/null
}

shard_boundary() {
  grep -q 'sparsity=0\.50 Reasoning=' "$SHARD_LOG" 2>/dev/null \
    && ! grep -q 'E1 Reasoning patch sparsity=0\.60' "$SHARD_LOG" 2>/dev/null
}

stop_orchestrator_once() {
  if [[ -f "$ORCH_STOPPED" ]]; then
    return 0
  fi
  local pids
  pids=$(pgrep -f "bbh1024_phase2_orchestrate.sh" || true)
  if [[ -n "${pids:-}" ]]; then
    log "SIGTERM legacy orchestrator: $pids"
    kill -TERM $pids || true
    sleep 5
  fi
  date -Is > "$ORCH_STOPPED"
}

backup_checkpoints_once() {
  if [[ -f "$BACKUP_DONE" ]]; then
    return 0
  fi
  local stamp backup_dir
  stamp="$(date +%Y%m%d_%H%M%S)"
  backup_dir="$CALIB_OUT/pre_chain_${stamp}"
  mkdir -p "$backup_dir"
  cp -a "$MAIN_OUT/e1_checkpoint.json" "$backup_dir/e1_checkpoint_main.json"
  cp -a "$SHARD_OUT/e1_checkpoint.json" "$backup_dir/e1_checkpoint_shard.json"
  log "checkpoint backup -> $backup_dir"
  date -Is > "$BACKUP_DONE"
}

kill_legacy_main_patch() {
  local pids
  pids=$(pgrep -f "run_e1_reasoning_patch.py.*--main-only" || true)
  if [[ -n "${pids:-}" ]]; then
    log "SIGTERM legacy main-only patch: $pids"
    kill -TERM $pids || true
    sleep 15
  fi
}

kill_legacy_shard_patch() {
  local pids
  pids=$(pgrep -f "run_e1_reasoning_patch.py.*--shard-only" || true)
  if [[ -n "${pids:-}" ]]; then
    log "SIGTERM legacy shard-only patch: $pids"
    kill -TERM $pids || true
    sleep 15
  fi
}

run_gpu0_chain() {
  if [[ -f "$GPU0_DONE" ]]; then
    log "GPU0 chain already done"
    return 0
  fi
  log "GPU0 waiting for main 30% boundary (poll ${POLL_SEC}s)"
  while ! main_boundary; do
    sleep "$POLL_SEC"
  done
  log "GPU0 boundary: main 30% written, 40% not started"
  if [[ -f "$GPU0_STARTED" ]]; then
    log "GPU0 chain already started; skip duplicate"
    return 0
  fi
  stop_orchestrator_once
  backup_checkpoints_once
  kill_legacy_main_patch
  date -Is > "$GPU0_STARTED"

  log "GPU0 chain: main 0.4 Reasoning 1024"
  CUDA_VISIBLE_DEVICES=0 python "$PATCH" \
    --config "$MAIN_CFG" \
    --shard-config "$SHARD_CFG" \
    --main-only \
    --skip-dense \
    --sparsities 0.4 \
    --device cuda:0 \
    >> "$MAIN_OUT/e1_reasoning_patch_chain_gpu0_0.4.log" 2>&1

  date -Is > "$GPU0_DONE"
  log "GPU0 chain done"
}

run_gpu1_chain() {
  if [[ -f "$GPU1_DONE" ]]; then
    log "GPU1 chain already done"
    return 0
  fi
  log "GPU1 waiting for shard 50% boundary (poll ${POLL_SEC}s)"
  while ! shard_boundary; do
    sleep "$POLL_SEC"
  done
  log "GPU1 boundary: shard 50% written, 60% not started"
  if [[ -f "$GPU1_STARTED" ]]; then
    log "GPU1 chain already started; skip duplicate"
    return 0
  fi
  stop_orchestrator_once
  backup_checkpoints_once
  kill_legacy_shard_patch
  date -Is > "$GPU1_STARTED"

  log "GPU1 chain: shard 0.6,0.7 Reasoning 1024"
  CUDA_VISIBLE_DEVICES=1 python "$PATCH" \
    --config "$MAIN_CFG" \
    --shard-config "$SHARD_CFG" \
    --shard-only \
    --skip-dense \
    --sparsities 0.6,0.7 \
    --device cuda:0 \
    >> "$SHARD_OUT/e1_reasoning_patch_chain_gpu1_0.6_0.7.log" 2>&1

  date -Is > "$GPU1_DONE"
  log "GPU1 chain done"
}

if [[ -f "$CHAIN_DONE" ]]; then
  log "chain already finished ($CHAIN_DONE); running E1 merge+report only"
  bash "$ROOT/scripts/bbh1024_e1_merge_and_report.sh" 2>&1 | tee -a "$LOG"
  exit 0
fi

mkdir -p "$CALIB_OUT" "$MAIN_OUT" "$SHARD_OUT"
log "start per-GPU chains (GPU0: 30->40, GPU1: 50->60/70)"

run_gpu0_chain &
PID0=$!
run_gpu1_chain &
PID1=$!

wait "$PID0" || { log "[ERROR] GPU0 chain failed"; exit 1; }
wait "$PID1" || { log "[ERROR] GPU1 chain failed"; exit 1; }

log "both GPU chains finished; post-chain verify"
python "$ROOT/scripts/bbh1024_verify_e1_1024_checkpoint.py" --phase post-chain \
  | tee -a "$LOG"

date -Is > "$CHAIN_DONE"
log "chain complete; E1 merge + report only (E2 deferred)"
bash "$ROOT/scripts/bbh1024_e1_merge_and_report.sh" 2>&1 | tee -a "$LOG"
