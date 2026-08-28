#!/usr/bin/env bash
# BBH1024 Phase2 orchestrator:
#   GPU1: E0 Reasoning gate (Instruct)
#   GPU0: E1 main Reasoning patch (--force while E0 runs; J0 base path still enforced)
#   After E0: GPU1 runs E1 shard Reasoning patch
#   After both E1: dual E2 Reasoning patch -> audit -> launch_e2_dual resume
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
E0_OUT="/mnt/data2/results/E0_dense_baseline"
E2_MAIN="/mnt/data2/results/E2_iterative_vs_oneshot"
E2_S0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
E2_S1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
LOG="$CALIB_OUT/bbh1024_phase2_orch.log"
mkdir -p "$CALIB_OUT" "$E2_MAIN"

log() { echo "[bbh1024-orch] $(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$CALIB_OUT/bbh1024_protocol_locked.flag" ]]; then
  log "already locked; launching E2 resume only"
  bash "$ROOT/scripts/launch_e2_dual.sh"
  exit 0
fi

log "start E0 Reasoning gate on GPU1"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e0_reasoning_gate.py" \
  --config "$ROOT/configs/stage_a/e0_dense.yaml" \
  --device cuda:0 \
  > "$E0_OUT/e0_reasoning_gate.log" 2>&1 &
E0_PID=$!
log "E0_PID=$E0_PID log=$E0_OUT/e0_reasoning_gate.log"

log "start E1 main Reasoning patch on GPU0 (--force: E0 parallel; Instruct≠base)"
CUDA_VISIBLE_DEVICES=0 nohup python "$ROOT/experiments/stage_a/run_e1_reasoning_patch.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --shard-config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
  --main-only --force --device cuda:0 \
  > "$MAIN_OUT/e1_reasoning_patch_main.log" 2>&1 &
E1M_PID=$!
log "E1_MAIN_PID=$E1M_PID log=$MAIN_OUT/e1_reasoning_patch_main.log"

log "wait E0 gate..."
while kill -0 "$E0_PID" 2>/dev/null; do sleep 30; done
wait "$E0_PID" || { log "[ERROR] E0 gate failed"; exit 1; }
[[ -f "$E0_OUT/e0_reasoning_gate_passed.flag" ]] || { log "[ERROR] missing E0 passed flag"; exit 1; }
log "E0 gate passed"

log "start E1 shard Reasoning patch on GPU1"
CUDA_VISIBLE_DEVICES=1 nohup python "$ROOT/experiments/stage_a/run_e1_reasoning_patch.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --shard-config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
  --shard-only --device cuda:0 \
  > "$SHARD_OUT/e1_reasoning_patch_shard.log" 2>&1 &
E1S_PID=$!
log "E1_SHARD_PID=$E1S_PID"

log "wait E1 main..."
while kill -0 "$E1M_PID" 2>/dev/null; do sleep 60; done
wait "$E1M_PID" || { log "[ERROR] E1 main patch failed"; exit 1; }
[[ -f "$MAIN_OUT/e1_reasoning_j2_passed.flag" ]] || { log "[ERROR] missing J2 flag"; exit 1; }
log "E1 main + J2 OK"

log "wait E1 shard..."
while kill -0 "$E1S_PID" 2>/dev/null; do sleep 60; done
wait "$E1S_PID" || { log "[ERROR] E1 shard patch failed"; exit 1; }
log "E1 shard OK"

log "regenerate E1 merged report/figures (Reasoning 1024)"
python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --merge-from "$SHARD_OUT" \
  --resume 2>&1 | tee -a "$LOG"
log "E1 report regen OK"

log "E2 Reasoning patch dual"
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
log "[OK] Phase2 orchestration finished; E2 resumed under 1024+batch4"
