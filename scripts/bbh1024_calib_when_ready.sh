#!/usr/bin/env bash
# Wait for E2 shard Reasoning dim complete (Instruction not started), run BBH1024 calib, resume shard.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CALIB_OUT="/mnt/data2/results/bbh_maxtok1024_calib"
SHARD0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
SHARD1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
SHARD0_CKPT="$SHARD0/e2_checkpoint.json"
SHARD1_CKPT="$SHARD1/e2_checkpoint.json"
CFG="$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml"
CALIB_CFG="$ROOT/configs/stage_a/bbh_maxtok1024_calib.yaml"
LOG="$CALIB_OUT/bbh1024_watcher.log"
POLL_SEC="${BBH1024_POLL_SEC:-300}"
FLAG="$CALIB_OUT/bbh1024_calib_done.flag"
CANCEL_FLAG="$CALIB_OUT/bbh1024_calib_cancelled.flag"

log() { echo "[bbh1024] $(date -Is) $*" | tee -a "$LOG"; }

mkdir -p "$CALIB_OUT"

if [[ -f "$CANCEL_FLAG" ]]; then
  log "cancelled; exit (see $CANCEL_FLAG)"
  exit 0
fi

if [[ -f "$FLAG" ]]; then
  log "calib already done (flag exists); exit"
  exit 0
fi

# Returns: gpu_index (0|1) or empty
ready_shard() {
  python - <<'PY'
import json
import sys
from pathlib import Path

def ready(path: Path) -> bool:
    if not path.is_file():
        return False
    d = json.loads(path.read_text(encoding="utf-8"))
    partial = d.get("partial") or {}
    done = set(partial.get("completed_dimensions") or [])
    return "Reasoning" in done and "Instruction" not in done

shards = [
    (0, Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json")),
    (1, Path("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json")),
]
for gpu, ckpt in shards:
    if ready(ckpt):
        print(gpu)
        sys.exit(0)
sys.exit(1)
PY
}

stop_e2_shard() {
  local shard_out="$1"
  local pids
  pids=$(pgrep -f "run_e2_iterative_vs_oneshot.py.*--output-root ${shard_out}" || true)
  if [[ -z "${pids:-}" ]]; then
    log "[WARNING] no E2 process for output-root=$shard_out"
    return 0
  fi
  log "SIGTERM E2 shard $shard_out PIDs: $pids"
  # shellcheck disable=SC2086
  kill -TERM $pids || true
  for _ in $(seq 1 60); do
    if ! pgrep -f "run_e2_iterative_vs_oneshot.py.*--output-root ${shard_out}" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  log "[WARNING] E2 shard still running; SIGKILL"
  pkill -9 -f "run_e2_iterative_vs_oneshot.py.*--output-root ${shard_out}" || true
  sleep 3
}

resume_e2_shard() {
  local gpu="$1"
  local shard_out="$2"
  local seeds="$3"
  source "$ROOT/venv/bin/activate"
  source "$ROOT/scripts/env_llm.sh"
  export PYTHONPATH="$ROOT"
  export HF_ALLOW_CODE_EVAL=1
  log "resume E2 GPU${gpu} seeds=${seeds} -> $shard_out"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
    --config "$CFG" \
    --seeds "$seeds" \
    --resume \
    --output-root "$shard_out" \
    >> "$MAIN_OUT/e2_gpu${gpu}.log" 2>&1 &
  log "E2 GPU${gpu} resumed PID=$!"
}

log "polling every ${POLL_SEC}s for E2 partial Reasoning done (Instruction not started)"
until gpu_idx=$(ready_shard 2>>"$LOG"); do
  sleep "$POLL_SEC"
done
log "trigger: GPU${gpu_idx} Reasoning complete in partial"

if [[ "$gpu_idx" == "0" ]]; then
  SHARD_OUT="$SHARD0"
  SHARD_CKPT="$SHARD0_CKPT"
  SEEDS="42,43"
else
  SHARD_OUT="$SHARD1"
  SHARD_CKPT="$SHARD1_CKPT"
  SEEDS="44"
fi

ts=$(date +%Y%m%d_%H%M%S)
if [[ -f "$SHARD_CKPT" ]]; then
  cp -a "$SHARD_CKPT" "$SHARD_OUT/e2_checkpoint.pre_bbh1024_calib_${ts}.json"
  log "checkpoint backup: e2_checkpoint.pre_bbh1024_calib_${ts}.json"
fi

stop_e2_shard "$SHARD_OUT"
sleep 5

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

log "starting BBH1024 calib on GPU${gpu_idx} (batch=2 limit=64 + batch4 probe limit=32)"
set +e
CUDA_VISIBLE_DEVICES="${gpu_idx}" python "$ROOT/experiments/stage_a/run_bbh_maxtok_calib.py" \
  --config "$CALIB_CFG" \
  --device cuda:0 \
  --batch4-limit 32 \
  2>&1 | tee -a "$LOG"
calib_rc=${PIPESTATUS[0]}
set -e
log "calib exit code=$calib_rc"

resume_e2_shard "$gpu_idx" "$SHARD_OUT" "$SEEDS"
date -Is > "$FLAG"
log "watcher complete flag=$FLAG calib_rc=$calib_rc"

if [[ "$calib_rc" -ne 0 ]]; then
  log "[WARNING] calib score gate failed; see $CALIB_OUT/bbh1024_calib.json"
  exit "$calib_rc"
fi
log "[OK] BBH1024 calib passed; report at $CALIB_OUT/bbh1024_calib.json"
