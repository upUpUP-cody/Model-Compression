#!/usr/bin/env bash
# Wait for GPU0 20% + GPU1 50% checkpoints, then switch eval batch_size 2->4 and resume.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
MAIN_CKPT="$MAIN_OUT/e1_checkpoint.json"
SHARD_CKPT="$SHARD_OUT/e1_checkpoint.json"
MAIN_CFG="$ROOT/configs/stage_a/e1_oneshot_curve.yaml"
SHARD_CFG="$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml"
LOG="$MAIN_OUT/e1_batch4_switch.log"
POLL_SEC="${E1_BATCH4_POLL_SEC:-300}"
FLAG="$MAIN_OUT/e1_batch4_applied.flag"
CANCEL_FLAG="$MAIN_OUT/e1_batch4_cancelled.flag"

log() { echo "[batch4] $(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$CANCEL_FLAG" ]]; then
  log "batch4 cancelled; exit (see $CANCEL_FLAG)"
  exit 0
fi

ready_check() {
  python - <<'PY'
import json
import sys
from pathlib import Path

def has(path: Path, sp: float) -> bool:
    if not path.is_file():
        return False
    d = json.loads(path.read_text(encoding="utf-8"))
    completed = [float(x) for x in d.get("completed_sparsities", [])]
    return any(abs(x - sp) < 1e-6 for x in completed)

main_ok = has(Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json"), 0.2)
shard_ok = has(Path("/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json"), 0.5)
print(f"gpu0_20={main_ok} gpu1_50={shard_ok}")
sys.exit(0 if (main_ok and shard_ok) else 1)
PY
}

patch_yaml_batch() {
  local cfg="$1"
  python - <<PY
from pathlib import Path
import re
p = Path("$cfg")
text = p.read_text(encoding="utf-8")
# Only patch evaluation.capability.batch_size and hardware.batch_size (not calibration)
lines = []
in_capability = False
in_hardware = False
for line in text.splitlines():
    if line.startswith("evaluation:"):
        in_capability = False
        in_hardware = False
    elif line.startswith("hardware:"):
        in_hardware = True
        in_capability = False
    elif line.startswith("pruning:") or line.startswith("logging:") or line.startswith("dataset:"):
        in_hardware = False
        in_capability = False
    elif "capability:" in line and line.strip() == "capability:":
        in_capability = True
    if in_capability and re.match(r"\s+batch_size:\s*2\s*$", line):
        line = re.sub(r"batch_size:\s*2", "batch_size: 4", line)
    if in_hardware and re.match(r"\s+batch_size:\s*2\s*$", line):
        line = re.sub(r"batch_size:\s*2", "batch_size: 4", line)
    lines.append(line)
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] patched batch_size -> 4 in {p}")
PY
}

if [[ -f "$FLAG" ]]; then
  log "batch4 already applied (flag exists); exit"
  exit 0
fi

log "polling every ${POLL_SEC}s for main 0.2 + shard 0.5 checkpoints"
until ready_check >>"$LOG" 2>&1; do
  sleep "$POLL_SEC"
done
log "trigger conditions met"

if pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
  PIDS=$(pgrep -f "run_e1_oneshot_curve.py" | tr '\n' ' ')
  log "SIGTERM E1 processes: $PIDS"
  kill -TERM $PIDS || true
  for _ in $(seq 1 60); do
    if ! pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
      break
    fi
    sleep 5
  done
  if pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
    log "[WARNING] E1 still running after SIGTERM; sending SIGKILL"
    pkill -9 -f "run_e1_oneshot_curve.py" || true
    sleep 3
  fi
fi

ts=$(date +%Y%m%d_%H%M%S)
if [[ -f "$MAIN_CKPT" ]]; then
  cp -a "$MAIN_CKPT" "$MAIN_OUT/e1_checkpoint.pre_batch4_${ts}.json"
fi
if [[ -f "$SHARD_CKPT" ]]; then
  cp -a "$SHARD_CKPT" "$SHARD_OUT/e1_checkpoint.pre_batch4_${ts}.json"
fi
log "checkpoint backups written (pre_batch4_${ts})"

patch_yaml_batch "$MAIN_CFG"
patch_yaml_batch "$SHARD_CFG"
log "yaml batch_size updated to 4 (calibration batch_size unchanged)"

"$ROOT/scripts/resume_e1_dual_33.sh" 2>&1 | tee -a "$LOG"
date -Is > "$FLAG"
log "batch4 switch complete; flag=$FLAG"
"$ROOT/scripts/e1_verify_batch4_switch.sh" 2>&1 | tee -a "$LOG" || log "[WARNING] post-switch verify failed (check OOM / yaml)"
