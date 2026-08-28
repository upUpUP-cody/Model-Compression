#!/usr/bin/env bash
# Wait for E2 dual shards to finish, then merge into main and verify Gate A fields.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
SHARD0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
SHARD1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
LOG="$MAIN_OUT/e2_merge_watcher.log"
CFG="$ROOT/configs/stage_a/e2_iterative_vs_oneshot.yaml"

mkdir -p "$MAIN_OUT"
log() { echo "[e2-merge-watcher] $(date -Is) $*" | tee -a "$LOG"; }

log "waiting for run_e2_iterative_vs_oneshot.py to finish"
while pgrep -f "run_e2_iterative_vs_oneshot.py" >/dev/null; do
  sleep 120
done

log "waiting for both shard checkpoints status=done"
while true; do
  ready=$(
    python - <<'PY'
import json
from pathlib import Path

def ok(p):
    path = Path(p)
    if not path.is_file():
        return False
    d = json.loads(path.read_text())
    return d.get("status") == "done"

print("yes" if ok("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json")
      and ok("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json")
      else "no")
PY
  )
  if [[ "$ready" == "yes" ]]; then
    log "both shards done; merging"
    break
  fi
  sleep 60
done

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

python "$ROOT/experiments/stage_a/run_e2_iterative_vs_oneshot.py" \
  --config "$CFG" \
  --output-root "$MAIN_OUT" \
  --merge-from "$SHARD0" \
  --merge-from "$SHARD1" \
  2>&1 | tee -a "$LOG"

log "verifying e2_summary.json"
python - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/data2/results/E2_iterative_vs_oneshot/e2_summary.json")
s = json.loads(p.read_text())
records = s.get("records", {}).get("comparison") or []
assert len(records) >= 15, f"expected >=15 records, got {len(records)}"
gate = (s.get("outputs") or {}).get("gate_a") or {}
print(f"[OK] records={len(records)} gate={gate}")
PY

date -Is > "$MAIN_OUT/e2_dual_done.flag"
log "merge watcher finished"
