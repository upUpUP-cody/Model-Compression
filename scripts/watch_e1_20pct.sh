#!/usr/bin/env bash
# Poll E1 checkpoint until 20% sparsity is completed; write flag file for launch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json"
FLAG="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_20pct_done.flag"
LOG="/mnt/data2/results/E1_oneshot_sparsity_curve/e1_watch_20pct.log"
INTERVAL="${E1_WATCH_INTERVAL_SEC:-600}"

source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT"

echo "[watch] polling $CKPT every ${INTERVAL}s" | tee -a "$LOG"
while true; do
  if [[ ! -f "$CKPT" ]]; then
    echo "[watch] $(date -Is) checkpoint missing" | tee -a "$LOG"
    sleep "$INTERVAL"
    continue
  fi
  done=$(
    python - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json")
d = json.loads(p.read_text())
completed = [float(x) for x in d.get("completed_sparsities", [])]
print("yes" if any(abs(x - 0.2) < 1e-6 for x in completed) else "no")
PY
  )
  echo "[watch] $(date -Is) completed_sparsities check 0.2=$done" | tee -a "$LOG"
  if [[ "$done" == "yes" ]]; then
    date -Is > "$FLAG"
    echo "[watch] 20% DONE — flag written to $FLAG" | tee -a "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
