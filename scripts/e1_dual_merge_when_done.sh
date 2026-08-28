#!/usr/bin/env bash
# Wait for E1 oneshot curve to finish, then:
#   1) patch shard Code (dense + 10/50/60/70%) with fixed HumanEval protocol
#   2) merge shard into main
#   3) verify summary (all sparsities + Code scores)
# Does NOT kill any running curve process. Safe under single-GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"
LOG="$MAIN_OUT/e1_merge_watcher.log"
CODE_PATCH_DONE="$MAIN_OUT/e1_code_patch_shard_incl70.flag"

log() { echo "[merge-watcher] $(date -Is) $*" | tee -a "$LOG"; }

log "waiting for E1 oneshot curve to finish (single-GPU OK)"
while pgrep -f "run_e1_oneshot_curve.py" >/dev/null; do
  sleep 120
done

# Also require sparsity 0.7 row in shard curve (Code may still be null/skipped).
log "waiting for sparsity 0.70 in shard completed_sparsities"
while true; do
  done70=$(
    python - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json")
if not p.is_file():
    print("no")
else:
    d = json.loads(p.read_text())
    completed = [float(x) for x in d.get("completed_sparsities") or []]
    print("yes" if any(abs(x - 0.7) < 1e-6 for x in completed) else "no")
PY
  )
  if [[ "$done70" == "yes" ]]; then
    log "0.70 completed in shard; proceeding to Code patch"
    break
  fi
  sleep 120
done

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

# Code patch BEFORE merge so merged summary includes new HumanEval scores.
if [[ ! -f "$CODE_PATCH_DONE" ]]; then
  log "patching shard Code (dense + 0.1/0.5/0.6/0.7) on cuda:0"
  CUDA_VISIBLE_DEVICES=0 python "$ROOT/experiments/stage_a/run_e1_code_patch.py" \
    --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
    --shard-config "$ROOT/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml" \
    --shard-only \
    --sparsities 0.1,0.5,0.6,0.7 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG"
  date -Is > "$CODE_PATCH_DONE"
  log "shard Code patch done (incl 70%)"
else
  log "shard Code patch already done ($CODE_PATCH_DONE)"
fi

log "merging shard into main"
python "$ROOT/experiments/stage_a/run_e1_oneshot_curve.py" \
  --config "$ROOT/configs/stage_a/e1_oneshot_curve.yaml" \
  --merge-from "$SHARD_OUT" \
  --resume 2>&1 | tee -a "$LOG"

log "verifying e1_summary.json (curve + Code)"
python - <<'PY'
import json
import math
from pathlib import Path

p = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_summary.json")
s = json.loads(p.read_text())
curve = s.get("records", {}).get("curve", [])
sparsities = sorted(float(r["sparsity"]) for r in curve)
expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
if sparsities != expected:
    raise SystemExit(f"curve incomplete: got {sparsities}, want {expected}")
dense = s.get("records", {}).get("dense_capability", {}).get("vector", {})
dims = ("PPL", "Math", "Knowledge", "Reasoning", "Instruction", "Code")
if not all(dense.get(d) is not None for d in dims):
    raise SystemExit(f"dense_capability incomplete: {dense}")

def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))

if not _finite(dense.get("Code")):
    raise SystemExit(f"dense Code not finite: {dense.get('Code')}")
for r in curve:
    code = (r.get("vector") or {}).get("Code")
    if not _finite(code):
        raise SystemExit(f"sparsity={r.get('sparsity')} Code not finite: {code}")
print("[OK] E1 summary verified: 7 sparsities + dense; all Code finite")
PY

date -Is > "$MAIN_OUT/e1_dual_done.flag"
python "$ROOT/scripts/e1_mark_docs_done.py" 2>&1 | tee -a "$LOG" || true

log "merge watcher finished"
