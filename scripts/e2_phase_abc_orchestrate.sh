#!/usr/bin/env bash
# Overnight chain: wait Phase A -> rebalance (B) -> balanced dual (C) -> merge -> close docs.
# Idempotent-ish: safe to leave running while Phase A jobs are alive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
SHARD0="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0"
SHARD1="/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1"
LOG="$MAIN_OUT/e2_phase_abc_orchestrator.log"
DONE_FLAG="$MAIN_OUT/e2_phase_abc_chain_done.flag"
FAIL_FLAG="$MAIN_OUT/e2_phase_abc_chain_failed.flag"
mkdir -p "$MAIN_OUT"
log() { echo "[e2-orch] $(date -Is) $*" | tee -a "$LOG"; }

trap 'log "[ERROR] orchestrator exited unexpectedly at line $LINENO"; date -Is > "$FAIL_FLAG"; echo "failed at line $LINENO" >> "$FAIL_FLAG"' ERR

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

if [[ -f "$DONE_FLAG" ]]; then
  log "already done ($DONE_FLAG); exit"
  exit 0
fi

rm -f "$FAIL_FLAG"

# ---- Phase A wait ----
log "waiting for Phase A run_e2_iterative_vs_oneshot.py to exit (or none running)"
while pgrep -f "run_e2_iterative_vs_oneshot.py" >/dev/null; do
  n=$(pgrep -fc "run_e2_iterative_vs_oneshot.py" || true)
  mem=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
  log "heartbeat Phase A still running procs=${n} gpu=${mem}"
  sleep 300
done
log "no E2 runner processes; verifying partials cleared"

python - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

for label, p in (
    ("gpu0", "/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json"),
    ("gpu1", "/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json"),
):
    d = json.loads(Path(p).read_text())
    partial = d.get("partial")
    n = len(d.get("completed_cells") or [])
    print(f"[check] {label} completed={n} partial={partial is not None and bool(partial)}")
    if isinstance(partial, dict) and partial.get("seed") is not None:
        raise SystemExit(f"[ERROR] {label} still has partial; refuse rebalance")
print("[OK] partials cleared")
PY

# ---- Phase B ----
log "Phase B: rebalance_e2_shards_for_resume.py"
python "$ROOT/scripts/rebalance_e2_shards_for_resume.py" 2>&1 | tee -a "$LOG"
date -Is > "$MAIN_OUT/e2_phase_b_rebalance_done.flag"

# ---- Phase C ----
log "Phase C: launch_e2_dual_balanced.sh"
bash "$ROOT/scripts/launch_e2_dual_balanced.sh" 2>&1 | tee -a "$LOG"
sleep 8
if ! pgrep -f "run_e2_iterative_vs_oneshot.py" >/dev/null; then
  log "[ERROR] Phase C runners did not stay up"
  exit 1
fi
pgrep -af "run_e2_iterative_vs_oneshot.py" | tee -a "$LOG" || true
date -Is > "$MAIN_OUT/e2_phase_c_launched.flag"

# ---- Merge (foreground wait) ----
log "Phase merge: e2_dual_merge_when_done.sh (blocking)"
bash "$ROOT/scripts/e2_dual_merge_when_done.sh" 2>&1 | tee -a "$LOG"

if [[ ! -f "$MAIN_OUT/e2_dual_done.flag" ]]; then
  log "[ERROR] merge finished but e2_dual_done.flag missing"
  exit 1
fi

# ---- Close docs ----
log "updating checklist / E_MAP after Gate A"
python - <<'PY' | tee -a "$LOG"
import json
import re
from pathlib import Path

main = Path("/mnt/data2/results/E2_iterative_vs_oneshot")
summary = json.loads((main / "e2_summary.json").read_text())
gate = (summary.get("outputs") or {}).get("gate_a") or summary.get("gate_a") or {}
passed = bool(gate.get("passed"))
wins = gate.get("iterative_cell_wins")
total = gate.get("total_cells")
status = "done (Gate A PASS)" if passed else "done (Gate A FAIL — 暂缓 Agent)"
line = f"| E2 | Iterative vs one-shot | 3B base | **{status}** | `/mnt/data2/results/E2_iterative_vs_oneshot/` |"

emap = Path("/root/Model-Compression/docs/project/EXPERIMENT_E_MAP.md")
text = emap.read_text()
text2, n = re.subn(
    r"\| E2 \| Iterative vs one-shot \| 3B base \|.*?\| `/mnt/data2/results/E2_iterative_vs_oneshot/` \|",
    line,
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"E_MAP E2 row replace failed n={n}")
# current step
text2 = re.sub(
    r"## 当前步\n\n.*",
    "## 当前步\n\n"
    f"**E2 已结束**：Gate A passed={passed} wins={wins}/{total}。"
    "下一步按 PDF 进入 E3（仅当 Gate A 通过）。**E0 Instruct 永不进入 Gate A**。\n",
    text2,
    count=1,
    flags=re.S,
)
emap.write_text(text2)

checklist = Path("/root/Model-Compression/docs/process/PDF_SETUP_CHECKLIST.md")
ct = checklist.read_text()
ct2, n2 = re.subn(
    r"\| E2 \|.*?\| `/mnt/data2/results/E2_iterative_vs_oneshot/` \|",
    f"| E2 | **{status}** | `/mnt/data2/results/E2_iterative_vs_oneshot/` |",
    ct,
    count=1,
)
if n2 != 1:
    raise SystemExit(f"checklist E2 row replace failed n={n2}")
checklist.write_text(ct2)
print(f"[OK] docs updated gate_passed={passed} wins={wins}/{total}")
PY

date -Is > "$DONE_FLAG"
echo "gate_summary=$(python -c "import json; s=json.load(open('$MAIN_OUT/e2_summary.json')); print((s.get('outputs') or {}).get('gate_a') or s.get('gate_a'))")" >> "$DONE_FLAG"
rm -f "$FAIL_FLAG"
log "[OK] full chain complete -> $DONE_FLAG"
