#!/usr/bin/env bash
# After balanced dual is launched: wait for merge done, then update E2 docs / Gate A flags.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E2_iterative_vs_oneshot"
LOG="$MAIN_OUT/e2_balanced_close.log"
DONE_FLAG="$MAIN_OUT/e2_phase_abc_chain_done.flag"
FAIL_FLAG="$MAIN_OUT/e2_phase_abc_chain_failed.flag"
mkdir -p "$MAIN_OUT"
log() { echo "[e2-close] $(date -Is) $*" | tee -a "$LOG"; }

trap 'log "[ERROR] close chain failed at line $LINENO"; date -Is > "$FAIL_FLAG"; echo "failed at line $LINENO" >> "$FAIL_FLAG"' ERR

source "$ROOT/venv/bin/activate"
source "$ROOT/scripts/env_llm.sh"
export PYTHONPATH="$ROOT"
export HF_ALLOW_CODE_EVAL=1

rm -f "$FAIL_FLAG"

log "waiting for e2_dual_done.flag"
while [[ ! -f "$MAIN_OUT/e2_dual_done.flag" ]]; do
  if ! pgrep -f "run_e2_iterative_vs_oneshot.py" >/dev/null \
     && ! pgrep -f "e2_dual_merge_when_done.sh" >/dev/null; then
    # runners gone and no merge watcher — check if shards done then merge ourselves
    ready=$(python - <<'PY'
import json
from pathlib import Path
def ok(p):
    path=Path(p)
    if not path.is_file():
        return False
    return json.loads(path.read_text()).get("status")=="done"
print("yes" if ok("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu0/e2_checkpoint.json")
      and ok("/mnt/data2/results/E2_iterative_vs_oneshot_shard_gpu1/e2_checkpoint.json")
      else "no")
PY
)
    if [[ "$ready" == "yes" && ! -f "$MAIN_OUT/e2_dual_done.flag" ]]; then
      log "shards done but no merge flag; running merge"
      bash "$ROOT/scripts/e2_dual_merge_when_done.sh" 2>&1 | tee -a "$LOG"
    fi
  fi
  log "heartbeat waiting merge/dual_done"
  sleep 300
done

log "dual done; updating docs"
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
echo "gate=$(python -c "import json; s=json.load(open('$MAIN_OUT/e2_summary.json')); print((s.get('outputs') or {}).get('gate_a') or s.get('gate_a'))")" >> "$DONE_FLAG"
log "[OK] balanced close complete -> $DONE_FLAG"
