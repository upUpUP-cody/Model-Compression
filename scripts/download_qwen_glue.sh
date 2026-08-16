#!/usr/bin/env bash
# Download GLUE SST-2 / RTE / QNLI onto /mnt/data/datasets/glue (Phase K KG).
# Default: all three tasks. Override with GLUE_TASK=sst2|rte|qnli for a single task.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env_llm.sh
source "$ROOT/scripts/env_llm.sh"

PYTHON="${ROOT}/venv/bin/python"
GLUE_DIR="${LLM_GLUE_DIR}"
if [[ -n "${GLUE_TASK:-}" ]]; then
  TASKS=("${GLUE_TASK}")
else
  TASKS=(sst2 rte qnli)
fi

echo "[KG.0] HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
echo "[KG.0] glue -> $GLUE_DIR (tasks=${TASKS[*]})"

"$PYTHON" - <<PY
from pathlib import Path
from datasets import load_dataset

cache_dir = Path("${GLUE_DIR}")
cache_dir.mkdir(parents=True, exist_ok=True)
tasks = """${TASKS[*]}""".split()
for task in tasks:
    print(f"[KG.0] downloading nyu-mll/glue/{task} ...")
    ds = load_dataset("nyu-mll/glue", task, cache_dir=str(cache_dir))
    for split_name, split in ds.items():
        print(f"[KG.0] {task}/{split_name}={len(split)}")
print(f"[KG.0] glue cache: {cache_dir}")
PY

echo "[KG.0] disk after download:"
df -h / /mnt/data /mnt/data2 2>/dev/null || df -h /
du -sh "$GLUE_DIR" "$HF_HOME" 2>/dev/null || true
