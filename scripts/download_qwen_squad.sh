#!/usr/bin/env bash
# Download Qwen2.5-1.5B-Instruct and SQuAD 2.0 onto /mnt/data only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env_llm.sh
source "$ROOT/scripts/env_llm.sh"

PYTHON="${ROOT}/venv/bin/python"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_DIR="${LLM_MODEL_DIR}"
SQUAD_DIR="${LLM_SQUAD_DIR}"

echo "[K1] HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
echo "[K1] model -> $MODEL_DIR"
echo "[K1] squad -> $SQUAD_DIR"

"$PYTHON" - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
local_dir = Path("${MODEL_DIR}")
local_dir.mkdir(parents=True, exist_ok=True)
print(f"[K1] downloading {model_id} ...")
path = snapshot_download(
    repo_id=model_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
)
print(f"[K1] model ready: {path}")
PY

"$PYTHON" - <<PY
from pathlib import Path
from datasets import load_dataset

cache_dir = Path("${SQUAD_DIR}")
cache_dir.mkdir(parents=True, exist_ok=True)
print("[K1] downloading rajpurkar/squad_v2 ...")
ds = load_dataset("rajpurkar/squad_v2", cache_dir=str(cache_dir))
print(f"[K1] train={len(ds['train'])} official_validation={len(ds['validation'])}")
print(f"[K1] squad cache: {cache_dir}")
PY

echo "[K1] disk after download:"
df -h / /mnt/data
du -sh "$MODEL_DIR" "$SQUAD_DIR" "$HF_HOME" 2>/dev/null || true
