#!/usr/bin/env bash
# Phase B GPU smoke: real auto near-lossless on Qwen2.5-3B + torchao INT8.
# Expected wall clock: ~20-45 minutes on 1x RTX 4090.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source venv/bin/activate
# shellcheck disable=SC1091
source scripts/env_llm.sh

export PYTHONPATH="harness/src:src${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] Starting Phase B auto (real torchao + hellaswag@64)"
python -m compression_harness.cli auto \
  --goal harness/recipes/goal_near_lossless_small.yaml

ART="$(ls -d harness/experiments/run_001_* 2>/dev/null | head -1 || true)"
if [[ -z "${ART}" ]]; then
  echo "[FAIL] no run_001_* artifact dir"
  exit 1
fi
python - <<PY
import json, sys
from pathlib import Path
art_dirs = sorted(Path("harness/experiments").glob("run_001_*"))
if not art_dirs:
    print("[FAIL] missing run dir")
    sys.exit(1)
path = art_dirs[0] / "artifact.json"
data = json.loads(path.read_text(encoding="utf-8"))
print("[INFO]", path, "status=", data.get("status"), "drop=", data.get("relative_drop"))
need = ("near_lossless_ok", "relative_drop", "baseline_score", "score", "metrics")
missing = [k for k in need if k not in data]
if missing:
    print("[FAIL] missing fields", missing)
    sys.exit(1)
comp = Path(data.get("compressed_dir") or "")
if not comp.exists():
    print("[FAIL] compressed dir missing", comp)
    sys.exit(1)
weights = comp / "quantized_state_dict.pt"
legacy = comp / "quantized_model.pt"
if not weights.exists() and not legacy.exists():
    print("[FAIL] compressed weights missing", comp)
    sys.exit(1)
print("[OK] Phase B smoke checks passed")
PY
