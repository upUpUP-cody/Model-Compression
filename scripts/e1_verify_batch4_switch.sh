#!/usr/bin/env bash
# Verify E1 batch4 switch: yaml batch=4, checkpoints intact, resume skips done sparsities.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve"
SHARD_OUT="/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1"

source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT"

python - <<'PY'
import json
import sys
from pathlib import Path
import yaml

def batch_in_cfg(path: Path, section: str) -> int:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if section == "capability":
        return int(cfg["evaluation"]["capability"]["batch_size"])
    return int(cfg["hardware"]["batch_size"])

main_cfg = Path("/root/Model-Compression/configs/stage_a/e1_oneshot_curve.yaml")
shard_cfg = Path("/root/Model-Compression/configs/stage_a/e1_oneshot_curve_shard_gpu1.yaml")
for p in (main_cfg, shard_cfg):
    assert batch_in_cfg(p, "capability") == 4, f"{p}: capability batch != 4"
    assert batch_in_cfg(p, "hardware") == 4, f"{p}: hardware batch != 4"
    cal = yaml.safe_load(p.read_text())["pruning"]["calibration"]["batch_size"]
    assert cal == 1, f"{p}: calibration batch changed ({cal})"

main_ckpt = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json")
shard_ckpt = Path("/mnt/data2/results/E1_oneshot_sparsity_curve_shard_gpu1/e1_checkpoint.json")
main = json.loads(main_ckpt.read_text())
completed_main = [float(x) for x in main.get("completed_sparsities", [])]
assert any(abs(x - 0.1) < 1e-6 for x in completed_main), "main missing 0.1"
assert any(abs(x - 0.2) < 1e-6 for x in completed_main), "main missing 0.2 after switch"

if shard_ckpt.is_file():
    shard = json.loads(shard_ckpt.read_text())
    completed_shard = [float(x) for x in shard.get("completed_sparsities", [])]
    assert any(abs(x - 0.5) < 1e-6 for x in completed_shard), "shard missing 0.5 after switch"

flag = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_batch4_applied.flag")
assert flag.is_file(), "batch4 applied flag missing"

print("[OK] batch4 verify: yaml=4, checkpoints preserved, flag present")
PY

# Optional: confirm new log lines show resumed jobs
if pgrep -f "run_e1_oneshot_curve.py" >/dev/null; then
  echo "[OK] E1 processes running after batch4 switch"
else
  echo "[WARNING] no E1 processes running; check e1_batch4_switch.log for OOM"
  exit 1
fi
