"""Compatibility shim: loads archive/experiments_legacy/run_p12_multiseed.py as experiments.run_p12_multiseed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_NAME = "experiments.run_p12_multiseed"
_LEGACY = Path(__file__).resolve().parents[1] / "archive" / "experiments_legacy" / "run_p12_multiseed.py"
_spec = importlib.util.spec_from_file_location(_NAME, _LEGACY)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register before exec so intra-module and monkeypatch see the same object.
sys.modules[_NAME] = _mod
_spec.loader.exec_module(_mod)
# Legacy scripts set PROJECT_ROOT to archive/; point back to repo root.
if hasattr(_mod, "PROJECT_ROOT"):
    _mod.PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_mod.PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_mod.PROJECT_ROOT))
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
