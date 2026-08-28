"""E1 import must reject Reasoning without max_gen_toks=1024."""
from pathlib import Path

import pytest

from src.experiments.stage_a_common import import_e1_oneshot_rows, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_import_e1_rejects_missing_reasoning_1024():
    e2 = load_yaml(ROOT / "configs/stage_a/e2_iterative_vs_oneshot.yaml")
    e1_path = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.pre_bbh1024_20260827_093037.json")
    if not e1_path.is_file():
        pytest.skip("pre_bbh1024 archive missing")
    with pytest.raises(ValueError, match="max_gen_toks=1024"):
        import_e1_oneshot_rows(e1_path, [0.4], e2_config=e2)
