"""Run CIFAR P1.2 studies across multiple seeds and optional compression targets."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_cifar_p12_comparison import run_study, run_test_report
from src.experiments.compression_targets import apply_compression_target, ensure_compression_target
from src.experiments.p12_aggregate import aggregate_root
from src.utils.experiment_artifacts import load_config


def _seeds_from_config(config: Dict[str, Any]) -> List[int]:
    seeds = config.get("seeds")
    if seeds is None:
        return [int(config["seed"])]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list when provided")
    return [int(seed) for seed in seeds]


def _compression_targets_from_config(config: Dict[str, Any], sweep: bool) -> List[float | None]:
    targets = config.get("comparison", {}).get("compression_targets")
    if not sweep:
        return [None]
    if not isinstance(targets, list) or not targets:
        raise ValueError("comparison.compression_targets must be a non-empty list for sweep mode")
    return [float(value) for value in targets]


def run_multiseed(config_path: Path, checkpoint: Path, sweep: bool) -> Dict[str, Any]:
    base_config = load_config(config_path)
    seeds = _seeds_from_config(base_config)
    targets = _compression_targets_from_config(base_config, sweep)
    study_dirs: List[str] = []
    for target in targets:
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            if target is not None:
                config = apply_compression_target(config, target)
                label = f"ratio_{target:g}_seed_{seed}"
                root = Path(config["logging"]["output_root"]) / label
                config["logging"]["output_root"] = str(root)
                config["run_label"] = f"{config.get('run_label', 'cifar_p12')}_{label}"
            else:
                config = ensure_compression_target(config)
            study_dir = run_study(config, checkpoint, " ".join(sys.argv))
            run_test_report(study_dir, config, checkpoint)
            study_dirs.append(str(study_dir))
    aggregate = aggregate_root(Path(base_config["logging"]["output_root"]))
    manifest = {
        "protocol": "cifar_p12_multiseed_bundle",
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "compression_targets": targets if sweep else [],
        "study_dirs": study_dirs,
        "aggregate_json": aggregate["aggregate_json"],
    }
    manifest_path = Path(base_config["logging"]["output_root"]) / "multiseed_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR P1.2 studies across seeds and compression targets")
    parser.add_argument("--config", default="configs/cifar_p12_gpu_multiseed.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_resnet18_baseline.pth")
    parser.add_argument("--sweep", action="store_true", help="Enable compression target sweep")
    args = parser.parse_args()
    manifest = run_multiseed(Path(args.config), Path(args.checkpoint), sweep=args.sweep)
    print(f"[OK] Multi-seed bundle manifest: {manifest['aggregate_json']}")


if __name__ == "__main__":
    main()
