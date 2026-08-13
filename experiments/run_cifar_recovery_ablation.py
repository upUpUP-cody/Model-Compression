"""Compare recovery levels 1/2/3 on a fixed CIFAR prune candidate."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.cifar_p12_comparison import _keep_indices, _layer_ratios, parameter_count
from src.experiments.model_factory import build_model_from_config
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.recovery_dispatch import run_recovery
from src.utils.data_loader import get_cifar10_loaders
from src.utils.device import configure_cuda, resolve_device
from src.utils.experiment_artifacts import load_config, set_seed, to_json_safe


def run_ablation(config: Dict[str, Any], checkpoint: Path) -> Path:
    device = resolve_device(config["hardware"]["device"])
    set_seed(config["seed"])
    configure_cuda(config)
    model = build_model_from_config(config).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))

    split_seed = config["dataset"].get("split_seed", config["seed"])
    train_loader, validation_loader = get_cifar10_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        validation_fraction=config["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
    )
    comparison = config["comparison"]
    model_type = config["model"].get("type", "resnet_cifar")
    backend = resolve_pruning_backend(model, model_type)
    ratios = _layer_ratios(model, comparison.get("oneshot_layer_ratios", {}), model_type)
    keep_indices = _keep_indices(
        backend,
        model,
        train_loader,
        ratios,
        "wanda",
        str(device),
        comparison,
        model_type,
    )
    pruned = backend.create_pruned_model_by_indices(keep_indices).to(device)
    levels = [int(level) for level in comparison.get("recovery_levels", [1, 2, 3])]
    rows: List[Dict[str, Any]] = []
    for level in levels:
        cfg = copy.deepcopy(config)
        cfg["recovery"] = dict(cfg.get("recovery", {}))
        cfg["recovery"]["level"] = level
        cfg["recovery"]["epochs"] = int(comparison.get("recovery_epochs", cfg["recovery"].get("epochs", 1)))
        recovered, history = run_recovery(
            copy.deepcopy(pruned),
            train_loader,
            validation_loader,
            cfg,
            teacher_model=model,
            verbose=False,
        )
        rows.append({
            "recovery_level": level,
            "parameter_count": parameter_count(recovered),
            "history": history,
        })

    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(to_json_safe({"rows": rows}), indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_root / "ABLATION_REPORT.md"
    lines = ["# CIFAR Recovery Ablation", "", "| Level | Parameters | Best Val Acc |", "|---|---:|---:|"]
    for row in rows:
        best = row["history"].get("best_validation_accuracy")
        if best is None:
            vals = row["history"].get("validation_accuracy") or row["history"].get("best_validation_accuracy")
            best = vals[-1] if isinstance(vals, list) and vals else "n/a"
        lines.append(f"| {row['recovery_level']} | {row['parameter_count']} | {best} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR recovery ablation")
    parser.add_argument("--config", default="configs/cifar_recovery_ablation.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_resnet18_baseline.pth")
    args = parser.parse_args()
    output = run_ablation(load_config(args.config), Path(args.checkpoint))
    print(f"[OK] Ablation artifacts: {output}")


if __name__ == "__main__":
    main()
