"""Compare recovery levels 1/2/3 on a fixed CIFAR prune candidate."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomous_search import full_evaluate
from src.experiments.cifar_p12_comparison import _keep_indices, _layer_ratios, parameter_count
from src.experiments.compression_targets import ensure_compression_target
from src.experiments.model_factory import build_model_from_config
from src.experiments.p12_comparison import evaluate_validation
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.recovery_dispatch import run_recovery
from src.utils.data_loader import get_cifar10_loaders
from src.utils.device import configure_cuda, resolve_device
from src.utils.experiment_artifacts import load_config, set_seed, to_json_safe


def _mean_std(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": len(values)}


def run_ablation(config: Dict[str, Any], checkpoint: Path) -> Path:
    config = ensure_compression_target(config)
    device = resolve_device(config["hardware"]["device"])
    device_str = str(device)
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
        pin_memory=bool(config["dataset"].get("pin_memory", False)),
        persistent_workers=bool(config["dataset"].get("persistent_workers", False)),
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
        device_str,
        comparison,
        model_type,
    )
    pruned = backend.create_pruned_model_by_indices(keep_indices).to(device)

    baseline_params = parameter_count(model)
    pruned_params = parameter_count(pruned)
    actual_compression = baseline_params / max(pruned_params, 1)
    baseline_val = float(evaluate_validation(model, validation_loader, device_str)["accuracy"])
    pruned_val = float(evaluate_validation(pruned, validation_loader, device_str)["accuracy"])

    print(
        f"[INFO] baseline_val={baseline_val:.2f}% pruned_val={pruned_val:.2f}% "
        f"compression={actual_compression:.2f}x ({baseline_params}->{pruned_params})"
    )

    report_test = bool(comparison.get("report_test", False))
    test_loader = None
    if report_test:
        _, _, test_loader = get_cifar10_loaders(
            data_dir=config["dataset"]["data_dir"],
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            validation_fraction=config["dataset"].get("validation_fraction", 0.1),
            split_seed=split_seed,
            pin_memory=bool(config["dataset"].get("pin_memory", False)),
            persistent_workers=bool(config["dataset"].get("persistent_workers", False)),
            return_test=True,
        )

    levels = [int(level) for level in comparison.get("recovery_levels", [1, 2, 3])]
    rows: List[Dict[str, Any]] = []
    for level in levels:
        cfg = copy.deepcopy(config)
        cfg["recovery"] = dict(cfg.get("recovery", {}))
        cfg["recovery"]["level"] = level
        cfg["recovery"]["epochs"] = int(comparison.get("recovery_epochs", cfg["recovery"].get("epochs", 1)))
        cfg["recovery"]["learning_rate"] = float(
            comparison.get("recovery_learning_rate", cfg["recovery"].get("learning_rate", 0.001))
        )
        start = time.perf_counter()
        recovered, history = run_recovery(
            copy.deepcopy(pruned),
            train_loader,
            validation_loader,
            cfg,
            teacher_model=model,
            verbose=False,
        )
        seconds = time.perf_counter() - start
        best = history.get("best_validation_accuracy")
        if best is None:
            vals = history.get("validation_accuracy")
            best = vals[-1] if isinstance(vals, list) and vals else None
        recovered_params = parameter_count(recovered)
        row = {
            "recovery_level": level,
            "pruned_parameter_count": pruned_params,
            "recovered_parameter_count": recovered_params,
            "actual_compression_ratio": actual_compression,
            "pre_recovery_validation_accuracy": pruned_val,
            "best_validation_accuracy": best,
            "recovery_seconds": seconds,
            "history": history,
        }
        if test_loader is not None:
            test_metrics = full_evaluate(recovered, test_loader, device_str)
            row["test_accuracy"] = float(test_metrics["accuracy"])
            print(f"[INFO] level={level} test={row['test_accuracy']:.2f}%")
        rows.append(row)
        print(
            f"[INFO] level={level} best_val={best} "
            f"recovered_params={recovered_params} seconds={seconds:.1f}"
        )

    summary = {
        "seed": int(config["seed"]),
        "baseline_parameter_count": baseline_params,
        "pruned_parameter_count": pruned_params,
        "actual_compression_ratio": actual_compression,
        "target_compression_ratio": float(comparison.get("target_compression_ratio", 1.0)),
        "baseline_validation_accuracy": baseline_val,
        "pre_recovery_validation_accuracy": pruned_val,
        "oneshot_layer_ratios": dict(comparison.get("oneshot_layer_ratios", {})),
        "layer_keep_indices": {name: list(indices) for name, indices in keep_indices.items()},
        "rows": rows,
    }

    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(to_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")

    report_path = output_root / "ABLATION_REPORT.md"
    lines = [
        "# CIFAR Recovery Ablation",
        "",
        f"- seed: {int(config['seed'])}",
        f"- baseline validation: {baseline_val:.2f}%",
        f"- pre-recovery validation: {pruned_val:.2f}%",
        f"- actual compression: {actual_compression:.2f}x "
        f"(target {float(comparison.get('target_compression_ratio', 1.0)):.1f}x)",
        f"- baseline params: {baseline_params}",
        f"- pruned params: {pruned_params}",
        "",
        "| Level | pruned_params | recovered_params | compression | pre_recovery_val | best_val | test | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best_validation_accuracy"]
        best_str = f"{best:.2f}" if isinstance(best, (int, float)) else "n/a"
        test = row.get("test_accuracy")
        test_str = f"{test:.2f}" if isinstance(test, (int, float)) else "n/a"
        lines.append(
            f"| {row['recovery_level']} | {row['pruned_parameter_count']} | "
            f"{row['recovered_parameter_count']} | {row['actual_compression_ratio']:.2f}x | "
            f"{row['pre_recovery_validation_accuracy']:.2f} | {best_str} | {test_str} | "
            f"{row['recovery_seconds']:.1f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_root


def run_multiseed_ablation(config: Dict[str, Any], checkpoint: Path) -> Path:
    seeds = config.get("seeds")
    if seeds is None:
        seeds = [int(config["seed"])]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list when provided")

    base_root = Path(config["logging"]["output_root"])
    seed_summaries: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg["seed"] = int(seed)
        cfg["logging"] = dict(cfg.get("logging", {}))
        cfg["logging"]["output_root"] = str(base_root / f"seed_{seed}")
        cfg["run_label"] = f"{config.get('run_label', 'cifar_recovery_ablation')}_seed_{seed}"
        print(f"[INFO] Running recovery ablation seed={seed}")
        seed_dir = run_ablation(cfg, checkpoint)
        summary = json.loads((seed_dir / "ablation_summary.json").read_text(encoding="utf-8"))
        seed_summaries.append(summary)

    levels = sorted({
        int(row["recovery_level"])
        for summary in seed_summaries
        for row in summary.get("rows", [])
    })
    aggregate_rows: List[Dict[str, Any]] = []
    for level in levels:
        vals = []
        tests = []
        comps = []
        for summary in seed_summaries:
            for row in summary.get("rows", []):
                if int(row["recovery_level"]) != level:
                    continue
                if isinstance(row.get("best_validation_accuracy"), (int, float)):
                    vals.append(float(row["best_validation_accuracy"]))
                if isinstance(row.get("test_accuracy"), (int, float)):
                    tests.append(float(row["test_accuracy"]))
                if isinstance(row.get("actual_compression_ratio"), (int, float)):
                    comps.append(float(row["actual_compression_ratio"]))
        aggregate_rows.append({
            "recovery_level": level,
            "best_validation_accuracy": _mean_std(vals),
            "test_accuracy": _mean_std(tests),
            "actual_compression_ratio": _mean_std(comps),
        })

    aggregate = {
        "protocol": "cifar_recovery_ablation_multiseed",
        "seeds": [int(seed) for seed in seeds],
        "checkpoint": str(checkpoint),
        "target_compression_ratio": float(
            config.get("comparison", {}).get("target_compression_ratio", 1.0)
        ),
        "seed_summaries": seed_summaries,
        "aggregate_rows": aggregate_rows,
    }
    base_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = base_root / "ablation_aggregate.json"
    aggregate_path.write_text(json.dumps(to_json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# CIFAR Recovery Ablation (Multi-seed)",
        "",
        f"- seeds: {', '.join(str(int(seed)) for seed in seeds)}",
        f"- target compression: {aggregate['target_compression_ratio']:.1f}x",
        "",
        "| Level | val mean±std | test mean±std | compression mean±std | n |",
        "|---|---|---|---|---:|",
    ]
    for row in aggregate_rows:
        val = row["best_validation_accuracy"]
        test = row["test_accuracy"]
        comp = row["actual_compression_ratio"]
        val_str = (
            f"{val['mean']:.2f}±{val['std']:.2f}" if val["mean"] is not None else "n/a"
        )
        test_str = (
            f"{test['mean']:.2f}±{test['std']:.2f}" if test["mean"] is not None else "n/a"
        )
        comp_str = (
            f"{comp['mean']:.2f}±{comp['std']:.2f}" if comp["mean"] is not None else "n/a"
        )
        lines.append(
            f"| {row['recovery_level']} | {val_str} | {test_str} | {comp_str} | {val['n']} |"
        )
    (base_root / "ABLATION_AGGREGATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR recovery ablation")
    parser.add_argument("--config", default="configs/cifar_recovery_ablation.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_resnet18_baseline.pth")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("seeds"):
        output = run_multiseed_ablation(config, Path(args.checkpoint))
    else:
        output = run_ablation(config, Path(args.checkpoint))
    print(f"[OK] Ablation artifacts: {output}")


if __name__ == "__main__":
    main()
