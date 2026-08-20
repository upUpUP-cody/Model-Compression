"""Ablate high-compression crossover: Uniform vs Incremental-no-gate vs Search-gated."""
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

from src.autonomous_search import AutonomousSearch, full_evaluate
from src.controller.heuristic_controller import HeuristicController
from src.experiments.cifar_p12_comparison import (
    _keep_indices,
    _layer_ratios,
    parameter_count,
)
from src.experiments.compression_targets import apply_compression_target, ensure_compression_target
from src.experiments.model_factory import build_model_from_config, model_type_from_config
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


def _load_baseline(model: torch.nn.Module, checkpoint: Path, device) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)


def _run_uniform(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """One-shot Wanda prune to target + Level-1 recovery."""
    device = str(config["hardware"]["device"])
    model_type = model_type_from_config(config)
    comparison = config["comparison"]
    backend = resolve_pruning_backend(model, model_type)
    ratios = _layer_ratios(model, comparison.get("oneshot_layer_ratios", {}), model_type)
    keep_indices = _keep_indices(
        backend, model, train_loader, ratios, "wanda", device, comparison, model_type
    )
    pruned = backend.create_pruned_model_by_indices(keep_indices).to(resolve_device(device))
    baseline_count = parameter_count(model)
    pruned_count = parameter_count(pruned)
    pre_val = float(evaluate_validation(pruned, validation_loader, device)["accuracy"])
    start = time.perf_counter()
    recovered, history = run_recovery(
        pruned, train_loader, validation_loader, config, teacher_model=model, verbose=False
    )
    seconds = time.perf_counter() - start
    post_val = float(evaluate_validation(recovered, validation_loader, device)["accuracy"])
    return {
        "arm": "uniform",
        "compression_ratio": baseline_count / max(pruned_count, 1),
        "parameter_count": parameter_count(recovered),
        "pre_recovery_validation_accuracy": pre_val,
        "validation_accuracy": post_val,
        "recovery_seconds": seconds,
        "details": {"layer_keep_indices": keep_indices, "recovery": history},
        "model": recovered,
    }


def _run_incremental_no_gate(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Search-style incremental steps with capability gate disabled."""
    cfg = copy.deepcopy(config)
    cfg["controller"] = dict(cfg.get("controller", {}))
    cfg["controller"]["max_accuracy_drop_points"] = 100.0
    search_config = cfg["search"]
    recovery_config = cfg["recovery"]
    controller = HeuristicController(**dict(cfg["controller"]))
    search = AutonomousSearch(controller=controller, model_type=model_type_from_config(cfg))
    baseline_count = parameter_count(model)
    start = time.perf_counter()
    accepted, history = search.run(
        model,
        train_loader,
        validation_loader,
        max_iterations=int(search_config["max_iterations"]),
        candidate_ratios=list(search_config["candidate_ratios"]),
        candidates_per_round=int(search_config["candidates_per_round"]),
        cheap_eval_samples=int(search_config["cheap_eval_samples"]),
        recovery_epochs=int(recovery_config["epochs"]),
        recovery_learning_rate=float(recovery_config["learning_rate"]),
        device=str(cfg["hardware"]["device"]),
        precision=str(cfg["hardware"].get("precision", "fp32")),
        enable_two_layer_candidates=bool(search_config.get("enable_two_layer_candidates", False)),
        recovery_top_k=int(search_config.get("recovery_top_k", 1)),
        target_compression_ratio=float(cfg["comparison"]["target_compression_ratio"]),
        max_step_compression=float(search_config.get("max_step_compression", 1.75)),
    )
    seconds = time.perf_counter() - start
    device = str(cfg["hardware"]["device"])
    post_val = float(evaluate_validation(accepted, validation_loader, device)["accuracy"])
    hist = history.to_dict()
    return {
        "arm": "incremental_no_gate",
        "compression_ratio": baseline_count / max(parameter_count(accepted), 1),
        "parameter_count": parameter_count(accepted),
        "validation_accuracy": post_val,
        "recovery_seconds": seconds,
        "details": {"history": hist, "max_accuracy_drop_points": 100.0},
        "model": accepted,
    }


def _formal_study_dirname(target: float, seed: int) -> str:
    return f"ratio_{float(target):g}_seed_{seed}"


def _load_search_gated_from_formal(
    seed: int,
    formal_root: Path,
    target: float,
) -> Dict[str, Any] | None:
    study = formal_root / _formal_study_dirname(target, seed)
    if not study.is_dir():
        return None
    manifest_path = next(study.rglob("manifest.json"), None)
    test_path = next(study.rglob("final_test_report.json"), None)
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_by_method: Dict[str, float] = {}
    if test_path is not None:
        for report in json.loads(test_path.read_text(encoding="utf-8")).get("reports", []):
            accuracy = (report.get("test") or {}).get("accuracy")
            if accuracy is not None:
                test_by_method[report["method"]] = float(accuracy)
    for record in manifest.get("records", []):
        if record.get("method") != "autonomous_search":
            continue
        validation = (record.get("validation") or {}).get("accuracy")
        return {
            "arm": "search_gated",
            "compression_ratio": float(record["compression_ratio"]),
            "parameter_count": int(record["parameter_count"]),
            "validation_accuracy": float(validation) if validation is not None else None,
            "test_accuracy": test_by_method.get("autonomous_search"),
            "source": str(manifest_path),
            "details": record.get("details") or {},
            "reused": True,
        }
    return None


def run_seed(
    config: Dict[str, Any],
    checkpoint: Path,
    seed: int,
    formal_root: Path,
) -> Dict[str, Any]:
    cfg = apply_compression_target(copy.deepcopy(config), float(config["comparison"]["target_compression_ratio"]))
    cfg["seed"] = int(seed)
    device = resolve_device(cfg["hardware"]["device"])
    device_str = str(device)
    set_seed(int(seed), deterministic=bool(cfg["hardware"].get("deterministic", True)))
    configure_cuda(cfg)

    model = build_model_from_config(cfg).to(device)
    _load_baseline(model, checkpoint, device)
    split_seed = cfg["dataset"].get("split_seed", cfg["seed"])
    train_loader, validation_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["dataset"]["data_dir"],
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        validation_fraction=cfg["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
        pin_memory=bool(cfg["dataset"].get("pin_memory", False)),
        persistent_workers=bool(cfg["dataset"].get("persistent_workers", False)),
        return_test=True,
    )

    baseline_val = float(evaluate_validation(model, validation_loader, device_str)["accuracy"])
    print(f"[INFO] seed={seed} baseline_val={baseline_val:.2f}% target={cfg['comparison']['target_compression_ratio']}x")

    arms: List[Dict[str, Any]] = []
    uniform = _run_uniform(copy.deepcopy(model), train_loader, validation_loader, cfg)
    uniform["test_accuracy"] = float(full_evaluate(uniform.pop("model"), test_loader, device_str)["accuracy"])
    print(
        f"[INFO] seed={seed} uniform comp={uniform['compression_ratio']:.3f}x "
        f"val={uniform['validation_accuracy']:.2f} test={uniform['test_accuracy']:.2f}"
    )
    arms.append(uniform)

    no_gate = _run_incremental_no_gate(copy.deepcopy(model), train_loader, validation_loader, cfg)
    no_gate["test_accuracy"] = float(full_evaluate(no_gate.pop("model"), test_loader, device_str)["accuracy"])
    print(
        f"[INFO] seed={seed} incremental_no_gate comp={no_gate['compression_ratio']:.3f}x "
        f"val={no_gate['validation_accuracy']:.2f} test={no_gate['test_accuracy']:.2f}"
    )
    arms.append(no_gate)

    gated = _load_search_gated_from_formal(
        seed,
        formal_root,
        float(cfg["comparison"]["target_compression_ratio"]),
    )
    if gated is None:
        raise FileNotFoundError(
            f"missing formal100_full search result for "
            f"{_formal_study_dirname(float(cfg['comparison']['target_compression_ratio']), seed)}"
        )
    if gated.get("test_accuracy") is None:
        # Should not happen for formal100_full; keep explicit failure.
        raise ValueError(f"formal100_full search for seed {seed} lacks test accuracy")
    print(
        f"[INFO] seed={seed} search_gated (reused) comp={gated['compression_ratio']:.3f}x "
        f"val={gated['validation_accuracy']} test={gated['test_accuracy']:.2f}"
    )
    arms.append(gated)

    return {
        "seed": seed,
        "baseline_validation_accuracy": baseline_val,
        "target_compression_ratio": float(cfg["comparison"]["target_compression_ratio"]),
        "arms": arms,
    }


def aggregate(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in seed_rows:
        for arm in row["arms"]:
            by_arm.setdefault(arm["arm"], []).append(arm)
    aggregate_rows = []
    for arm_name, items in sorted(by_arm.items()):
        tests = [float(item["test_accuracy"]) for item in items if item.get("test_accuracy") is not None]
        vals = [
            float(item["validation_accuracy"])
            for item in items
            if item.get("validation_accuracy") is not None
        ]
        comps = [float(item["compression_ratio"]) for item in items]
        aggregate_rows.append({
            "arm": arm_name,
            "test_accuracy": _mean_std(tests),
            "validation_accuracy": _mean_std(vals),
            "compression_ratio": _mean_std(comps),
        })
    return {"seed_rows": seed_rows, "aggregate_rows": aggregate_rows}


def _verdict(aggregate_rows: List[Dict[str, Any]]) -> str:
    by_name = {row["arm"]: row for row in aggregate_rows}
    needed = ("uniform", "incremental_no_gate", "search_gated")
    if any(name not in by_name or by_name[name]["test_accuracy"]["mean"] is None for name in needed):
        return "incomplete"
    u = by_name["uniform"]["test_accuracy"]["mean"]
    n = by_name["incremental_no_gate"]["test_accuracy"]["mean"]
    s = by_name["search_gated"]["test_accuracy"]["mean"]
    # 1.0 point margins for coarse attribution.
    if n - u >= 1.0 and abs(s - n) < 1.0:
        return "path_dominant"
    if s - n >= 1.0 and abs(n - u) < 1.0:
        return "gate_dominant"
    if s - n >= 1.0 and n - u >= 1.0:
        return "path_and_gate"
    if abs(u - n) < 1.0 and abs(n - s) < 1.0:
        return "inconclusive_close"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR crossover path ablation")
    parser.add_argument("--config", default="configs/cifar_crossover_path_ablation.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_resnet18_baseline_formal100.pth")
    parser.add_argument(
        "--formal-root",
        default="results/cifar_p12_comparison_gpu_formal100_full",
        help="Reuse search_gated arms from this formal100 full sweep",
    )
    args = parser.parse_args()
    config = ensure_compression_target(load_config(args.config))
    seeds = config.get("seeds") or [int(config["seed"])]
    formal_root = Path(args.formal_root)
    seed_rows = []
    for seed in seeds:
        seed_rows.append(run_seed(config, Path(args.checkpoint), int(seed), formal_root))

    bundled = aggregate(seed_rows)
    bundled["protocol"] = "cifar_crossover_path_ablation"
    bundled["checkpoint"] = str(args.checkpoint)
    bundled["formal_root"] = str(formal_root)
    bundled["verdict"] = _verdict(bundled["aggregate_rows"])
    output_root = Path(config["logging"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(to_json_safe(bundled), indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# CIFAR Crossover Path Ablation",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- target: {float(config['comparison']['target_compression_ratio']):.1f}x",
        f"- seeds: {', '.join(str(int(seed)) for seed in seeds)}",
        f"- search_gated source: `{formal_root}` (reused)",
        f"- verdict: **{bundled['verdict']}**",
        "",
        "| Arm | test mean±std | val mean±std | compression mean±std | n |",
        "|---|---|---|---|---:|",
    ]
    for row in bundled["aggregate_rows"]:
        t = row["test_accuracy"]
        v = row["validation_accuracy"]
        c = row["compression_ratio"]
        t_str = f"{t['mean']:.2f}±{t['std']:.2f}" if t["mean"] is not None else "n/a"
        v_str = f"{v['mean']:.2f}±{v['std']:.2f}" if v["mean"] is not None else "n/a"
        c_str = f"{c['mean']:.2f}±{c['std']:.2f}" if c["mean"] is not None else "n/a"
        lines.append(f"| {row['arm']} | {t_str} | {v_str} | {c_str} | {t['n']} |")
    lines.extend([
        "",
        "Verdict key: `path_dominant` = incremental path explains gain; "
        "`gate_dominant` = 2pt gate explains gain; "
        "`path_and_gate` = both; `inconclusive_close` = arms too close.",
        "",
    ])
    (output_root / "ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Ablation artifacts: {output_root}")
    print(f"[OK] Verdict: {bundled['verdict']}")


if __name__ == "__main__":
    main()
