"""Run validation-only CIFAR P1.2 study or final frozen-model test report."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomous_search import full_evaluate
from src.experiments.cifar_p12_comparison import METHOD_NAMES, results_to_records, run_comparison
from src.experiments.model_factory import build_model_from_config
from src.utils.data_loader import cifar10_split_metadata, get_cifar10_loaders
from src.utils.device import configure_cuda, resolve_device
from src.utils.experiment_artifacts import (
    RunArtifacts,
    config_hash,
    file_sha256,
    git_sha,
    load_config,
    runtime_metadata,
    set_cpu_threads,
    set_seed,
)
from src.utils.gpu_benchmark import benchmark_model


def load_baseline(model: torch.nn.Module, checkpoint_path: Path, device: str | torch.device = "cpu") -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(resolve_device(device))


def _gpu_inference_benchmark(model: torch.nn.Module, loader, device: str, precision: str) -> Dict[str, Any]:
    batch = next(iter(loader))
    data = batch[0][: min(32, batch[0].shape[0])]
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    data = data.to(resolved)
    with torch.inference_mode():
        return benchmark_model(model, data, device=resolved, warmup=1, repeat=5, batch_size=int(data.shape[0]))


def run_study(config: Dict[str, Any], checkpoint_source: Path, command: str) -> Path:
    device = resolve_device(config["hardware"]["device"])
    set_seed(config["seed"], deterministic=bool(config["hardware"].get("deterministic", True)))
    set_cpu_threads(config["hardware"].get("cpu_threads"))
    cuda_policy = configure_cuda(config)
    model = build_model_from_config(config).to(device)
    load_baseline(model, checkpoint_source, device)
    split_seed = config["dataset"].get("split_seed", config["seed"])
    train_loader, validation_loader = get_cifar10_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        validation_fraction=config["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
        pin_memory=config["dataset"].get("pin_memory"),
        persistent_workers=bool(config["dataset"].get("persistent_workers", False)),
        prefetch_factor=config["dataset"].get("prefetch_factor"),
    )
    split = cifar10_split_metadata(train_loader, validation_loader, split_seed)
    run_id = f"{config.get('run_label', 'cifar_p12')}_{config_hash(config)[:12]}"
    artifacts = RunArtifacts(config["logging"]["output_root"], run_name="cifar_p12_comparison", run_id=run_id)
    artifacts.save_config(config)
    results, models = run_comparison(model, train_loader, validation_loader, config)
    records = results_to_records(results)
    for record in records:
        result_model = models[record["method"]]
        checkpoint = artifacts.save_checkpoint(result_model, f"{record['method']}.pth")
        record["checkpoint"] = str(checkpoint)
        record["checkpoint_sha256"] = file_sha256(checkpoint)
    comparison_json, comparison_csv = artifacts.save_comparison(records)
    runtime = runtime_metadata(str(device), config["hardware"].get("precision", "fp32"))
    runtime["inference_benchmark"] = _gpu_inference_benchmark(
        models["dense_baseline"],
        validation_loader,
        str(device),
        str(config["hardware"].get("precision", "fp32")),
    )
    manifest = {
        "protocol": "p12_validation_only_study",
        "dataset": "cifar10",
        "selection_frozen": True,
        "config_hash": config_hash(config),
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_source": str(checkpoint_source),
        "checkpoint_source_sha256": file_sha256(checkpoint_source),
        "split": split,
        "seed": config["seed"],
        "command": command,
        "runtime": runtime,
        "cuda_policy": cuda_policy,
        "comparison_json": str(comparison_json),
        "comparison_csv": str(comparison_csv),
        "records": records,
    }
    artifacts.save_manifest(manifest)
    artifacts.save_summary({"selection_frozen": True, "records": records})
    return artifacts.run_dir


def verify_frozen_study(study_dir: Path, config: Dict[str, Any], checkpoint_source: Path) -> Dict[str, Any]:
    study_dir = study_dir.resolve()
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("study manifest does not exist")
    if (study_dir / "final_test_report.json").exists():
        raise ValueError("study already has a final test report")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "p12_validation_only_study":
        raise ValueError("study protocol does not match P1.2 validation-only protocol")
    if manifest.get("selection_frozen") is not True:
        raise ValueError("study selection is not frozen")
    if manifest.get("config_hash") != config_hash(config):
        raise ValueError("study config hash does not match")
    if manifest.get("git_sha") != git_sha(PROJECT_ROOT):
        raise ValueError("study git SHA does not match")
    if manifest.get("checkpoint_source_sha256") != file_sha256(checkpoint_source):
        raise ValueError("study source checkpoint hash does not match")

    split = manifest.get("split")
    expected_split_seed = config["dataset"].get("split_seed", config["seed"])
    if not isinstance(split, dict) or split.get("split_seed") != expected_split_seed:
        raise ValueError("study split metadata does not match configured split seed")

    runtime = manifest.get("runtime")
    expected_device = resolve_device(config["hardware"]["device"])
    if expected_device.type == "cuda":
        device_meta = runtime.get("device") if isinstance(runtime, dict) else None
        if not isinstance(device_meta, dict) or device_meta.get("type") != "cuda":
            raise ValueError("study runtime is missing CUDA metadata")
        benchmark = runtime.get("inference_benchmark") if isinstance(runtime, dict) else None
        if not isinstance(benchmark, dict) or "peak_allocated_bytes" not in benchmark:
            raise ValueError("study runtime is missing GPU inference benchmark metadata")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(METHOD_NAMES):
        raise ValueError("study must contain exactly six frozen result records")
    methods = [record.get("method") for record in records if isinstance(record, dict)]
    if len(methods) != len(METHOD_NAMES) or set(methods) != set(METHOD_NAMES):
        raise ValueError("study frozen result methods do not match the P1.2 method set")
    return manifest


def run_test_report(study_dir: Path, config: Dict[str, Any], checkpoint_source: Path) -> Path:
    manifest = verify_frozen_study(study_dir, config, checkpoint_source)
    device = resolve_device(config["hardware"]["device"])
    configure_cuda(config)
    split_seed = config["dataset"].get("split_seed", config["seed"])
    train_loader, validation_loader, test_loader = get_cifar10_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        validation_fraction=config["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
        pin_memory=config["dataset"].get("pin_memory"),
        persistent_workers=bool(config["dataset"].get("persistent_workers", False)),
        prefetch_factor=config["dataset"].get("prefetch_factor"),
        return_test=True,
    )
    if cifar10_split_metadata(train_loader, validation_loader, split_seed) != manifest["split"]:
        raise ValueError("study split metadata does not match")
    reports = []
    for record in manifest["records"]:
        model = build_model_from_config(config).to(device)
        load_baseline(model, Path(record["checkpoint"]), device)
        reports.append({
            "method": record["method"],
            "checkpoint": record["checkpoint"],
            "test": full_evaluate(model, test_loader, config["hardware"]["device"]),
        })
    output = study_dir / "final_test_report.json"
    output.write_text(
        json.dumps({"study_dir": str(study_dir), "reports": reports}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR P1.2 comparison protocol")
    parser.add_argument("mode", choices=("study", "report-test"))
    parser.add_argument("--config", default="configs/cifar_p12_gpu_smoke.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_resnet18_baseline.pth")
    parser.add_argument("--study-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if args.mode == "study":
        print(f"[OK] Validation-only study artifacts: {run_study(config, checkpoint, ' '.join(sys.argv))}")
    else:
        if args.study_dir is None:
            parser.error("--study-dir is required for report-test")
        print(f"[OK] Final test report: {run_test_report(args.study_dir, config, checkpoint)}")


if __name__ == "__main__":
    main()
