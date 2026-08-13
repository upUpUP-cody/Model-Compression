"""Configuration, reproducibility, and run artifact helpers."""
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml

from src.utils.device import configure_cuda, device_metadata, resolve_device
from src.utils.precision import precision_metadata, resolve_precision


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate an autonomous-search YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_paths = [
        ("hardware", "device"),
        ("dataset", "batch_size"),
        ("search", "max_iterations"),
        ("search", "candidate_ratios"),
        ("controller", "max_accuracy_drop_points"),
        ("recovery", "epochs"),
        ("seed",),
    ]
    for path in required_paths:
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise ValueError(f"missing configuration key: {'.'.join(path)}")
            current = current[key]
    hardware = config["hardware"]
    try:
        device = resolve_device(hardware["device"])
        precision = resolve_precision(hardware.get("precision", "fp32"), device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid hardware configuration: {error}") from error
    if device.type == "cpu" and hardware.get("mixed_precision", False):
        raise ValueError("mixed_precision is unavailable on CPU; use precision: fp32")
    if hardware.get("mixed_precision", False) and precision == "fp32":
        raise ValueError("mixed_precision requires precision fp16 or bf16")
    if config["dataset"]["batch_size"] <= 0:
        raise ValueError("dataset.batch_size must be positive")
    if config["search"]["max_iterations"] <= 0:
        raise ValueError("search.max_iterations must be positive")
    if not isinstance(config["search"].get("candidates_per_round"), int) or config["search"]["candidates_per_round"] < 1:
        raise ValueError("search.candidates_per_round must be positive")
    if not isinstance(config["search"].get("recovery_top_k", 1), int) or config["search"].get("recovery_top_k", 1) < 1:
        raise ValueError("search.recovery_top_k must be positive")
    if not isinstance(config["search"].get("enable_two_layer_candidates", False), bool):
        raise ValueError("search.enable_two_layer_candidates must be boolean")
    ratios = config["search"]["candidate_ratios"]
    if not isinstance(ratios, list) or not ratios or any(not 0.0 < float(ratio) < 1.0 for ratio in ratios):
        raise ValueError("search.candidate_ratios must contain values in (0.0, 1.0)")
    if config["controller"]["max_accuracy_drop_points"] < 0:
        raise ValueError("controller.max_accuracy_drop_points must be non-negative")
    if config["recovery"]["epochs"] < 0:
        raise ValueError("recovery.epochs must be non-negative")
    if not 0.0 < float(config["dataset"].get("validation_fraction", 0.1)) < 1.0:
        raise ValueError("dataset.validation_fraction must be between 0 and 1")
    if not isinstance(config["dataset"].get("split_seed", config["seed"]), int):
        raise ValueError("dataset.split_seed must be an integer")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set reproducible Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def to_json_safe(value: Any) -> Any:
    """Convert supported experiment values to JSON-native types."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError("only scalar tensors are JSON-safe")
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def set_cpu_threads(num_threads: Optional[int] = None) -> int:
    """Set and return the configured CPU intra-op thread count."""
    if num_threads is not None:
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        torch.set_num_threads(num_threads)
    return int(torch.get_num_threads())


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Dict[str, Any]) -> str:
    encoded = json.dumps(to_json_safe(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha(project_root: str | Path = ".") -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(device: str | torch.device = "cpu", precision: str = "fp32") -> Dict[str, Any]:
    """Return serializable host, device, and precision metadata."""
    resolved = resolve_device(device)
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cpu_threads": int(torch.get_num_threads()),
        "device": device_metadata(resolved),
        "precision": precision_metadata(resolved, precision),
    }
    if resolved.type == "cuda":
        metadata["cuda_policy"] = configure_cuda({"hardware": {"device": str(resolved)}})
    return metadata


class RunArtifacts:
    """Persist configuration, event history, CSV, summary, and checkpoints."""

    def __init__(self, output_root: str | Path, run_name: str = "autonomous_search", run_id: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = run_id or timestamp
        self.run_dir = Path(output_root) / f"{run_name}_{suffix}"
        counter = 1
        while self.run_dir.exists():
            self.run_dir = Path(output_root) / f"{run_name}_{suffix}_{counter}"
            counter += 1
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"

    def save_config(self, config: Dict[str, Any]) -> None:
        normalized = to_json_safe(config)
        with (self.run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(normalized, config_file, sort_keys=True)
        self._write_json(self.run_dir / "resolved_config.json", normalized)

    def append_event(self, event: Dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(to_json_safe(event), sort_keys=True) + "\n")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        self._write_json(self.run_dir / "summary.json", summary)

    def save_history_csv(self, events: Iterable[Dict[str, Any]]) -> None:
        rows = []
        for event in events:
            candidates = event.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    rows.append(self._history_row(event, candidate))
            else:
                rows.append(self._history_row(event, event))
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "iteration", "generation_order", "candidate_type", "audit_status", "action",
                "reason", "fingerprint", "cheap_accuracy", "cheap_loss", "parameter_count",
                "validation_accuracy", "shortlist_rank", "recovery_rank",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _history_row(event: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        cheap_critic = candidate.get("cheap_critic", event.get("cheap_critic", {}))
        validation = candidate.get("validation", event.get("validation", {}))
        return {
            "iteration": event.get("iteration"),
            "generation_order": candidate.get("generation_order"),
            "candidate_type": candidate.get("candidate_type"),
            "audit_status": candidate.get("audit_status"),
            "action": candidate.get("final_action", event.get("final_action", event.get("action"))),
            "reason": candidate.get("final_reason", event.get("final_reason", event.get("reason"))),
            "fingerprint": candidate.get("fingerprint", event.get("fingerprint")),
            "cheap_accuracy": cheap_critic.get("accuracy"),
            "cheap_loss": cheap_critic.get("loss"),
            "parameter_count": candidate.get("actual_parameter_count", cheap_critic.get("parameter_count")),
            "validation_accuracy": validation.get("accuracy"),
            "shortlist_rank": candidate.get("shortlist_rank"),
            "recovery_rank": candidate.get("recovery_rank"),
        }

    def save_frontier(self, frontier: Any) -> tuple[Path, Path]:
        """Persist a Pareto frontier as JSON and a flat CSV table."""
        payload = frontier.to_dict() if hasattr(frontier, "to_dict") else frontier
        json_path = self.run_dir / "frontier.json"
        self._write_json(json_path, payload)
        csv_path = self.run_dir / "frontier.csv"
        points = payload.get("points", []) if isinstance(payload, dict) else []
        fieldnames = ["validation_accuracy", "parameter_count", "compression_ratio", "recovery_seconds", "seed", "run", "iteration"]
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for point in points:
                writer.writerow({field: point.get(field) for field in fieldnames})
        return json_path, csv_path

    def save_manifest(self, manifest: Dict[str, Any]) -> Path:
        path = self.run_dir / "manifest.json"
        self._write_json(path, manifest)
        return path

    def save_comparison(self, records: Iterable[Dict[str, Any]]) -> tuple[Path, Path]:
        """Persist validation-only comparison records as JSON and CSV."""
        normalized = [to_json_safe(record) for record in records]
        json_path = self.run_dir / "comparison.json"
        self._write_json(json_path, {"records": normalized})
        csv_path = self.run_dir / "comparison.csv"
        fieldnames = [
            "method", "status", "target_compression_ratio", "baseline_parameter_count",
            "parameter_count", "compression_ratio", "validation_accuracy", "validation_loss",
            "selection_seconds", "recovery_seconds", "checkpoint", "checkpoint_sha256",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in normalized:
                validation = record.get("validation", {})
                writer.writerow({
                    "method": record.get("method"),
                    "status": record.get("status"),
                    "target_compression_ratio": record.get("target_compression_ratio"),
                    "baseline_parameter_count": record.get("baseline_parameter_count"),
                    "parameter_count": record.get("parameter_count"),
                    "compression_ratio": record.get("compression_ratio"),
                    "validation_accuracy": validation.get("accuracy"),
                    "validation_loss": validation.get("loss"),
                    "selection_seconds": record.get("selection_seconds"),
                    "recovery_seconds": record.get("recovery_seconds"),
                    "checkpoint": record.get("checkpoint"),
                    "checkpoint_sha256": record.get("checkpoint_sha256"),
                })
        return json_path, csv_path

    def measure_inference(self, model: torch.nn.Module, loader: Any, device: str = "cpu", warmup: int = 1) -> Dict[str, float]:
        """Measure warm inference latency and throughput for a fixed loader."""
        model = model.to(device)
        model.eval()
        iterator = iter(loader)
        batches = []
        for _ in range(max(1, warmup + 1)):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        if not batches:
            raise ValueError("loader must contain at least one batch")
        for data, _ in batches[:warmup]:
            with torch.inference_mode():
                model(data.to(device))
        data, _ = batches[-1]
        start = time.perf_counter()
        with torch.inference_mode():
            model(data.to(device))
        elapsed = time.perf_counter() - start
        batch_size = int(data.shape[0])
        return {
            "latency_seconds": float(elapsed),
            "throughput_samples_per_second": float(batch_size / max(elapsed, 1e-12)),
        }

    def save_checkpoint(self, model: torch.nn.Module, name: str = "accepted_model.pth") -> Path:
        path = self.run_dir / name
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def _write_json(self, path: Path, value: Any) -> None:
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(to_json_safe(value), json_file, indent=2, sort_keys=True)
