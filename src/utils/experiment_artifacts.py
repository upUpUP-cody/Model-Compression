"""Configuration, reproducibility, and run artifact helpers."""
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


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
    if config["hardware"]["device"] != "cpu":
        raise ValueError("the MVP runner supports only device: cpu")
    if config["hardware"].get("mixed_precision", False):
        raise ValueError("mixed_precision must be false for the CPU MVP")
    if config["dataset"]["batch_size"] <= 0:
        raise ValueError("dataset.batch_size must be positive")
    if config["search"]["max_iterations"] <= 0:
        raise ValueError("search.max_iterations must be positive")
    ratios = config["search"]["candidate_ratios"]
    if not isinstance(ratios, list) or not ratios or any(not 0.0 < float(ratio) < 1.0 for ratio in ratios):
        raise ValueError("search.candidate_ratios must contain values in (0.0, 1.0)")
    if config["controller"]["max_accuracy_drop_points"] < 0:
        raise ValueError("controller.max_accuracy_drop_points must be non-negative")
    if config["recovery"]["epochs"] < 0:
        raise ValueError("recovery.epochs must be non-negative")


def set_seed(seed: int) -> None:
    """Set reproducible CPU seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


class RunArtifacts:
    """Persist configuration, event history, CSV, summary, and checkpoints."""

    def __init__(self, output_root: str | Path, run_name: str = "autonomous_search") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / f"{run_name}_{timestamp}"
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
            row = {
                "iteration": event.get("iteration"),
                "action": event.get("final_action", event.get("action")),
                "reason": event.get("final_reason", event.get("reason")),
                "fingerprint": event.get("fingerprint"),
                "cheap_accuracy": event.get("cheap_critic", {}).get("accuracy"),
                "cheap_loss": event.get("cheap_critic", {}).get("loss"),
                "parameter_count": event.get("cheap_critic", {}).get("parameter_count"),
                "validation_accuracy": event.get("validation", {}).get("accuracy"),
            }
            rows.append(row)
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()) if rows else ["iteration"])
            writer.writeheader()
            writer.writerows(rows)

    def save_checkpoint(self, model: torch.nn.Module, name: str = "accepted_model.pth") -> Path:
        path = self.run_dir / name
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def _write_json(self, path: Path, value: Any) -> None:
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(to_json_safe(value), json_file, indent=2, sort_keys=True)
