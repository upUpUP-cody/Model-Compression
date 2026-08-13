"""Aggregate P1.2 study outputs across seeds and compression targets."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def discover_study_dirs(root: Path) -> List[Path]:
    """Return study directories containing a manifest.json file."""
    root = root.resolve()
    if (root / "manifest.json").is_file():
        return [root]
    studies = sorted(path.parent for path in root.rglob("manifest.json"))
    return studies


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_study_row(study_dir: Path) -> List[Dict[str, Any]]:
    """Load one row per method from a frozen study directory."""
    manifest = _load_json(study_dir / "manifest.json")
    comparison = _load_json(study_dir / "comparison.json")
    test_report = _load_json(study_dir / "final_test_report.json") if (study_dir / "final_test_report.json").is_file() else None
    test_by_method = {}
    if test_report:
        test_by_method = {entry["method"]: entry["test"] for entry in test_report.get("reports", [])}
    comparison_by_method = {record["method"]: record for record in comparison.get("records", [])}
    rows: List[Dict[str, Any]] = []
    for record in manifest.get("records", []):
        method = record["method"]
        comparison_record = comparison_by_method.get(method, {})
        validation = comparison_record.get("validation", record.get("validation", {}))
        test = test_by_method.get(method, {})
        rows.append({
            "study_dir": str(study_dir),
            "seed": manifest.get("seed"),
            "target_compression_ratio": comparison_record.get("target_compression_ratio"),
            "method": method,
            "validation_accuracy": validation.get("accuracy"),
            "validation_loss": validation.get("loss"),
            "test_accuracy": test.get("accuracy"),
            "test_loss": test.get("loss"),
            "compression_ratio": comparison_record.get("compression_ratio", record.get("compression_ratio")),
            "parameter_count": comparison_record.get("parameter_count", record.get("parameter_count")),
            "config_hash": manifest.get("config_hash"),
        })
    return rows


def collect_rows(study_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for study_dir in study_dirs:
        rows.extend(load_study_row(study_dir))
    return rows


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compute mean/std grouped by method and target compression ratio."""
    grouped: Dict[tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("method"), row.get("target_compression_ratio"))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (method, target_ratio), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        val_accs = [float(item["validation_accuracy"]) for item in items if item.get("validation_accuracy") is not None]
        test_accs = [float(item["test_accuracy"]) for item in items if item.get("test_accuracy") is not None]
        summary.append({
            "method": method,
            "target_compression_ratio": target_ratio,
            "num_runs": len(items),
            "validation_accuracy_mean": statistics.mean(val_accs) if val_accs else None,
            "validation_accuracy_std": statistics.pstdev(val_accs) if len(val_accs) > 1 else 0.0,
            "test_accuracy_mean": statistics.mean(test_accs) if test_accs else None,
            "test_accuracy_std": statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0,
        })
    return summary


def write_aggregate_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    payload = {"rows": rows, "summary": summary}
    json_path = output_dir / "aggregate_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "aggregate_summary.csv"
    fieldnames = [
        "study_dir", "seed", "target_compression_ratio", "method",
        "validation_accuracy", "validation_loss", "test_accuracy", "test_loss",
        "compression_ratio", "parameter_count", "config_hash",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def aggregate_root(root: Path, output_dir: Path | None = None) -> Dict[str, Any]:
    study_dirs = discover_study_dirs(root)
    if not study_dirs:
        raise ValueError(f"no study directories found under {root}")
    rows = collect_rows(study_dirs)
    destination = output_dir or (root / "aggregate")
    json_path, csv_path = write_aggregate_outputs(rows, destination)
    return {
        "study_dirs": [str(path) for path in study_dirs],
        "aggregate_json": str(json_path),
        "aggregate_csv": str(csv_path),
        "summary": summarize_rows(rows),
    }
