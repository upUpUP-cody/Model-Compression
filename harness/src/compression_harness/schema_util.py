"""Load and lightly validate Goal / Recipe YAML against JSON Schema when available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from compression_harness.paths import SCHEMAS_DIR


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_instance(instance: dict[str, Any], schema_name: str) -> list[str]:
    """Return list of error strings; empty if ok.

    Uses jsonschema if installed; otherwise only checks required top-level keys
    from the schema's required list (lightweight fallback).
    """
    schema = load_schema(schema_name)
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        return [e.message for e in sorted(validator.iter_errors(instance), key=lambda e: list(e.path))]
    except ImportError:
        errors: list[str] = []
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                errors.append(f"missing required key: {key}")
        return errors
