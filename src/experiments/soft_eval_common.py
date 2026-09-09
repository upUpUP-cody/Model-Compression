"""Soft-eval side protocol: Sentiment + easy/soft/lite capability variants (not Gate A)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.experiments.stage_a_common import ensure_dir, write_json

SOFT_EVAL_CHECKPOINT_VERSION = 1
SOFT_DIM_ORDER: Tuple[str, ...] = (
    "Sentiment",
    "Math_easy",
    "Reasoning_easy",
    "Knowledge_easy",
    "Instruction_easy",
    "Code_easy",
)

SOFT6_DIM_ORDER: Tuple[str, ...] = (
    "PPL_soft",
    "Math_soft",
    "Knowledge_soft",
    "Reasoning_soft",
    "Instruction_soft",
    "Code_soft",
)

SOFT_LITE_DIM_ORDER: Tuple[str, ...] = (
    "Math_lite",
    "Instruction_lite",
    "Code_lite",
)

SOFT_DIM_SPECS: Dict[str, Dict[str, Any]] = {
    "Sentiment": {
        "kind": "glue_sst2",
        "metric": "classification_acc",
        "max_samples": 256,
        "max_seq_len": 256,
        "max_new_tokens": 8,
    },
    "Math_easy": {
        "kind": "capability",
        "base_dim": "Math",
        "scan_limit": 32,
        "num_fewshot": 3,
    },
    "Reasoning_easy": {
        "kind": "capability",
        "base_dim": "Reasoning",
        "task": "bbh_fewshot_boolean_expressions_easy",
        "task_include_path": "configs/lm_eval_tasks/easy",
        "scan_limit": 32,
        # Q/A completion (not chat); short gens + until avoid emoji/role leak / degenerate loops.
        "apply_chat_template": False,
        "gen_kwargs": {
            "max_gen_toks": 32,
            "do_sample": False,
            "until": ["\n", "Q:", "Q ", "</s>", "<|im_end|>", "<|endoftext|>"],
        },
        "metric_candidates": (
            "exact_match,flexible-extract",
            "exact_match,strict-match",
            "exact_match,none",
            "exact_match",
        ),
    },
    "Knowledge_easy": {
        "kind": "capability",
        "base_dim": "Knowledge",
        "task": "mmlu_high_school_psychology",
        "scan_limit": 64,
        "num_fewshot": 1,
    },
    "Instruction_easy": {
        "kind": "capability",
        "base_dim": "Instruction",
        "scan_limit": 32,
        "metric_candidates": (
            "prompt_level_loose_acc,none",
            "prompt_level_loose_acc",
            "prompt_level_strict_acc,none",
            "prompt_level_strict_acc",
        ),
    },
    "Code_easy": {
        "kind": "capability",
        "base_dim": "Code",
        "scan_limit": 16,
    },
    "PPL_soft": {
        "kind": "capability",
        "base_dim": "PPL",
        "scan_limit": 4,
        "max_length": 128,
    },
    "Math_soft": {
        "kind": "capability",
        "base_dim": "Math",
        "task": "asdiv",
        "scan_limit": 32,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "metric_candidates": (
            "acc,none",
            "acc",
            "exact_match,flexible-extract",
            "exact_match",
        ),
    },
    "Knowledge_soft": {
        "kind": "capability",
        "base_dim": "Knowledge",
        "task": "arc_easy",
        "scan_limit": 64,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "metric_candidates": (
            "acc,none",
            "acc",
            "acc_norm,none",
            "acc_norm",
        ),
    },
    "Reasoning_soft": {
        "kind": "capability",
        "base_dim": "Reasoning",
        "task": "boolq",
        "scan_limit": 64,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "metric_candidates": (
            "acc,none",
            "acc",
        ),
    },
    "Instruction_soft": {
        "kind": "capability",
        "base_dim": "Instruction",
        "scan_limit": 16,
        "metric_candidates": (
            "prompt_level_loose_acc,none",
            "prompt_level_loose_acc",
            "prompt_level_strict_acc,none",
            "prompt_level_strict_acc",
        ),
    },
    "Code_soft": {
        "kind": "capability",
        "base_dim": "Code",
        "task": "mbpp",
        "scan_limit": 16,
        "num_fewshot": 3,
        "apply_chat_template": False,
        "confirm_run_unsafe_code": True,
        "gen_kwargs": {
            "until": ["[DONE]"],
            "do_sample": False,
            "max_gen_toks": 256,
        },
        "metric_candidates": (
            "pass@1,none",
            "pass@1",
            "pass_at_1,none",
            "pass_at_1",
        ),
    },
    "Math_lite": {
        "kind": "capability",
        "base_dim": "Math",
        "task": "arithmetic_2da_local",
        "task_include_path": "configs/lm_eval_tasks/lite",
        "scan_limit": 64,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "metric_candidates": (
            "acc,none",
            "acc",
        ),
    },
    "Instruction_lite": {
        "kind": "glue_rte",
        # Primary score for ladder/report: balanced_accuracy (collapse-aware).
        "metric": "balanced_accuracy",
        "glue_task": "rte",
        "max_samples": 256,
        "max_seq_len": 256,
        "max_new_tokens": 8,
    },
    "Code_lite": {
        "kind": "capability",
        "base_dim": "Code",
        "task": "humaneval_single_line_infilling_local",
        "task_include_path": "configs/lm_eval_tasks/lite",
        "scan_limit": 16,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "confirm_run_unsafe_code": True,
        "gen_kwargs": {
            "until": ["\n"],
            "do_sample": False,
            "max_gen_toks": 64,
        },
        "metric_candidates": (
            "pass@1,create_test",
            "pass@1,none",
            "pass@1",
            "pass_at_1,none",
            "pass_at_1",
        ),
    },
}


def soft_eval_checkpoint_filename(*, smoke: bool = False) -> str:
    return "soft_eval_checkpoint_smoke.json" if smoke else "soft_eval_checkpoint.json"


def soft_cell_key(sparsity: float, dim: str) -> str:
    return f"{float(sparsity):.2f}|{dim}"


def soft_protocol_canonical(config: Mapping[str, Any], *, smoke: bool) -> Dict[str, Any]:
    dims = dict(config.get("soft_dimensions") or SOFT_DIM_SPECS)
    return {
        "smoke": bool(smoke),
        "model": dict(config.get("model") or {}),
        "sparsity_grid": [float(x) for x in config.get("sparsity_grid", [])],
        "pruning": dict(config.get("pruning") or {}),
        "soft_dimensions": dims,
        "seed": int(config.get("seed", 42)),
        "recovery": config.get("recovery"),
        "e1_oneshot_checkpoint": config.get("e1_oneshot_checkpoint"),
    }


def soft_protocol_digest(config: Mapping[str, Any], *, smoke: bool) -> str:
    payload = json.dumps(
        soft_protocol_canonical(config, smoke=smoke),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_soft_checkpoint(
    path: Path,
    expected_digest: str,
    *,
    alternate_digests: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != SOFT_EVAL_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported soft_eval checkpoint version at {path}")
    digest = str(data.get("protocol_digest") or "")
    allowed = {expected_digest, *[str(x) for x in alternate_digests]}
    if digest not in allowed:
        raise ValueError(
            f"soft_eval checkpoint digest mismatch at {path}: "
            f"got {digest[:12]}… expected one of {[d[:12] for d in allowed]}"
        )
    return data


def save_soft_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, dict(payload))
    tmp.replace(path)


def build_soft_checkpoint_payload(
    *,
    protocol_digest: str,
    smoke: bool,
    sparsity_grid: Sequence[float],
    soft_dims: Sequence[str],
    completed_cells: Sequence[str],
    scores: Mapping[str, Mapping[str, Optional[float]]],
    started_at: str,
    elapsed_sec: float,
    status: str = "in_progress",
    scores_extra: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "version": SOFT_EVAL_CHECKPOINT_VERSION,
        "status": status,
        "protocol_digest": protocol_digest,
        "smoke": bool(smoke),
        "sparsity_grid": [float(x) for x in sparsity_grid],
        "soft_dims": list(soft_dims),
        "completed_cells": list(completed_cells),
        "scores": {str(k): dict(v) for k, v in scores.items()},
        "started_at": started_at,
        "updated_at": now,
        "elapsed_sec": float(elapsed_sec),
    }
    if scores_extra is not None:
        payload["scores_extra"] = {
            str(lab): {str(dim): dict(metrics) for dim, metrics in row.items()}
            for lab, row in scores_extra.items()
        }
    return payload


def cell_is_completed(completed_cells: Sequence[str], sparsity: float, dim: str) -> bool:
    return soft_cell_key(sparsity, dim) in set(completed_cells)


def resolve_soft_dim_specs(config: Mapping[str, Any], *, smoke: bool) -> Dict[str, Dict[str, Any]]:
    """Resolve soft dims: YAML lists which dims run; each merges over SOFT_DIM_SPECS defaults."""
    raw = config.get("soft_dimensions")
    if raw is None:
        specs = {k: dict(v) for k, v in SOFT_DIM_SPECS.items()}
    else:
        specs = {}
        for name, override in dict(raw).items():
            base = dict(SOFT_DIM_SPECS.get(name) or {})
            ov = dict(override)
            if "gen_kwargs" in ov and isinstance(ov["gen_kwargs"], Mapping):
                gk = dict(base.get("gen_kwargs") or {})
                gk.update(dict(ov["gen_kwargs"]))
                ov["gen_kwargs"] = gk
            base.update(ov)
            specs[name] = base
    if smoke:
        for name, spec in specs.items():
            if str(spec.get("kind", "")).startswith("glue_"):
                spec["max_samples"] = min(int(spec.get("max_samples", 256)), 8)
            if "scan_limit" in spec:
                spec["scan_limit"] = min(int(spec["scan_limit"]), 2)
    ordered: Dict[str, Dict[str, Any]] = {}
    for name in (*SOFT_DIM_ORDER, *SOFT6_DIM_ORDER, *SOFT_LITE_DIM_ORDER):
        if name in specs:
            ordered[name] = specs[name]
    for name, spec in specs.items():
        if name not in ordered:
            ordered[name] = spec
    return ordered


def apply_easy_capability_overrides(
    base_config: Mapping[str, Any],
    soft_dim: str,
    spec: Mapping[str, Any],
    *,
    smoke: bool,
) -> Dict[str, Any]:
    """Build a capability config that evaluates the base dim with easy/soft/lite overrides."""
    if spec.get("kind") != "capability":
        raise ValueError(f"{soft_dim} is not a capability soft dim")
    base_dim = str(spec["base_dim"])
    cfg = json.loads(json.dumps(dict(base_config)))
    ev = dict(cfg.get("evaluation") or {})
    cap = dict(ev.get("capability") or {})
    caps_limits = dict(cap.get("scan_limits") or {})
    caps_limits[base_dim] = int(spec.get("scan_limit", caps_limits.get(base_dim, 32)))
    cap["scan_limits"] = caps_limits
    cap["only_dimensions"] = [base_dim]
    if "max_length" in spec:
        cap["max_length"] = int(spec["max_length"])
    if "task_include_path" in spec:
        raw_path = spec["task_include_path"]
        repo_root = Path(__file__).resolve().parents[2]
        if isinstance(raw_path, (list, tuple)):
            resolved_paths = []
            for p in raw_path:
                pp = Path(str(p))
                resolved_paths.append(str(pp if pp.is_absolute() else (repo_root / pp)))
            cap["task_include_path"] = resolved_paths
        else:
            pp = Path(str(raw_path))
            cap["task_include_path"] = str(pp if pp.is_absolute() else (repo_root / pp))
    dims = dict(cap.get("dimensions") or {})
    dim_over = dict(dims.get(base_dim) or {})
    if "task" in spec:
        dim_over["task"] = spec["task"]
    if "num_fewshot" in spec:
        dim_over["num_fewshot"] = int(spec["num_fewshot"])
    if "apply_chat_template" in spec:
        dim_over["apply_chat_template"] = bool(spec["apply_chat_template"])
    if "confirm_run_unsafe_code" in spec:
        dim_over["confirm_run_unsafe_code"] = bool(spec["confirm_run_unsafe_code"])
    if "gen_kwargs" in spec:
        gk = dict(dim_over.get("gen_kwargs") or {})
        gk.update(dict(spec["gen_kwargs"]))
        dim_over["gen_kwargs"] = gk
    if "metric_candidates" in spec:
        dim_over["metric_candidates"] = list(spec["metric_candidates"])
    dims[base_dim] = dim_over
    cap["dimensions"] = dims
    if smoke:
        cap["limit_override"] = 2
    ev["capability"] = cap
    cfg["evaluation"] = ev
    return cfg


def sparsity_label(sp: float) -> str:
    if abs(sp) < 1e-12:
        return "dense"
    return f"{sp:.2f}"


def empty_scores_table(
    sparsity_grid: Sequence[float],
    soft_dims: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for sp in sparsity_grid:
        out[sparsity_label(sp)] = {d: None for d in soft_dims}
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic() -> float:
    return time.monotonic()


__all__ = [
    "SOFT6_DIM_ORDER",
    "SOFT_DIM_ORDER",
    "SOFT_DIM_SPECS",
    "SOFT_EVAL_CHECKPOINT_VERSION",
    "SOFT_LITE_DIM_ORDER",
    "apply_easy_capability_overrides",
    "build_soft_checkpoint_payload",
    "cell_is_completed",
    "empty_scores_table",
    "load_soft_checkpoint",
    "monotonic",
    "resolve_soft_dim_specs",
    "save_soft_checkpoint",
    "soft_cell_key",
    "soft_eval_checkpoint_filename",
    "soft_protocol_digest",
    "sparsity_label",
    "utc_now",
]
