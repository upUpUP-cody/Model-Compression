"""PDF capability vector P(M)=[PPL, Math, Knowledge, Reasoning, Instruction, Code].

Uses lm-evaluation-harness for community-comparable metrics.
E0 and E1+ share the same standard small-sample protocol (mode=scan).
Modes:
  - scan: frozen small limits + seed (default; E0/E1+)
  - e0_anchor: full splits (internal only; not used in E0 YAML)
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Vector order matches experiment-plan PDF E0.
DIM_ORDER: Tuple[str, ...] = (
    "PPL",
    "Math",
    "Knowledge",
    "Reasoning",
    "Instruction",
    "Code",
)

# Default frozen scan caps (E1+); indices/seed fixed via harness seed.
DEFAULT_SCAN_LIMITS: Dict[str, int] = {
    "PPL": 4,
    "Math": 64,
    "Knowledge": 128,
    "Reasoning": 64,
    "Instruction": 64,
    "Code": 32,
}

# Per-dimension harness task + extraction + fewshot/chat defaults.
_DIM_SPECS: Dict[str, Dict[str, Any]] = {
    "PPL": {
        "task": "wikitext",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "confirm_run_unsafe_code": False,
        # Prefer word_perplexity; harness may key as word_perplexity,none
        "metric_candidates": (
            "word_perplexity,none",
            "word_perplexity",
            "byte_perplexity,none",
            "byte_perplexity",
        ),
    },
    "Math": {
        "task": "gsm8k",
        "num_fewshot": 5,
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
        "metric_candidates": (
            "exact_match,flexible-extract",
            "exact_match,strict-match",
            "exact_match",
            "acc",
        ),
    },
    "Knowledge": {
        "task": "mmlu",
        "num_fewshot": 5,
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
        "metric_candidates": (
            "acc,none",
            "acc",
            "acc_norm,none",
            "acc_norm",
        ),
    },
    "Reasoning": {
        "task": "bbh",
        "num_fewshot": 3,
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
        # Cap generation length (lm_eval default is 2048); calibrated vs E1 dense.
        "gen_kwargs": {
            "max_gen_toks": 1024,
            "do_sample": False,
        },
        "metric_candidates": (
            "exact_match,get-answer",
            "acc_norm,none",
            "acc,none",
            "acc_norm",
            "acc",
            "exact_match,none",
            "exact_match",
        ),
    },
    "Instruction": {
        "task": "ifeval",
        "num_fewshot": 0,
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
        "metric_candidates": (
            "prompt_level_strict_acc,none",
            "prompt_level_strict_acc",
            "inst_level_strict_acc,none",
            "inst_level_strict_acc",
            "prompt_level_loose_acc,none",
        ),
    },
    "Code": {
        # Official HumanEval is code completion, not chat.
        "task": "humaneval",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "confirm_run_unsafe_code": True,
        "gen_kwargs": {
            "until": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
            "max_gen_toks": 512,
            "do_sample": False,
        },
        "metric_candidates": (
            "pass@1,create_test",
            "pass@1",
            "pass@1,none",
        ),
    },
}


def _as_mapping(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(config or {})


def _parse_dim_list(raw: Any, *, field_name: str) -> Tuple[str, ...]:
    """Validate a dim-name list is a subset of DIM_ORDER; preserve DIM_ORDER order."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        items = list(raw)
    else:
        raise ValueError(f"{field_name} must be a list of dim names, got {type(raw)!r}")
    unknown = [d for d in items if d not in DIM_ORDER]
    if unknown:
        raise ValueError(f"{field_name} not in DIM_ORDER: {unknown}")
    want = set(items)
    return tuple(d for d in DIM_ORDER if d in want)


def _parse_skip_dimensions(raw: Any) -> Tuple[str, ...]:
    return _parse_dim_list(raw, field_name="skip_dimensions")


def _parse_only_dimensions(raw: Any) -> Tuple[str, ...]:
    return _parse_dim_list(raw, field_name="only_dimensions")


def resolve_capability_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge YAML evaluation.capability with defaults."""
    root = _as_mapping(config)
    eval_cfg = _as_mapping(root.get("evaluation"))
    cap = _as_mapping(eval_cfg.get("capability"))
    mode = str(cap.get("mode", "scan"))
    if mode not in ("e0_anchor", "scan"):
        raise ValueError(f"capability.mode must be e0_anchor|scan, got {mode!r}")

    dims = _as_mapping(cap.get("dimensions"))
    resolved_dims: Dict[str, Dict[str, Any]] = {}
    for dim in DIM_ORDER:
        base = dict(_DIM_SPECS[dim])
        # Deep-copy nested gen_kwargs so per-dim overrides do not mutate defaults.
        if "gen_kwargs" in base and isinstance(base["gen_kwargs"], dict):
            base["gen_kwargs"] = dict(base["gen_kwargs"])
        override = _as_mapping(dims.get(dim))
        if "gen_kwargs" in override and isinstance(override["gen_kwargs"], Mapping):
            merged_gk = dict(base.get("gen_kwargs") or {})
            merged_gk.update(dict(override["gen_kwargs"]))
            override = dict(override)
            override["gen_kwargs"] = merged_gk
        base.update(override)
        resolved_dims[dim] = base

    scan_limits = dict(DEFAULT_SCAN_LIMITS)
    scan_limits.update(_as_mapping(cap.get("scan_limits")))
    skip_dimensions = _parse_skip_dimensions(cap.get("skip_dimensions"))
    only_dimensions = _parse_only_dimensions(cap.get("only_dimensions"))
    if only_dimensions and skip_dimensions:
        # only_dimensions wins for membership; still honor explicit skips inside only-set.
        only_set = set(only_dimensions)
        skip_dimensions = tuple(d for d in skip_dimensions if d in only_set)

    return {
        "mode": mode,
        "dimensions": resolved_dims,
        "scan_limits": scan_limits,
        "skip_dimensions": skip_dimensions,
        "only_dimensions": only_dimensions,
        "seed": int(cap.get("seed", root.get("seed", 42))),
        "batch_size": cap.get(
            "batch_size",
            _as_mapping(root.get("hardware")).get("batch_size", 1),
        ),
        "device": str(
            cap.get(
                "device",
                _as_mapping(root.get("hardware")).get("device", "cuda:0"),
            )
        ),
        "dtype": str(
            cap.get(
                "dtype",
                _as_mapping(root.get("model")).get("torch_dtype", "float16"),
            )
        ),
        "limit_override": cap.get("limit_override"),  # smoke: int applied to all dims
        "log_samples": bool(cap.get("log_samples", False)),
        "bootstrap_iters": int(cap.get("bootstrap_iters", 0)),
    }


def _pick_metric(task_result: Mapping[str, Any], candidates: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    skip_keys = {"alias", "name", "sample_len", "sample_count"}
    for key in candidates:
        if key in task_result and task_result[key] is not None:
            try:
                return float(task_result[key]), key
            except (TypeError, ValueError):
                continue
    # Fallback: first numeric metric-like field
    for key, value in task_result.items():
        if key in skip_keys or key.endswith("_stderr") or key.endswith(",stderr"):
            continue
        if isinstance(value, (int, float)):
            return float(value), key
    return None, None


def _extract_dim_score(
    results: Mapping[str, Any],
    task_name: str,
    candidates: Sequence[str],
) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
    """Pull score for a task or group name from harness results."""
    table = results.get("results") or {}
    if task_name in table:
        score, key = _pick_metric(table[task_name], candidates)
        return score, key, dict(table[task_name])

    # Group aggregates sometimes nested; average leaf tasks with matching prefix
    prefix = task_name + "_"
    leaves = {k: v for k, v in table.items() if k == task_name or k.startswith(prefix)}
    if not leaves:
        # bbh / mmlu group key variants
        leaves = {k: v for k, v in table.items() if task_name in k}
    scores: List[float] = []
    used_key: Optional[str] = None
    for _name, payload in leaves.items():
        if not isinstance(payload, Mapping):
            continue
        score, key = _pick_metric(payload, candidates)
        if score is None:
            continue
        scores.append(score)
        used_key = used_key or key
    if scores:
        return float(sum(scores) / len(scores)), used_key, {"leaf_count": len(scores), "leaves": list(leaves)}
    return None, None, {}


def _build_lm(
    model_path: str,
    *,
    device: str,
    dtype: str,
    batch_size: Any,
    model: Any = None,
    tokenizer: Any = None,
):
    from lm_eval.models.huggingface import HFLM

    if model is not None:
        return HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            trust_remote_code=True,
        )
    return HFLM(
        pretrained=model_path,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        trust_remote_code=True,
    )


def eval_capability_vector(
    model_path: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    mode: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Run six-dimensional capability eval via lm_eval.

    Returns dict with vector, per-dim details, mode, and raw harness slices.
    """
    from lm_eval import simple_evaluate

    resolved = resolve_capability_config(config)
    if mode is not None:
        resolved["mode"] = mode
    if resolved["mode"] not in ("e0_anchor", "scan"):
        raise ValueError(f"invalid mode {resolved['mode']!r}")

    seed = int(resolved["seed"])
    limit_override = resolved.get("limit_override")
    vector: Dict[str, Optional[float]] = {d: None for d in DIM_ORDER}
    details: Dict[str, Any] = {}
    raw_by_dim: Dict[str, Any] = {}

    lm = _build_lm(
        model_path,
        device=resolved["device"],
        dtype=resolved["dtype"],
        batch_size=resolved["batch_size"],
        model=model,
        tokenizer=tokenizer,
    )

    skip_set = set(resolved.get("skip_dimensions") or ())
    only = tuple(resolved.get("only_dimensions") or ())
    only_set = set(only) if only else None
    try:
        for dim in DIM_ORDER:
            if only_set is not None and dim not in only_set:
                vector[dim] = None
                details[dim] = {
                    "skipped": True,
                    "reason": "only_dimensions",
                    "task": resolved["dimensions"][dim].get("task"),
                }
                print(f"[INFO] capability dim={dim} skipped (only_dimensions)")
                continue
            if dim in skip_set:
                vector[dim] = None
                details[dim] = {"skipped": True, "task": resolved["dimensions"][dim].get("task")}
                print(f"[INFO] capability dim={dim} skipped (skip_dimensions)")
                continue

            spec = resolved["dimensions"][dim]
            task = str(spec["task"])
            if limit_override is not None:
                limit: Optional[float] = float(limit_override)
            elif resolved["mode"] == "scan":
                limit = float(resolved["scan_limits"][dim])
            else:
                limit = None  # full / official

            gen_kwargs = spec.get("gen_kwargs")
            print(
                f"[INFO] capability dim={dim} task={task} mode={resolved['mode']} "
                f"limit={limit} fewshot={spec.get('num_fewshot')} "
                f"chat={spec.get('apply_chat_template')} gen_kwargs={bool(gen_kwargs)}"
            )
            eval_kwargs: Dict[str, Any] = dict(
                model=lm,
                tasks=[task],
                num_fewshot=spec.get("num_fewshot"),
                batch_size=resolved["batch_size"],
                device=resolved["device"],
                limit=limit,
                bootstrap_iters=int(resolved["bootstrap_iters"]),
                log_samples=bool(resolved["log_samples"]),
                apply_chat_template=bool(spec.get("apply_chat_template", False)),
                confirm_run_unsafe_code=bool(spec.get("confirm_run_unsafe_code", False)),
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed,
            )
            if gen_kwargs:
                eval_kwargs["gen_kwargs"] = dict(gen_kwargs)
            out = simple_evaluate(**eval_kwargs)
            if out is None:
                details[dim] = {"task": task, "error": "simple_evaluate returned None"}
                continue

            score, metric_key, task_payload = _extract_dim_score(
                out,
                task,
                tuple(spec.get("metric_candidates") or ()),
            )
            vector[dim] = score
            details[dim] = {
                "task": task,
                "metric_key": metric_key,
                "score": score,
                "num_fewshot": spec.get("num_fewshot"),
                "apply_chat_template": bool(spec.get("apply_chat_template", False)),
                "limit": limit,
                "gen_kwargs": dict(gen_kwargs) if gen_kwargs else None,
                "task_result": task_payload,
            }
            # Keep compact raw: results table only
            raw_by_dim[dim] = {"results": out.get("results"), "n-samples": out.get("n-samples")}
    finally:
        del lm

    return {
        "mode": resolved["mode"],
        "seed": seed,
        "model_path": model_path,
        "vector": vector,
        "vector_list": [vector[d] for d in DIM_ORDER],
        "dim_order": list(DIM_ORDER),
        "skip_dimensions": list(resolved.get("skip_dimensions") or ()),
        "only_dimensions": list(resolved.get("only_dimensions") or ()),
        "details": details,
        "raw": raw_by_dim,
        "protocol": {
            "PPL": "wikitext word_perplexity (WikiText-2; lower better)",
            "Math": "gsm8k exact_match flexible-extract",
            "Knowledge": "mmlu acc 5-shot",
            "Reasoning": "bbh acc 3-shot",
            "Instruction": "ifeval prompt/inst strict",
            "Code": "humaneval pass@1 (completion; no chat template)",
            "mode": resolved["mode"],
            "seed": seed,
            "scan_limits": dict(resolved["scan_limits"]),
            "skip_dimensions": list(resolved.get("skip_dimensions") or ()),
            "only_dimensions": list(resolved.get("only_dimensions") or ()),
        },
    }
