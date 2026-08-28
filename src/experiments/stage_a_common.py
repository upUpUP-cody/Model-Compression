"""Shared helpers for Stage A E0–E3 smoke (PDF order; proxy_1.5B allowed)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml

from src.evaluation.capability import DIM_ORDER
from src.experiments.qwen_k5_comparison import (
    count_params,    magnitude_importance_mlp,
    mlp_uniform_ratios,
    wanda_importance_mlp,
)
from src.pruning.pruning_backend import resolve_pruning_backend
from src.recovery.qwen_lm_recovery import evaluate_lm_loss
from src.utils.device import resolve_device
from src.utils.glue_protocol import assert_test_not_in_selection_path
from src.utils.qwen_glue_eval import load_qwen_for_eval, prepare_glue_from_config
from src.utils.qwen_glue_train_data import build_glue_lm_loaders
from src.utils.qwen_squad_eval import write_json


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mirror_docs_report(experiment_id: str, slug: str, report_text: str) -> Path:
    docs = Path("docs/results")
    ensure_dir(docs)
    out = docs / f"{experiment_id}_{slug}.md"
    write_report(out, report_text)
    return out


def gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def prepare_sst2_loaders(config: Mapping[str, Any], tokenizer, device: str):
    """Build tiny SST-2 LM loaders for Stage A smoke metrics."""
    cfg = dict(config)
    ds = dict(cfg.get("dataset") or {})
    ds.setdefault("name", "nyu-mll/glue")
    ds.setdefault("task", "sst2")
    ds.setdefault("cache_dir", "/mnt/data/datasets/glue")
    ds.setdefault("validation_fraction", 0.1)
    ds.setdefault("split_seed", int(cfg.get("seed", 42)))
    ds.setdefault("max_seq_len", int(cfg.get("evaluation", {}).get("max_seq_len", 256)))
    ds.setdefault("train_max_samples", int(cfg.get("evaluation", {}).get("train_max_samples", 128)))
    ds.setdefault("validation_max_samples", int(cfg.get("evaluation", {}).get("validation_max_samples", 64)))
    cfg["dataset"] = ds
    splits = prepare_glue_from_config(cfg)
    selection = splits.selection_splits()
    assert_test_not_in_selection_path(list(selection.keys()))
    train_loader, val_loader = build_glue_lm_loaders(
        selection,
        tokenizer,
        task="sst2",
        batch_size=int(cfg.get("hardware", {}).get("batch_size", 1)),
        max_seq_len=int(ds["max_seq_len"]),
        train_max_samples=ds.get("train_max_samples"),
        validation_max_samples=ds.get("validation_max_samples"),
    )
    return train_loader, val_loader, splits.metadata()


def load_model(config: Mapping[str, Any]):
    device = resolve_device(str(config.get("hardware", {}).get("device", "cuda")))
    model, tokenizer = load_qwen_for_eval(
        str(config["model"]["path"]),
        device=device,
        torch_dtype=str(config["model"].get("torch_dtype", "float16")),
    )
    return model, tokenizer, device


def prepare_wanda_calibration_loader(config: Mapping[str, Any], tokenizer, device: str):
    """Build SST-2 LM train loader for Wanda calibration (separate from capability eval)."""
    cfg = dict(config)
    pruning = dict(cfg.get("pruning") or {})
    cal = dict(pruning.get("calibration") or {})
    ds = dict(cfg.get("dataset") or {})
    ds.setdefault("name", "nyu-mll/glue")
    ds.setdefault("task", "sst2")
    ds.setdefault("cache_dir", "/mnt/data/datasets/glue")
    ds.setdefault("validation_fraction", 0.1)
    ds.setdefault("split_seed", int(cfg.get("seed", 42)))
    ds.setdefault("max_seq_len", int(cal.get("max_seq_len", 256)))
    ds.setdefault("train_max_samples", int(cal.get("train_max_samples", 128)))
    ds.setdefault("validation_max_samples", 0)
    cfg["dataset"] = ds
    splits = prepare_glue_from_config(cfg)
    selection = splits.selection_splits()
    assert_test_not_in_selection_path(list(selection.keys()))
    train_loader, _val_loader = build_glue_lm_loaders(
        selection,
        tokenizer,
        task="sst2",
        batch_size=int(cal.get("batch_size", cfg.get("hardware", {}).get("batch_size", 1))),
        max_seq_len=int(ds["max_seq_len"]),
        train_max_samples=ds.get("train_max_samples"),
        validation_max_samples=ds.get("validation_max_samples"),
    )
    return train_loader


def prune_mlp_ratio(model: nn.Module, prune_ratio: float) -> nn.Module:
    """One-shot uniform MLP intermediate prune (magnitude fallback; prefer prune_mlp_wanda)."""
    ratios = mlp_uniform_ratios(model, float(prune_ratio))
    cpu = model.cpu()
    pruned = resolve_pruning_backend(cpu, "qwen").create_pruned_model(ratios)
    return pruned


def prune_mlp_by_importance(
    model: nn.Module,
    prune_ratio: float,
    *,
    max_layers: int = 4,
) -> nn.Module:
    importance = magnitude_importance_mlp(model, max_layers=max_layers)
    keep_indices: Dict[str, List[int]] = {}
    for name, scores in importance.items():
        n = int(scores.numel())
        keep = max(1, int(round(n * (1.0 - float(prune_ratio)))))
        ranked = sorted(range(n), key=lambda i: (-float(scores[i]), i))
        keep_indices[name] = sorted(ranked[:keep])
    # Fill remaining MLP layers with full keep (backend may require all or subset)
    cpu = model.cpu()
    backend = resolve_pruning_backend(cpu, "qwen")
    return backend.create_pruned_model_by_indices(keep_indices)


def prune_mlp_wanda(
    model: nn.Module,
    prune_ratio: float,
    dataloader,
    device: str,
    *,
    num_batches: int = 4,
    max_layers: int | None = None,
) -> nn.Module:
    """One-shot Wanda MLP intermediate prune using calibration activations."""
    importance = wanda_importance_mlp(
        model,
        dataloader,
        device,
        num_batches=int(num_batches),
        max_layers=max_layers,
    )
    keep_indices: Dict[str, List[int]] = {}
    ratio = float(prune_ratio)
    for name, scores in importance.items():
        size = int(scores.numel())
        keep_count = max(1, int(round(size * (1.0 - ratio))))
        ranked = sorted(range(size), key=lambda i: (-float(scores[i]), i))
        keep_indices[name] = sorted(ranked[:keep_count])
    cpu = model.cpu()
    backend = resolve_pruning_backend(cpu, "qwen")
    return backend.create_pruned_model_by_indices(keep_indices)


def resolve_stage_a_prune_fn(config: Mapping[str, Any]):
    pruning = dict(config.get("pruning") or {})
    method = str(pruning.get("method", "wanda")).lower()
    if method == "wanda":
        return "wanda"
    if method in ("magnitude", "ratio", "uniform"):
        return "magnitude"
    raise ValueError(f"unsupported pruning.method {method!r}; use wanda|magnitude")


def stage_a_prune_mlp(
    model: nn.Module,
    prune_ratio: float,
    config: Mapping[str, Any],
    *,
    calib_loader,
    device: str,
) -> nn.Module:
    """Dispatch Stage A one-shot MLP prune (default Wanda)."""
    pruning = dict(config.get("pruning") or {})
    method = resolve_stage_a_prune_fn(config)
    if method == "wanda":
        return prune_mlp_wanda(
            model,
            prune_ratio,
            calib_loader,
            device,
            num_batches=int(pruning.get("wanda_batches", 4)),
        )
    return prune_mlp_ratio(model, prune_ratio)


def eval_proxy(model: nn.Module, loader, device: str) -> Dict[str, float]:
    out = evaluate_lm_loss(model, loader, device=device)
    loss = float(out.get("loss", float("nan")))
    ppl = float(torch.exp(torch.tensor(loss)).item()) if loss == loss else float("nan")
    return {
        "loss": loss,
        "ppl": ppl,
        "proxy_accuracy": float(out.get("accuracy", float("nan"))),
    }


def compute_capability_deltas(
    dense_vector: Mapping[str, Optional[float]],
    pruned_vector: Mapping[str, Optional[float]],
    *,
    dim_order: Sequence[str] = ("PPL", "Math", "Knowledge", "Reasoning", "Instruction", "Code"),
) -> Dict[str, Optional[float]]:
    """Per-dimension delta: pruned - dense. PPL increase = degradation (positive delta)."""
    out: Dict[str, Optional[float]] = {}
    for dim in dim_order:
        d = dense_vector.get(dim)
        p = pruned_vector.get(dim)
        if d is None or p is None:
            out[dim] = None
            continue
        out[dim] = float(p) - float(d)
    return out


def detect_capability_cliffs(
    rows: Sequence[Mapping[str, Any]],
    *,
    dim_order: Sequence[str] = ("PPL", "Math", "Knowledge", "Reasoning", "Instruction", "Code"),
    min_sparsity_jump: float = 0.10,
) -> Dict[str, Any]:
    """Heuristic cliff: largest step-to-step delta increase per dimension across sparsity grid."""
    per_dim: Dict[str, Dict[str, Any]] = {}
    for dim in dim_order:
        series = [(float(r["sparsity"]), r.get("delta", {}).get(dim)) for r in rows]
        series = [(s, v) for s, v in series if v is not None]
        cliff_sparsity = None
        max_accel = 0.0
        for i in range(1, len(series)):
            s_prev, v_prev = series[i - 1]
            s_cur, v_cur = series[i]
            step = s_cur - s_prev
            if step < min_sparsity_jump - 1e-9:
                continue
            accel = float(v_cur) - float(v_prev)
            # PPL: positive delta = worse; acc metrics: negative delta = worse
            harm = accel if dim == "PPL" else -accel
            if harm > max_accel:
                max_accel = harm
                cliff_sparsity = s_cur
        per_dim[dim] = {
            "cliff_sparsity": cliff_sparsity,
            "max_harm_step": max_accel if cliff_sparsity is not None else None,
            "has_cliff_signal": cliff_sparsity is not None and max_accel > 0.05,
        }
    cliffs = [d for d in dim_order if per_dim[d]["has_cliff_signal"]]
    return {
        "per_dimension": per_dim,
        "capability_specific": len(set(cliffs)) >= 2,
        "cliff_dimensions": cliffs,
    }


E1_CHECKPOINT_VERSION = 1


def e1_checkpoint_filename(*, smoke: bool = False) -> str:
    return "e1_checkpoint_smoke.json" if smoke else "e1_checkpoint.json"


def _strip_perf_knobs_from_evaluation(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Remove resume-safe knobs so batch_size / skip|only_dimensions do not invalidate checkpoints."""
    out = dict(evaluation)
    cap = dict(out.get("capability") or {})
    cap.pop("batch_size", None)
    cap.pop("skip_dimensions", None)
    cap.pop("only_dimensions", None)
    if cap:
        out["capability"] = cap
    elif "capability" in out:
        out.pop("capability")
    return out


def e1_canonical_run_config(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    include_perf_knobs: bool = False,
) -> Dict[str, Any]:
    """Fields that must match for E1 checkpoint resume."""
    run_config = dict(config)
    if smoke:
        run_config.setdefault("evaluation", {}).setdefault("capability", {})["limit_override"] = 2
    evaluation = dict(run_config.get("evaluation") or {})
    if not include_perf_knobs:
        evaluation = _strip_perf_knobs_from_evaluation(evaluation)
    return {
        "smoke": bool(smoke),
        "model": dict(config.get("model") or {}),
        "sparsity_grid": [float(x) for x in config.get("sparsity_grid", [])],
        "pruning": dict(config.get("pruning") or {}),
        "evaluation": evaluation,
        "seed": int(config.get("seed", 42)),
        "recovery": config.get("recovery"),
    }


def e1_config_digest(config: Mapping[str, Any], *, smoke: bool) -> str:
    payload = json.dumps(
        e1_canonical_run_config(config, smoke=smoke, include_perf_knobs=False),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def e1_config_digest_legacy(config: Mapping[str, Any], *, smoke: bool) -> str:
    """Pre-dual-GPU digest (batch_size included); for loading older checkpoints."""
    payload = json.dumps(
        e1_canonical_run_config(config, smoke=smoke, include_perf_knobs=True),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def e1_checkpoint_digest_candidates(config: Mapping[str, Any], *, smoke: bool) -> Tuple[str, ...]:
    """Digest values that may appear in on-disk checkpoints for this protocol config."""
    candidates = [
        e1_config_digest(config, smoke=smoke),
        e1_config_digest_legacy(config, smoke=smoke),
    ]
    # Checkpoints from the initial batch=1 formal run before perf knobs were excluded.
    cap_bs = (
        (config.get("evaluation") or {}).get("capability") or {}
    ).get("batch_size")
    if cap_bs != 1:
        cfg_b1 = dict(config)
        eval_b1 = dict(cfg_b1.get("evaluation") or {})
        cap_b1 = dict(eval_b1.get("capability") or {})
        cap_b1["batch_size"] = 1
        eval_b1["capability"] = cap_b1
        cfg_b1["evaluation"] = eval_b1
        legacy_b1 = e1_config_digest_legacy(cfg_b1, smoke=smoke)
        if legacy_b1 not in candidates:
            candidates.append(legacy_b1)
    return tuple(candidates)


def dense_capability_complete(
    dense_cap: Mapping[str, Any],
    *,
    dim_order: Sequence[str] = DIM_ORDER,
    skip_dimensions: Sequence[str] = (),
) -> bool:
    """True if all required dims have scores. Skipped dims are not required."""
    vector = dense_cap.get("vector") or {}
    skip = set(skip_dimensions)
    required = [d for d in dim_order if d not in skip]
    return all(vector.get(dim) is not None for dim in required)


def e1_required_dimensions(
    *,
    dim_order: Sequence[str] = DIM_ORDER,
    skip_dimensions: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Dimensions that must be scored for a sparsity row to be complete."""
    skip = set(skip_dimensions)
    return tuple(d for d in dim_order if d not in skip)


def e1_partial_dims_complete(
    partial: Optional[Mapping[str, Any]],
    *,
    dim_order: Sequence[str] = DIM_ORDER,
    skip_dimensions: Sequence[str] = (),
) -> bool:
    """True if partial has scores for all required dimensions."""
    if not partial:
        return False
    vector = partial.get("vector") or {}
    required = e1_required_dimensions(dim_order=dim_order, skip_dimensions=skip_dimensions)
    return all(vector.get(dim) is not None for dim in required)


def new_e1_partial(
    sparsity: float,
    *,
    dim_order: Sequence[str] = DIM_ORDER,
    params: Any = None,
) -> Dict[str, Any]:
    """Empty in-progress sparsity row for per-dimension checkpointing."""
    return {
        "sparsity": float(sparsity),
        "vector": {d: None for d in dim_order},
        "details": {},
        "raw": {},
        "completed_dimensions": [],
        "params": params,
    }


def merge_capability_dim_result(
    partial: Mapping[str, Any],
    dim: str,
    cap_slice: Mapping[str, Any],
    *,
    dim_order: Sequence[str] = DIM_ORDER,
) -> Dict[str, Any]:
    """Merge one-dim eval_capability_vector output into a partial row."""
    if dim not in dim_order:
        raise ValueError(f"unknown capability dim: {dim}")
    out = {
        "sparsity": float(partial["sparsity"]),
        "vector": {d: (partial.get("vector") or {}).get(d) for d in dim_order},
        "details": dict(partial.get("details") or {}),
        "raw": dict(partial.get("raw") or {}),
        "completed_dimensions": [],
        "params": partial.get("params"),
    }
    out["vector"][dim] = (cap_slice.get("vector") or {}).get(dim)
    details = cap_slice.get("details") or {}
    if dim in details:
        out["details"][dim] = details[dim]
    raw = cap_slice.get("raw") or {}
    if dim in raw:
        out["raw"][dim] = raw[dim]
    prev_done = set(partial.get("completed_dimensions") or [])
    prev_done.add(dim)
    out["completed_dimensions"] = [d for d in dim_order if d in prev_done]
    return out


def finalize_e1_partial_capability(
    partial: Mapping[str, Any],
    *,
    model_path: str,
    mode: str = "scan",
    seed: int = 42,
    dim_order: Sequence[str] = DIM_ORDER,
    skip_dimensions: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build a full capability blob from a completed partial for curve storage."""
    vector = {d: (partial.get("vector") or {}).get(d) for d in dim_order}
    return {
        "mode": mode,
        "seed": int(seed),
        "model_path": model_path,
        "vector": vector,
        "vector_list": [vector[d] for d in dim_order],
        "dim_order": list(dim_order),
        "skip_dimensions": list(skip_dimensions),
        "only_dimensions": [],
        "details": dict(partial.get("details") or {}),
        "raw": dict(partial.get("raw") or {}),
    }


def sparsity_in_completed(sparsity: float, completed: Sequence[float], *, tol: float = 1e-6) -> bool:
    return any(abs(float(s) - float(sparsity)) <= tol for s in completed)


def load_e1_checkpoint(
    path: Path,
    expected_digest: str,
    *,
    alternate_digests: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    stored = str(data.get("config_digest"))
    allowed = {str(expected_digest), *(str(d) for d in alternate_digests)}
    if stored not in allowed:
        raise ValueError(
            f"E1 checkpoint config mismatch at {path}; use --fresh to restart "
            f"(checkpoint={stored}, expected one of {sorted(allowed)})"
        )
    if int(data.get("version", 0)) != E1_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E1 checkpoint version at {path}")
    return data


def import_e1_checkpoint_seed(
    path: Path,
    *,
    alternate_digests: Sequence[str] = (),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load dense + curve from a checkpoint for shard seeding (no strict digest)."""
    if not path.is_file():
        raise FileNotFoundError(f"E1 seed checkpoint missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if int(data.get("version", 0)) != E1_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E1 checkpoint version at {path}")
    stored = str(data.get("config_digest"))
    if alternate_digests and stored not in {str(d) for d in alternate_digests}:
        raise ValueError(
            f"E1 seed checkpoint digest mismatch at {path} "
            f"(checkpoint={stored}, expected one of {list(alternate_digests)})"
        )
    dense_cap = dict(data.get("dense_capability") or {})
    rows = [dict(r) for r in (data.get("curve") or [])]
    if not dense_capability_complete(dense_cap):
        raise ValueError(f"E1 seed checkpoint at {path} has incomplete dense_capability")
    return dense_cap, rows


def filter_e1_sparsity_grid(
    grid: Sequence[float],
    sparsities: Optional[Sequence[float]],
) -> List[float]:
    """Restrict run grid to an explicit subset (order preserved from grid)."""
    if not sparsities:
        return [float(x) for x in grid]
    want = {float(s) for s in sparsities}
    full = {float(x) for x in grid}
    unknown = sorted(want - full)
    if unknown:
        raise ValueError(f"--sparsities not in config sparsity_grid: {unknown}")
    return [float(x) for x in grid if float(x) in want]


def merge_e1_curves(
    *curve_sets: Sequence[Mapping[str, Any]],
    require_disjoint: bool = True,
) -> List[Dict[str, Any]]:
    """Merge curve rows from main + shard runs; sort by sparsity ascending."""
    by_sp: Dict[float, Dict[str, Any]] = {}
    for curves in curve_sets:
        for row in curves:
            sp = float(row["sparsity"])
            if require_disjoint and sp in by_sp:
                raise ValueError(f"duplicate sparsity {sp} in merge_e1_curves")
            by_sp[sp] = dict(row)
    return [by_sp[sp] for sp in sorted(by_sp)]


def merge_e1_main_with_shard(
    main_rows: Sequence[Mapping[str, Any]],
    shard_rows: Sequence[Mapping[str, Any]],
    *,
    overwrite_sparsities: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Merge shard rows into main: novel sparsities append; overwrite_sparsities replace."""
    by_sp: Dict[float, Dict[str, Any]] = {float(r["sparsity"]): dict(r) for r in main_rows}
    main_sps = set(by_sp)
    overwrite = {float(s) for s in (overwrite_sparsities or [])}
    for row in shard_rows:
        sp = float(row["sparsity"])
        if sp in overwrite:
            by_sp[sp] = dict(row)
        elif sp not in main_sps:
            by_sp[sp] = dict(row)
    return [by_sp[sp] for sp in sorted(by_sp)]


def save_e1_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)


def build_e1_checkpoint_payload(
    *,
    config_digest: str,
    smoke: bool,
    dense_capability: Mapping[str, Any],
    curve: Sequence[Mapping[str, Any]],
    grid: Sequence[float],
    started_at: float,
    elapsed_sec: float,
    status: str = "in_progress",
    partial: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    completed = [float(r["sparsity"]) for r in curve]
    pending = [float(s) for s in grid if not sparsity_in_completed(float(s), completed)]
    payload: Dict[str, Any] = {
        "version": E1_CHECKPOINT_VERSION,
        "status": status,
        "config_digest": config_digest,
        "smoke": bool(smoke),
        "dense_capability": dict(dense_capability),
        "curve": [dict(r) for r in curve],
        "completed_sparsities": completed,
        "pending_sparsities": pending,
        "started_at": float(started_at),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(elapsed_sec),
    }
    if partial is not None:
        payload["partial"] = dict(partial)
    return payload


# ---------------------------------------------------------------------------
# E2 Iterative vs One-shot helpers
# ---------------------------------------------------------------------------

E2_CHECKPOINT_VERSION = 1
E2_MAIN_DIMS: Tuple[str, ...] = ("PPL", "Math", "Knowledge", "Reasoning")
E2_SECONDARY_DIMS: Tuple[str, ...] = ("Instruction", "Code")


def e2_checkpoint_filename(*, smoke: bool = False) -> str:
    return "e2_checkpoint_smoke.json" if smoke else "e2_checkpoint.json"


def e2_cell_key(
    seed: int,
    target_sparsity: float,
    method: str,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "target_sparsity": float(target_sparsity),
        "method": str(method),
    }


def e2_cell_tuple(cell: Mapping[str, Any]) -> Tuple[int, float, str]:
    return (int(cell["seed"]), float(cell["target_sparsity"]), str(cell["method"]))


def e2_cells_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return e2_cell_tuple(a) == e2_cell_tuple(b)


def incremental_step_targets(
    target_sparsity: float,
    *,
    step: float = 0.05,
) -> List[float]:
    """Cumulative absolute sparsity targets: step, 2*step, ..., target."""
    target = float(target_sparsity)
    step = float(step)
    if step <= 0 or target <= 0:
        raise ValueError(f"invalid incremental step/target: step={step} target={target}")
    n = int(round(target / step))
    if abs(n * step - target) > 1e-9:
        raise ValueError(
            f"target_sparsity={target} must be an integer multiple of step={step}"
        )
    return [round(step * (i + 1), 10) for i in range(n)]


def remaining_relative_prune_ratio(prev_sparsity: float, next_sparsity: float) -> float:
    """Fraction of *current* channels to remove to go from prev→next absolute sparsity."""
    prev = float(prev_sparsity)
    nxt = float(next_sparsity)
    if nxt <= prev:
        raise ValueError(f"next_sparsity ({nxt}) must exceed prev ({prev})")
    remain = 1.0 - prev
    if remain <= 1e-12:
        raise ValueError("cannot prune further; prev_sparsity already ~1")
    return (nxt - prev) / remain


def incremental_prune_5pct(
    model: nn.Module,
    target_sparsity: float,
    config: Mapping[str, Any],
    *,
    calib_loader,
    device: str,
    step: Optional[float] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """PDF 5% incremental: cumulative absolute sparsity targets; no recovery.

    Each step prunes a remaining-relative ratio so final sparsity ≈ target
    (not fixed 0.05×N compound which undershoots).
    """
    step_sz = float(
        step
        if step is not None
        else config.get("incremental_step_sparsity", 0.05)
    )
    targets = incremental_step_targets(target_sparsity, step=step_sz)
    dense_params = count_params(model)
    current = model
    prev = 0.0
    stage_log: List[Dict[str, Any]] = []
    for cum in targets:
        ratio = remaining_relative_prune_ratio(prev, cum)
        current = stage_a_prune_mlp(
            current,
            ratio,
            config,
            calib_loader=calib_loader,
            device=device,
        ).to(device)
        params = count_params(current)
        actual = 1.0 - (float(params) / max(float(dense_params), 1.0))
        stage_log.append(
            {
                "cumulative_target": float(cum),
                "step_prune_ratio": float(ratio),
                "params": int(params),
                "actual_sparsity_vs_dense": float(actual),
            }
        )
        prev = cum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_params = count_params(current)
    meta = {
        "dense_params": int(dense_params),
        "final_params": int(final_params),
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": 1.0 - (float(final_params) / max(float(dense_params), 1.0)),
        "stages": stage_log,
        "incremental_step": float(step_sz),
    }
    return current, meta


def pruning_eval_protocol_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Shared protocol fields for E1↔E2 import validation (no sparsity grid)."""
    evaluation = _strip_perf_knobs_from_evaluation(dict(config.get("evaluation") or {}))
    return {
        "model": dict(config.get("model") or {}),
        "pruning": dict(config.get("pruning") or {}),
        "evaluation": evaluation,
        "recovery": config.get("recovery"),
    }


def pruning_eval_protocol_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        pruning_eval_protocol_payload(config),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def e2_canonical_run_config(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    evaluation = _strip_perf_knobs_from_evaluation(dict(config.get("evaluation") or {}))
    if smoke:
        cap = dict(evaluation.get("capability") or {})
        cap["limit_override"] = 2
        evaluation["capability"] = cap
    seed_list = [int(s) for s in (seeds if seeds is not None else config.get("seeds") or [config.get("seed", 42)])]
    return {
        "smoke": bool(smoke),
        "model": dict(config.get("model") or {}),
        "target_sparsities": [float(x) for x in config.get("target_sparsities", [])],
        "incremental_step_sparsity": float(config.get("incremental_step_sparsity", 0.05)),
        "pruning": dict(config.get("pruning") or {}),
        "evaluation": evaluation,
        "recovery": config.get("recovery"),
        "seeds": sorted(seed_list),
    }


def e2_config_digest(
    config: Mapping[str, Any],
    *,
    smoke: bool = False,
    seeds: Optional[Sequence[int]] = None,
) -> str:
    payload = json.dumps(
        e2_canonical_run_config(config, smoke=smoke, seeds=seeds),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def import_e1_oneshot_rows(
    checkpoint_path: Path,
    sparsities: Sequence[float],
    *,
    e2_config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Import dense + oneshot curve rows for given sparsities from E1 checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"E1 checkpoint missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if int(data.get("version", 0)) != E1_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E1 checkpoint version at {path}")

    # Protocol must match (model + pruning + capability); ignore E1 sparsity_grid.
    e2_proto = pruning_eval_protocol_digest(e2_config)
    # Recompute expected from E1-shaped view: same model/pruning/eval as E2.
    # Stored E1 digest includes sparsity_grid — do not compare raw digests.
    # Instead verify Code details carry new HumanEval protocol when present.
    dense_cap = dict(data.get("dense_capability") or {})
    if not dense_capability_complete(dense_cap):
        raise ValueError(f"E1 checkpoint at {path} has incomplete dense_capability")

    want = {float(s) for s in sparsities}
    by_sp: Dict[float, Dict[str, Any]] = {}
    for row in data.get("curve") or []:
        sp = float(row["sparsity"])
        if any(abs(sp - w) < 1e-9 for w in want):
            by_sp[sp] = dict(row)

    rows: List[Dict[str, Any]] = []
    for sp in sorted(want):
        match = None
        for k, v in by_sp.items():
            if abs(k - sp) < 1e-9:
                match = v
                break
        if match is None:
            raise ValueError(f"E1 checkpoint missing sparsity={sp} for E2 oneshot import")
        vector = dict(match.get("vector") or {})
        for dim in DIM_ORDER:
            if vector.get(dim) is None:
                raise ValueError(f"E1 row sparsity={sp} missing finite score for {dim}")
        details = dict((match.get("capability") or {}).get("details") or match.get("details") or {})
        code = details.get("Code") or {}
        if code.get("apply_chat_template") is not False:
            raise ValueError(
                f"E1 Code at sparsity={sp} missing apply_chat_template=false "
                "(HumanEval completion protocol)"
            )
        if not code.get("gen_kwargs"):
            raise ValueError(f"E1 Code at sparsity={sp} missing gen_kwargs")
        reasoning = details.get("Reasoning") or {}
        r_gk = reasoning.get("gen_kwargs") or {}
        if int(r_gk.get("max_gen_toks") or 0) != 1024:
            raise ValueError(
                f"E1 Reasoning at sparsity={sp} requires gen_kwargs.max_gen_toks=1024 "
                f"(got {r_gk.get('max_gen_toks')!r}); refuse mixed 2048/1024 protocol"
            )
        if r_gk.get("do_sample") is not False:
            raise ValueError(
                f"E1 Reasoning at sparsity={sp} requires gen_kwargs.do_sample=false "
                f"(got {r_gk.get('do_sample')!r})"
            )
        rows.append(
            {
                "sparsity": float(sp),
                "vector": vector,
                "delta": dict(match.get("delta") or {}),
                "details": details,
                "params": match.get("params"),
                "source": "e1_import",
                "protocol_digest": e2_proto,
            }
        )
    return dense_cap, rows


def dim_better_iterative(
    dim: str,
    oneshot_score: Optional[float],
    iterative_score: Optional[float],
) -> Optional[bool]:
    """True if iterative wins on dim; False if oneshot wins; None if tie/missing."""
    if oneshot_score is None or iterative_score is None:
        return None
    o = float(oneshot_score)
    i = float(iterative_score)
    if dim == "PPL":
        if i < o:
            return True
        if o < i:
            return False
        return None
    if i > o:
        return True
    if o > i:
        return False
    return None


def compare_iterative_vs_oneshot(
    oneshot_vector: Mapping[str, Optional[float]],
    iterative_vector: Mapping[str, Optional[float]],
    *,
    main_dims: Sequence[str] = E2_MAIN_DIMS,
    win_threshold: int = 3,
) -> Dict[str, Any]:
    """Per-dim winners + cell win if iterative better on >= win_threshold main dims."""
    winner_per_dim: Dict[str, str] = {}
    main_wins = 0
    for dim in DIM_ORDER:
        better = dim_better_iterative(dim, oneshot_vector.get(dim), iterative_vector.get(dim))
        if better is True:
            winner_per_dim[dim] = "iterative"
            if dim in main_dims:
                main_wins += 1
        elif better is False:
            winner_per_dim[dim] = "oneshot"
        else:
            winner_per_dim[dim] = "tie"
    cell_iterative_win = main_wins >= int(win_threshold)
    return {
        "winner_per_dim": winner_per_dim,
        "main_dim_iterative_wins": main_wins,
        "cell_iterative_win": cell_iterative_win,
        "win_threshold": int(win_threshold),
        "main_dims": list(main_dims),
    }


def gate_a_iterative_advantage(
    cell_results: Sequence[Mapping[str, Any]],
    *,
    min_wins: int = 7,
) -> Dict[str, Any]:
    """Gate A E2 half: iterative wins on >= min_wins of the (target, seed) cells."""
    wins = sum(1 for c in cell_results if c.get("cell_iterative_win"))
    total = len(cell_results)
    return {
        "iterative_cell_wins": int(wins),
        "total_cells": int(total),
        "min_wins": int(min_wins),
        "passed": total > 0 and wins >= int(min_wins),
    }


def new_e2_partial(
    seed: int,
    target_sparsity: float,
    method: str,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "target_sparsity": float(target_sparsity),
        "method": str(method),
        "vector": {d: None for d in DIM_ORDER},
        "details": {},
        "raw": {},
        "completed_dimensions": [],
        "params": None,
        "actual_sparsity": None,
        "prune_meta": None,
    }


def e2_partial_matches(
    partial: Optional[Mapping[str, Any]],
    seed: int,
    target_sparsity: float,
    method: str,
) -> bool:
    if not partial:
        return False
    return e2_cells_equal(partial, e2_cell_key(seed, target_sparsity, method))


def e2_partial_dims_complete(
    partial: Optional[Mapping[str, Any]],
    *,
    dim_order: Sequence[str] = DIM_ORDER,
) -> bool:
    if not partial:
        return False
    vector = partial.get("vector") or {}
    done = set(partial.get("completed_dimensions") or [])
    return all(d in done and vector.get(d) is not None for d in dim_order)


def merge_e2_capability_dim(
    partial: Mapping[str, Any],
    dim: str,
    cap_slice: Mapping[str, Any],
    *,
    dim_order: Sequence[str] = DIM_ORDER,
) -> Dict[str, Any]:
    """Merge one-dim eval into an E2 partial (seed/target/method keyed)."""
    if dim not in dim_order:
        raise ValueError(f"unknown capability dim: {dim}")
    out = {
        "seed": int(partial["seed"]),
        "target_sparsity": float(partial["target_sparsity"]),
        "method": str(partial["method"]),
        "vector": {d: (partial.get("vector") or {}).get(d) for d in dim_order},
        "details": dict(partial.get("details") or {}),
        "raw": dict(partial.get("raw") or {}),
        "completed_dimensions": [],
        "params": partial.get("params"),
        "actual_sparsity": partial.get("actual_sparsity"),
        "prune_meta": partial.get("prune_meta"),
    }
    out["vector"][dim] = (cap_slice.get("vector") or {}).get(dim)
    details = cap_slice.get("details") or {}
    if dim in details:
        out["details"][dim] = details[dim]
    raw = cap_slice.get("raw") or {}
    if dim in raw:
        out["raw"][dim] = raw[dim]
    prev_done = set(partial.get("completed_dimensions") or [])
    prev_done.add(dim)
    out["completed_dimensions"] = [d for d in dim_order if d in prev_done]
    return out


def finalize_e2_partial(
    partial: Mapping[str, Any],
    *,
    model_path: str,
    mode: str = "scan",
    seed: int = 42,
) -> Dict[str, Any]:
    return finalize_e1_partial_capability(
        partial,
        model_path=model_path,
        mode=mode,
        seed=seed,
        skip_dimensions=(),
    )


def save_e2_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)


def load_e2_checkpoint(
    path: Path,
    expected_digest: str,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    stored = str(data.get("config_digest"))
    if stored != str(expected_digest):
        raise ValueError(
            f"E2 checkpoint config mismatch at {path}; use --fresh to restart "
            f"(checkpoint={stored}, expected={expected_digest})"
        )
    if int(data.get("version", 0)) != E2_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E2 checkpoint version at {path}")
    return data


def load_e2_checkpoint_any(path: Path) -> Optional[Dict[str, Any]]:
    """Load E2 checkpoint checking version only (for shard merge)."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if int(data.get("version", 0)) != E2_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported E2 checkpoint version at {path}")
    return data


def build_e2_checkpoint_payload(
    *,
    config_digest: str,
    smoke: bool,
    dense_capability: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    completed_cells: Sequence[Mapping[str, Any]],
    started_at: float,
    elapsed_sec: float,
    status: str = "in_progress",
    partial: Optional[Mapping[str, Any]] = None,
    imported_oneshot_seed42: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": E2_CHECKPOINT_VERSION,
        "status": status,
        "config_digest": config_digest,
        "smoke": bool(smoke),
        "dense_capability": dict(dense_capability),
        "records": [dict(r) for r in records],
        "completed_cells": [dict(c) for c in completed_cells],
        "imported_oneshot_seed42": bool(imported_oneshot_seed42),
        "started_at": float(started_at),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(elapsed_sec),
    }
    if partial is not None:
        payload["partial"] = dict(partial)
    return payload


def merge_e2_records(
    *record_sets: Sequence[Mapping[str, Any]],
    require_disjoint: bool = True,
) -> List[Dict[str, Any]]:
    """Merge E2 records from shards; key = (seed, target, method)."""
    by_key: Dict[Tuple[int, float, str], Dict[str, Any]] = {}
    for records in record_sets:
        for row in records:
            key = (
                int(row["seed"]),
                float(row["target_sparsity"]),
                str(row["method"]),
            )
            if require_disjoint and key in by_key:
                raise ValueError(f"duplicate E2 cell {key} in merge_e2_records")
            by_key[key] = dict(row)
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[0], x[1], x[2]))]


def cell_is_completed(
    completed_cells: Sequence[Mapping[str, Any]],
    seed: int,
    target_sparsity: float,
    method: str,
) -> bool:
    want = e2_cell_tuple(e2_cell_key(seed, target_sparsity, method))
    return any(e2_cell_tuple(c) == want for c in completed_cells)
