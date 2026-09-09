"""Runtime model handle + checkpoint IO for iterative real plugins."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from compression_harness.paths import REPO_ROOT
from compression_harness.plugins.base import ModelState

RUNTIME_META_KEYS = ("_model", "_tokenizer", "_calib_loader")
_WEIGHTS_NAME = "weights.pt"
_META_NAME = "ckpt_meta.json"


def strip_runtime_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in meta.items()
        if k not in RUNTIME_META_KEYS and not str(k).startswith("_")
    }


def attach_runtime(
    state: ModelState,
    *,
    model: nn.Module,
    tokenizer: Any,
    calib_loader: Any = None,
) -> None:
    state.meta["_model"] = model
    state.meta["_tokenizer"] = tokenizer
    if calib_loader is not None:
        state.meta["_calib_loader"] = calib_loader


def get_model(state: ModelState) -> nn.Module:
    model = state.meta.get("_model")
    if model is None:
        raise RuntimeError("ModelState has no attached _model; load or attach first")
    return model


def get_tokenizer(state: ModelState) -> Any:
    tok = state.meta.get("_tokenizer")
    if tok is None:
        raise RuntimeError("ModelState has no attached _tokenizer")
    return tok


def clear_runtime(state: ModelState) -> None:
    for k in RUNTIME_META_KEYS:
        state.meta.pop(k, None)


def _ensure_repo_src() -> None:
    """Prefer ``src/`` packages over repo-root stubs on ``sys.path``."""
    src = str(REPO_ROOT / "src")
    for key in list(sys.modules):
        if key == "experiments" or key.startswith("experiments."):
            mod = sys.modules.get(key)
            path = getattr(mod, "__file__", None) or ""
            if path.startswith(str(REPO_ROOT / "experiments")):
                del sys.modules[key]
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)


def load_hf_model(model_ref: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[INFO] load HF model={model_ref} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _replay_structure(model: nn.Module, prune_ops: list[dict[str, list[int]]]) -> nn.Module:
    """Replay structured MLP keep-indices from dense shell (CPU then return)."""
    if not prune_ops:
        return model
    _ensure_repo_src()
    from pruning.pruning_backend import resolve_pruning_backend

    device = next(model.parameters()).device
    cpu_model = model.cpu()
    for op in prune_ops:
        keep = {str(k): list(v) for k, v in op.items()}
        if not keep:
            continue
        backend = resolve_pruning_backend(cpu_model, "qwen")
        cpu_model = backend.create_pruned_model_by_indices(keep)
    if device.type == "cuda":
        cpu_model = cpu_model.to(device)
    cpu_model.eval()
    return cpu_model


def save_checkpoint(state: ModelState, ckpt_dir: str | Path) -> str:
    """Persist weights + structure ops for exact reload after structured prune."""
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = get_model(state)
    tokenizer = get_tokenizer(state)
    try:
        tokenizer.save_pretrained(out)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] tokenizer save failed: {exc}")
    if getattr(model, "config", None) is not None:
        try:
            model.config.save_pretrained(out)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] config save failed: {exc}")

    weights_path = out / _WEIGHTS_NAME
    # Move to CPU snapshot for portability; keep live model on GPU.
    cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Write via tmp then rename to avoid partial zip on ENOSPC.
    tmp_path = out / (_WEIGHTS_NAME + ".tmp")
    torch.save(cpu_sd, tmp_path)
    tmp_path.replace(weights_path)


    meta = {
        "model_ref": state.model_ref,
        "layer_sparsity": {str(k): v for k, v in state.layer_sparsity.items()},
        "layer_bits": {str(k): v for k, v in state.layer_bits.items()},
        "round_index": state.round_index,
        "prune_ops": list(state.meta.get("prune_ops") or []),
        "applied_steps": list(state.applied_steps),
        "weights_file": _WEIGHTS_NAME,
    }
    with (out / _META_NAME).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    state.checkpoint_dir = str(out)
    print(f"[OK] checkpoint saved -> {out}")
    return str(out)


def load_checkpoint_into_state(state: ModelState, ckpt_dir: str | Path) -> ModelState:
    """Rebuild structure from dense + prune_ops, optionally INT8, then load weights."""
    out = Path(ckpt_dir)
    meta_path = out / _META_NAME
    prune_ops: list[dict[str, list[int]]] = []
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            ckpt_meta = json.load(f)
        state.layer_sparsity = {
            int(k): float(v) for k, v in (ckpt_meta.get("layer_sparsity") or {}).items()
        }
        state.layer_bits = {int(k): int(v) for k, v in (ckpt_meta.get("layer_bits") or {}).items()}
        state.round_index = int(ckpt_meta.get("round_index") or state.round_index)
        state.applied_steps = list(ckpt_meta.get("applied_steps") or state.applied_steps)
        prune_ops = list(ckpt_meta.get("prune_ops") or [])
        state.meta["prune_ops"] = prune_ops

    model, tokenizer = load_hf_model(state.model_ref)
    model = _replay_structure(model, prune_ops)

    bits_layers = sorted(state.layer_bits.keys())
    if bits_layers:
        from compression_harness.plugins.quantize_torchao import apply_int8_to_layer_range

        lo, hi = min(bits_layers), max(bits_layers)
        apply_int8_to_layer_range(model, lo, hi, exclude=("lm_head",))

    weights_path = out / _WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights in {out}")
    try:
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARNING] checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARNING] checkpoint unexpected keys: {len(unexpected)}")
    model.eval()
    calib = state.meta.get("_calib_loader")
    attach_runtime(state, model=model, tokenizer=tokenizer, calib_loader=calib)
    state.checkpoint_dir = str(out)
    print(f"[OK] checkpoint loaded <- {out}")
    return state
