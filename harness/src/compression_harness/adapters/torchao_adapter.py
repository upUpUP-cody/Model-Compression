"""TorchAO real INT8 weight-only adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Sequence

import torch
import torch.nn as nn


_ALLOWED_METHODS = frozenset({"int8_weight_only", "int8"})
_WEIGHTS_NAME = "quantized_state_dict.pt"


def _exclude_filter(exclude: Sequence[str]) -> Callable[[nn.Module, str], bool]:
    exclude_set = {str(x) for x in exclude}

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        parts = fqn.split(".")
        for ex in exclude_set:
            if fqn == ex or fqn.endswith("." + ex) or ex in parts:
                return False
        return True

    return filter_fn


def _build_int8_config(group_size: Any) -> Any:
    from torchao.quantization import Int8WeightOnlyConfig

    gs = group_size
    if gs is not None:
        gs = int(gs)
        if gs <= 0:
            gs = None
    return Int8WeightOnlyConfig(group_size=gs)


def _load_hf_model(model_ref: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[INFO] torchao load model={model_ref} device={device}")
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


def apply_int8_weight_only(
    model: nn.Module,
    *,
    exclude: Sequence[str],
    group_size: Any = None,
) -> None:
    from torchao.quantization import quantize_

    cfg = _build_int8_config(group_size)
    print(f"[INFO] torchao quantize_ Int8WeightOnlyConfig exclude={list(exclude)}")
    quantize_(model, cfg, filter_fn=_exclude_filter(exclude))


class TorchAOAdapter:
    """Apply torchao Int8WeightOnlyConfig and persist artifacts."""

    name = "torchao"

    def compress(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        method = str(recipe.get("method", ""))
        layers = recipe.get("layers") or {}
        default = layers.get("default") or {}
        weight_bits = int(default.get("weight_bits", 8))
        if method not in _ALLOWED_METHODS or weight_bits != 8:
            raise ValueError(
                f"TorchAOAdapter Phase B only supports INT8 weight-only "
                f"(method in {sorted(_ALLOWED_METHODS)}, weight_bits=8); "
                f"got method={method!r} weight_bits={weight_bits}"
            )

        if output_dir is None:
            raise ValueError("output_dir is required for real TorchAO compress")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = _load_hf_model(model_ref)
        exclude: List[str] = list(recipe.get("exclude") or ["lm_head"])
        group_size = default.get("group_size")
        apply_int8_weight_only(model, exclude=exclude, group_size=group_size)

        tokenizer.save_pretrained(out)
        if getattr(model, "config", None) is not None:
            model.config.save_pretrained(out)

        weights_path = out / _WEIGHTS_NAME
        # Full-module pickle fails on AO partials; state_dict round-trips.
        torch.save(model.state_dict(), weights_path)
        meta = {
            "backend": "torchao",
            "method": method,
            "recipe_id": recipe.get("recipe_id"),
            "model_ref": model_ref,
            "exclude": exclude,
            "group_size": group_size if group_size not in (None, 0) else None,
            "weight_bits": weight_bits,
            "weights_file": weights_path.name,
            "save_format": "state_dict_after_quantize",
        }
        with (out / "harness_compress_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

        print(f"[OK] torchao compressed -> {out}")
        return {
            "status": "ok",
            "backend": "torchao",
            "method": method,
            "model_ref": model_ref,
            "recipe_id": recipe.get("recipe_id"),
            "output_dir": str(out),
            "weights_path": str(weights_path),
            "model": model,
            "tokenizer": tokenizer,
            "message": "[OK] TorchAO INT8 weight-only compress finished",
        }


def load_quantized_model(compressed_dir: str | Path) -> Any:
    """Reload: dense HF model -> same INT8 recipe -> load quantized state_dict."""
    out = Path(compressed_dir)
    meta_path = out / "harness_compress_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta: {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    model_ref = str(meta["model_ref"])
    weights_path = out / str(meta.get("weights_file") or _WEIGHTS_NAME)
    if not weights_path.exists():
        raise FileNotFoundError(f"missing quantized weights: {weights_path}")

    model, _tokenizer = _load_hf_model(model_ref)
    apply_int8_weight_only(
        model,
        exclude=list(meta.get("exclude") or ["lm_head"]),
        group_size=meta.get("group_size"),
    )
    try:
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model
