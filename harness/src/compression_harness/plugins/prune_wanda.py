"""Real Wanda MLP short-step prune plugin."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compression_harness.model_io import get_model, get_tokenizer
from compression_harness.paths import REPO_ROOT
from compression_harness.plugins.base import CompressionPlugin, ModelState, StepSpec
from compression_harness.plugins.layer_utils import layer_indices, mlp_intermediate_names


def _ensure_repo_src() -> None:
    """Prefer ``src/experiments`` over repo-root ``experiments/`` (empty stub)."""
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


def _linear_weight_l2_row_norms(weight: torch.Tensor) -> torch.Tensor:
    """Row L2 norms; dequantize AO Int8Tensor when needed."""
    w = weight.detach()
    try:
        return torch.norm(w.float(), p=2, dim=1).float().cpu()
    except NotImplementedError:
        pass
    # torchao Int8Tensor / AffineQuantizedTensor style
    for attr in ("dequantize", "to_float", "get_plain"):
        fn = getattr(w, attr, None)
        if callable(fn):
            try:
                plain = fn() if attr != "get_plain" else fn()[0]
                if isinstance(plain, torch.Tensor):
                    return torch.norm(plain.float(), p=2, dim=1).float().cpu()
            except Exception:
                continue
    if hasattr(w, "int_data") and hasattr(w, "scale"):
        plain = w.int_data.float() * w.scale.float()
        return torch.norm(plain, p=2, dim=1).float().cpu()
    raise TypeError(f"cannot compute row norms for weight type {type(w)}")


def wanda_importance_mlp_names(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    target_names: Set[str],
    *,
    num_batches: int = 2,
) -> Dict[str, torch.Tensor]:
    """Wanda scores only for named MLP intermediates (skips other / INT8 layers)."""
    _ensure_repo_src()
    from pruning.transformer_structured_pruning import TransformerStructuredPruning

    if not target_names:
        return {}
    if not isinstance(num_batches, int) or num_batches <= 0:
        raise ValueError("num_batches must be a positive integer")

    pruner = TransformerStructuredPruning(model)
    names = [n for n in pruner.prunable_layer_names() if n in target_names]
    if not names:
        names = sorted(target_names)

    activation_sums: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    hooks = []
    resolved = torch.device(device)
    non_blocking = resolved.type == "cuda"
    was_training = model.training
    model.eval()
    model.to(resolved)

    def _accumulate(layer_name: str, tensor: torch.Tensor) -> None:
        values = tensor.detach().abs()
        if values.ndim == 3:
            batch_sum = values.sum(dim=(0, 1))
            count = int(values.shape[0] * values.shape[1])
        elif values.ndim == 2:
            batch_sum = values.sum(dim=0)
            count = int(values.shape[0])
        else:
            raise ValueError(f"expected 2D or 3D MLP activations for {layer_name}, got {values.ndim}D")
        if layer_name not in activation_sums:
            activation_sums[layer_name] = batch_sum.clone()
            token_counts[layer_name] = count
        else:
            activation_sums[layer_name].add_(batch_sum)
            token_counts[layer_name] += count

    try:
        for name in names:
            kind, index = pruner._parse_name(name)
            if kind != "mlp":
                continue
            mlp = pruner.base.layers[index].mlp

            def make_pre_hook(layer_name: str):
                def hook(_module: nn.Module, inputs: tuple) -> None:
                    _accumulate(layer_name, inputs[0])

                return hook

            hooks.append(mlp.down_proj.register_forward_pre_hook(make_pre_hook(name)))

        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_batches:
                    break
                input_ids = batch["input_ids"].to(resolved, non_blocking=non_blocking)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(resolved, non_blocking=non_blocking)
                model(input_ids=input_ids, attention_mask=attention_mask)
                batch_count += 1
        if batch_count == 0:
            raise ValueError("dataloader must yield at least one batch for Wanda calibration")
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    importance: Dict[str, torch.Tensor] = {}
    for name in names:
        kind, index = pruner._parse_name(name)
        if kind != "mlp":
            continue
        mlp = pruner.base.layers[index].mlp
        weight_mag = _linear_weight_l2_row_norms(mlp.gate_proj.weight)
        mean_act = (activation_sums[name] / max(token_counts[name], 1)).float().cpu()
        if mean_act.numel() != weight_mag.numel():
            raise ValueError(
                f"activation width mismatch for {name}: {mean_act.numel()} vs {weight_mag.numel()}"
            )
        importance[name] = weight_mag * mean_act
    return importance


class WandaPrunePlugin(CompressionPlugin):
    """Apply Stage-A Wanda MLP intermediate prune on a short layer window."""

    name = "prune_wanda"
    kind = "prune"

    def apply(self, state: ModelState, step: StepSpec) -> tuple[ModelState, dict[str, Any]]:
        _ensure_repo_src()
        from experiments.stage_a_common import (
            prepare_wanda_calibration_loader,
            remaining_relative_prune_ratio,
        )
        from pruning.pruning_backend import resolve_pruning_backend

        model = get_model(state)
        tokenizer = get_tokenizer(state)
        indices = layer_indices(step.get("layer_range"))
        delta = float(step.get("sparsity_delta") or step.get("sparsity") or 0.05)
        target_names = set(mlp_intermediate_names(indices))

        layer_ratios: Dict[str, float] = {}
        for i in indices:
            prev = float(state.layer_sparsity.get(i, 0.0))
            nxt = min(0.95, prev + delta)
            if nxt <= prev + 1e-12:
                continue
            ratio = remaining_relative_prune_ratio(prev, nxt)
            layer_ratios[f"layers.{i}.mlp.intermediate"] = ratio

        if not layer_ratios:
            metrics = {
                "status": "ok",
                "plugin": self.name,
                "kind": self.kind,
                "layers": indices,
                "message": "[OK] prune_wanda no-op (already at target sparsity)",
            }
            return state, metrics

        device = "cuda" if _cuda() else "cpu"
        calib = state.meta.get("_calib_loader")
        if calib is None:
            cfg = {
                "seed": 42,
                "model": {"torch_dtype": "bfloat16"},
                "hardware": {"batch_size": 1, "device": device},
                "pruning": {
                    "calibration": {
                        "batch_size": 1,
                        "max_seq_len": 256,
                        "train_max_samples": 64,
                    }
                },
                "dataset": {"cache_dir": "/mnt/data/datasets/glue"},
            }
            calib = prepare_wanda_calibration_loader(cfg, tokenizer, device)
            state.meta["_calib_loader"] = calib

        num_batches = int(step.get("wanda_batches") or state.meta.get("wanda_batches") or 2)
        print(f"[INFO] prune_wanda layers={indices} delta={delta} batches={num_batches}")
        # Only score the window — avoids torch.norm on previously INT8 layers.
        importance = wanda_importance_mlp_names(
            model,
            calib,
            device,
            set(layer_ratios.keys()),
            num_batches=num_batches,
        )
        if not importance:
            raise RuntimeError(f"Wanda importance empty for layers {indices}")

        keep_indices: Dict[str, List[int]] = {}
        for name, scores in importance.items():
            ratio = float(layer_ratios.get(name) or delta)
            size = int(scores.numel())
            keep_count = max(1, int(round(size * (1.0 - ratio))))
            ranked = sorted(range(size), key=lambda i: (-float(scores[i]), i))
            keep_indices[name] = sorted(ranked[:keep_count])

        cpu_model = model.cpu()
        backend = resolve_pruning_backend(cpu_model, "qwen")
        pruned = backend.create_pruned_model_by_indices(keep_indices)
        if device == "cuda":
            pruned = pruned.cuda()
        pruned.eval()
        state.meta["_model"] = pruned

        ops = list(state.meta.get("prune_ops") or [])
        ops.append({k: list(v) for k, v in keep_indices.items()})
        state.meta["prune_ops"] = ops

        for i in indices:
            prev = float(state.layer_sparsity.get(i, 0.0))
            state.layer_sparsity[i] = min(0.95, prev + delta)
        state.applied_steps.append(dict(step))

        metrics = {
            "status": "ok",
            "plugin": self.name,
            "kind": self.kind,
            "layers": indices,
            "sparsity_delta": delta,
            "layer_ratios": layer_ratios,
            "keep_indices": {k: list(v) for k, v in keep_indices.items()},
            "layer_sparsity": {str(i): state.layer_sparsity[i] for i in indices},
            "message": "[OK] prune_wanda applied",
        }
        return state, metrics


def _cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
