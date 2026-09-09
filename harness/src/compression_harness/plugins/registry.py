"""Builtin plugin registry."""

from __future__ import annotations

from typing import Dict

from compression_harness.plugins.base import CompressionPlugin
from compression_harness.plugins.evaluate import EvaluatePlugin
from compression_harness.plugins.evaluate_lm import RealEvaluatePlugin
from compression_harness.plugins.prune import PrunePlugin
from compression_harness.plugins.prune_wanda import WandaPrunePlugin
from compression_harness.plugins.quantize import QuantizePlugin
from compression_harness.plugins.quantize_torchao import TorchAOLayerQuantizePlugin

_REGISTRY: Dict[str, CompressionPlugin] = {}
_BUILTINS_LOADED = False


def register_plugin(plugin: CompressionPlugin) -> None:
    _REGISTRY[plugin.name] = plugin


def register_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for cls in (
        PrunePlugin,
        QuantizePlugin,
        EvaluatePlugin,
        WandaPrunePlugin,
        TorchAOLayerQuantizePlugin,
        RealEvaluatePlugin,
    ):
        register_plugin(cls())
    _BUILTINS_LOADED = True


def get_plugin(name: str) -> CompressionPlugin:
    register_builtin_plugins()
    if name not in _REGISTRY:
        for p in _REGISTRY.values():
            if p.kind == name or p.name == name:
                return p
        raise KeyError(f"Unknown plugin {name!r}; known={sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_plugin_by_kind(kind: str) -> CompressionPlugin:
    register_builtin_plugins()
    for p in _REGISTRY.values():
        if p.kind == kind:
            return p
    raise KeyError(f"No plugin for kind {kind!r}")


def list_plugins() -> list[dict[str, str]]:
    register_builtin_plugins()
    return [{"name": p.name, "kind": p.kind} for p in _REGISTRY.values()]
