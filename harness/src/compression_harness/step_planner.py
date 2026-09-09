"""Propose short layer-wise prune/quantize steps."""

from __future__ import annotations

from typing import Any, Mapping

from compression_harness.plugins.base import ModelState, StepSpec, validate_step_spec


class StepPlanner:
    """Alternate prune then quantize over successive layer windows."""

    def __init__(self, goal: Mapping[str, Any], *, use_real: bool = False) -> None:
        g = goal.get("goal", goal)
        search = g.get("search") or {}
        model = g.get("model") or {}
        self.num_layers = int(model.get("num_layers") or 36)
        self.max_layers = int(search.get("max_layers_per_step") or 2)
        self.max_sparsity_delta = float(search.get("max_sparsity_delta") or 0.05)
        self.min_weight_bits = int(search.get("min_weight_bits") or 8)
        self.plugins_order = list(search.get("plugins_order") or [])
        self.sensitive_layers = set(
            int(x) for x in (g.get("knowledge") or {}).get("sensitive_layers") or []
        )
        self.use_real = use_real
        self.prune_plugin = "prune_wanda" if use_real else "prune_stub"
        self.quantize_plugin = "quantize_torchao_layers" if use_real else "quantize_stub"

    def next_step(self, state: ModelState, round_index: int) -> StepSpec | None:
        """Return next short step or None if search space exhausted."""
        kind = self._kind_for_round(round_index)
        window = self._next_window(state, kind)
        if window is None:
            return None
        lo, hi = window
        if kind == "prune":
            step: StepSpec = {
                "step_id": f"r{round_index:03d}_prune_{lo}_{hi}",
                "plugin": self.prune_plugin,
                "kind": "prune",
                "layer_range": [lo, hi],
                "sparsity_delta": self.max_sparsity_delta,
            }
        else:
            step = {
                "step_id": f"r{round_index:03d}_quant_{lo}_{hi}",
                "plugin": self.quantize_plugin,
                "kind": "quantize",
                "layer_range": [lo, hi],
                "weight_bits": self.min_weight_bits,
            }
        errs = validate_step_spec(
            step,
            max_layers_per_step=self.max_layers,
            max_sparsity_delta=self.max_sparsity_delta,
            min_weight_bits=self.min_weight_bits,
        )
        if errs:
            raise ValueError(f"planner produced invalid step: {errs}")
        return step

    def _kind_for_round(self, round_index: int) -> str:
        if self.plugins_order:
            return self.plugins_order[(round_index - 1) % len(self.plugins_order)]
        return "prune" if round_index % 2 == 1 else "quantize"

    def _next_window(self, state: ModelState, kind: str) -> tuple[int, int] | None:
        """Scan layers for the next window not yet fully processed for this kind."""
        cursor_key = f"cursor_{kind}"
        start = int(state.meta.get(cursor_key) or 0)
        while start < self.num_layers:
            if start in self.sensitive_layers:
                start += 1
                continue
            hi = min(start + self.max_layers - 1, self.num_layers - 1)
            if kind == "prune":
                if all(
                    state.layer_sparsity.get(i, 0.0) >= self.max_sparsity_delta - 1e-12
                    for i in range(start, hi + 1)
                ):
                    start = hi + 1
                    continue
            else:
                if all(
                    state.layer_bits.get(i, 16) <= self.min_weight_bits
                    for i in range(start, hi + 1)
                ):
                    start = hi + 1
                    continue
            state.meta[cursor_key] = hi + 1
            return start, hi
        return None

    def mark_sensitive(self, state: ModelState, layers: list[int]) -> None:
        for i in layers:
            self.sensitive_layers.add(int(i))
        state.meta["sensitive_layers"] = sorted(self.sensitive_layers)
