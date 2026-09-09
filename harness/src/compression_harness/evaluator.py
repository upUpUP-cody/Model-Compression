"""Evaluate compressed vs dense; near-lossless gate (hellaswag@64)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from compression_harness.paths import EXPERIMENTS_DIR, REPO_ROOT

BASELINE_CACHE_NAME = "_baseline_hellaswag64.json"
PRIMARY_DIM = "Reasoning"


def _ensure_repo_src_on_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def hellaswag_capability_config(goal_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Build eval_capability_vector config: Reasoning -> hellaswag limit 64."""
    g = goal_doc.get("goal", goal_doc)
    ev = g.get("evaluation") or {}
    limit = int(ev.get("limit", 64))
    seed = int(g.get("seed", 42))
    dtype = str((g.get("model") or {}).get("dtype", "bfloat16"))
    return {
        "seed": seed,
        "hardware": {"device": "cuda:0", "batch_size": 1},
        "model": {"torch_dtype": dtype},
        "evaluation": {
            "capability": {
                "mode": "scan",
                "only_dimensions": [PRIMARY_DIM],
                "scan_limits": {PRIMARY_DIM: limit},
                "batch_size": 1,
                "device": "cuda:0",
                "dtype": dtype,
                "bootstrap_iters": 0,
                "log_samples": False,
                "dimensions": {
                    PRIMARY_DIM: {
                        "task": "hellaswag",
                        "num_fewshot": 0,
                        "apply_chat_template": False,
                        "gen_kwargs": None,
                        "metric_candidates": (
                            "acc_norm,none",
                            "acc_norm",
                            "acc,none",
                            "acc",
                        ),
                    }
                },
            }
        },
    }


def _score_from_capability(result: Mapping[str, Any]) -> float:
    vector = result.get("vector") or {}
    score = vector.get(PRIMARY_DIM)
    if score is None:
        details = (result.get("details") or {}).get(PRIMARY_DIM) or {}
        score = details.get("score")
    if score is None:
        raise RuntimeError(f"hellaswag score missing in capability result: {result.get('details')}")
    return float(score)


class Evaluator:
    def __init__(self, *, use_real: bool = True, cache_dir: Path | None = None) -> None:
        self.use_real = use_real
        self.cache_dir = Path(cache_dir) if cache_dir else EXPERIMENTS_DIR

    def _cache_path(self, model_ref: str, limit: int) -> Path:
        safe = model_ref.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"_baseline_hellaswag{limit}_{safe}.json"

    def _legacy_cache_path(self) -> Path:
        return self.cache_dir / BASELINE_CACHE_NAME

    def relative_drop(self, baseline: float, score: float) -> float:
        if baseline <= 0:
            return 1.0
        return max(0.0, (baseline - score) / baseline)

    def near_lossless_ok(self, baseline: float, score: float, goal: dict[str, Any]) -> bool:
        quality = goal.get("goal", goal).get("quality", {})
        max_drop = float(quality.get("max_relative_drop", 0.02))
        return self.relative_drop(baseline, score) <= max_drop + 1e-12

    def _run_capability(
        self,
        model_path: str,
        goal_doc: Mapping[str, Any],
        *,
        model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        _ensure_repo_src_on_path()
        from evaluation.capability import eval_capability_vector

        cfg = hellaswag_capability_config(goal_doc)
        print(f"[INFO] harness evaluate hellaswag model_path={model_path} use_model={model is not None}")
        return eval_capability_vector(
            model_path,
            cfg,
            model=model,
            tokenizer=tokenizer,
        )

    def baseline(
        self,
        model_ref: str,
        goal: dict[str, Any],
        *,
        force_refresh: bool = False,
    ) -> float:
        if not self.use_real:
            return 1.0
        g = goal.get("goal", goal)
        limit = int((g.get("evaluation") or {}).get("limit", 64))
        cache_path = self._cache_path(model_ref, limit)
        legacy = self._legacy_cache_path()
        if not force_refresh:
            for path in (cache_path, legacy):
                if path.exists():
                    with path.open(encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("model_ref") == model_ref and int(data.get("limit", limit)) == limit:
                        print(f"[OK] baseline cache hit {path}")
                        return float(data["score"])

        result = self._run_capability(model_ref, goal)
        score = _score_from_capability(result)
        payload = {
            "model_ref": model_ref,
            "limit": limit,
            "task": "hellaswag",
            "metric": "acc_norm",
            "score": score,
            "details": result.get("details"),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        # Also write stable name for docs / smoke scripts.
        with legacy.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"[OK] baseline hellaswag@{limit}={score:.6f} cached -> {cache_path}")
        return score

    def score_run(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        goal: dict[str, Any],
        *,
        model: Any = None,
        tokenizer: Any = None,
        compressed_dir: str | None = None,
    ) -> float:
        if not self.use_real:
            bits = (recipe.get("layers") or {}).get("default", {}).get("weight_bits", 16)
            if bits >= 8:
                return 0.99
            if bits >= 4:
                return 0.95
            return 0.90

        eval_path = compressed_dir or model_ref
        loaded_model = model
        loaded_tok = tokenizer
        if loaded_model is None and compressed_dir:
            weights = Path(compressed_dir) / "quantized_state_dict.pt"
            legacy = Path(compressed_dir) / "quantized_model.pt"
            if weights.exists() or legacy.exists():
                from compression_harness.adapters.torchao_adapter import load_quantized_model
                from transformers import AutoTokenizer

                loaded_model = load_quantized_model(compressed_dir)
                loaded_tok = AutoTokenizer.from_pretrained(compressed_dir, trust_remote_code=True)
                eval_path = model_ref

        result = self._run_capability(
            eval_path if loaded_model is None else model_ref,
            goal,
            model=loaded_model,
            tokenizer=loaded_tok,
        )
        return _score_from_capability(result)

    def evaluate(
        self,
        model_ref: str,
        recipe: dict[str, Any],
        goal: dict[str, Any],
        *,
        baseline_score: float | None = None,
        model: Any = None,
        tokenizer: Any = None,
        compressed_dir: str | None = None,
    ) -> dict[str, Any]:
        g = goal.get("goal", goal)
        base = baseline_score if baseline_score is not None else self.baseline(model_ref, goal)
        score = self.score_run(
            model_ref,
            recipe,
            goal,
            model=model,
            tokenizer=tokenizer,
            compressed_dir=compressed_dir,
        )
        drop = self.relative_drop(base, score)
        max_drop = float(g.get("quality", {}).get("max_relative_drop", 0.02))
        ok = drop <= max_drop + 1e-12
        return {
            "status": "ok" if self.use_real else "dry_run",
            "baseline_score": base,
            "score": score,
            "relative_drop": drop,
            "near_lossless_ok": ok,
            "primary_metric": g.get("evaluation", {}).get("primary_metric", "acc_norm"),
            "task": "hellaswag",
            "limit": int((g.get("evaluation") or {}).get("limit", 64)),
        }
