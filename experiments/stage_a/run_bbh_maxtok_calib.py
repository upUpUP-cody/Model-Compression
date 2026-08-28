#!/usr/bin/env python3
"""BBH Reasoning max_gen_toks=1024 calibration vs E1 2048 baseline.

Evaluates base 3B dense Reasoning only (limit=64, batch=2), compares to E1 dense
Reasoning score (2048 default). Optional batch=4 VRAM probe (limit_override=8).
Writes bbh1024_calib.json + bbh1024_calib_report.md. Exit 0 when score gate passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.evaluation.capability import eval_capability_vector, resolve_capability_config
from src.experiments.stage_a_common import ensure_dir, gpu_mem_gb, load_yaml, write_report
from src.utils.qwen_squad_eval import write_json

DEFAULT_E1_CKPT = Path("/mnt/data2/results/E1_oneshot_sparsity_curve/e1_checkpoint.json")
BATCH4_PEAK_GB_LIMIT = 22.0


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _load_e1_baseline_reasoning(e1_ckpt: Path) -> float:
    if not e1_ckpt.is_file():
        raise FileNotFoundError(f"E1 checkpoint missing: {e1_ckpt}")
    data = json.loads(e1_ckpt.read_text(encoding="utf-8"))
    dense = data.get("dense_capability") or {}
    vec = dense.get("vector") or {}
    score = vec.get("Reasoning")
    if score is None:
        raise ValueError(f"E1 checkpoint has no dense Reasoning score: {e1_ckpt}")
    return float(score)


def _reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _run_reasoning_eval(
    model_path: str,
    run_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, Optional[float]]:
    _reset_cuda_peak()
    t0 = time.time()
    cap = eval_capability_vector(model_path, run_config)
    elapsed = time.time() - t0
    mem = gpu_mem_gb()
    return cap, elapsed, mem


def _run_batch4_probe(
    model_path: str,
    base_config: Dict[str, Any],
    *,
    limit_override: int,
) -> Dict[str, Any]:
    probe_config = copy.deepcopy(base_config)
    cap = probe_config.setdefault("evaluation", {}).setdefault("capability", {})
    cap["batch_size"] = 4
    cap["limit_override"] = int(limit_override)
    probe_config.setdefault("hardware", {})["batch_size"] = 4
    result: Dict[str, Any] = {
        "batch_size": 4,
        "limit_override": int(limit_override),
        "max_gen_toks": 1024,
        "oom": False,
        "score": None,
        "wall_sec": None,
        "gpu_mem_gb_peak": None,
        "passed": False,
        "note": None,
    }
    try:
        cap_out, elapsed, mem = _run_reasoning_eval(model_path, probe_config)
        score = cap_out.get("vector", {}).get("Reasoning")
        result["score"] = score
        result["wall_sec"] = elapsed
        result["gpu_mem_gb_peak"] = mem
        if mem is not None and float(mem) > BATCH4_PEAK_GB_LIMIT:
            result["note"] = f"peak {mem:.2f}GB > {BATCH4_PEAK_GB_LIMIT}GB limit"
        else:
            result["passed"] = True
            if mem is not None:
                result["note"] = f"peak {mem:.2f}GB <= {BATCH4_PEAK_GB_LIMIT}GB"
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        result["note"] = "torch.cuda.OutOfMemoryError during batch=4 probe"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result["oom"] = True
            result["note"] = str(exc)
        else:
            raise
    return result


def _build_report(payload: Dict[str, Any]) -> str:
    probe = payload.get("batch4_probe") or {}
    batch4_advice = "可试 batch=4（1024）" if probe.get("passed") else "保持 batch=2（1024）"
    return f"""# BBH Reasoning max_gen_toks=1024 Calibration

## 设置

| 项 | 值 |
|----|-----|
| Model | `{payload.get("model_path")}` |
| Task | bbh (Reasoning only) |
| limit | {payload.get("limit")} |
| batch_size (main) | {payload.get("batch_size")} |
| max_gen_toks | 1024 |
| seed | {payload.get("seed")} |
| E1 baseline (2048) | {_fmt(payload.get("baseline_2048"))} |

## 主校准结果 (batch=2)

| 指标 | 值 |
|------|-----|
| score_1024 | {_fmt(payload.get("score_1024"))} |
| delta vs E1 | {_fmt(payload.get("delta"))} |
| max_abs_delta threshold | {_fmt(payload.get("max_abs_delta"))} |
| score_passed | {payload.get("score_passed")} |
| wall_sec | {_fmt(payload.get("wall_sec"))} |
| gpu_mem_gb_peak | {_fmt(payload.get("gpu_mem_gb_peak"))} |

## batch=4 探针 (limit_override={probe.get("limit_override", "n/a")})

| 指标 | 值 |
|------|-----|
| oom | {probe.get("oom")} |
| score | {_fmt(probe.get("score"))} |
| gpu_mem_gb_peak | {_fmt(probe.get("gpu_mem_gb_peak"))} |
| probe_passed | {probe.get("passed")} |
| note | {probe.get("note", "n/a")} |

## 结论

- **分数门禁**: {"PASS" if payload.get("passed") else "FAIL"}
- **batch=4 建议**: {batch4_advice}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="BBH Reasoning max_gen_toks=1024 calibration")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/stage_a/bbh_maxtok1024_calib.yaml"),
    )
    parser.add_argument("--e1-checkpoint", default=str(DEFAULT_E1_CKPT))
    parser.add_argument("--max-abs-delta", type=float, default=0.02)
    parser.add_argument("--device", default=None, help="override capability.device (e.g. cuda:0)")
    parser.add_argument("--skip-batch4-probe", action="store_true")
    parser.add_argument("--batch4-limit", type=int, default=32)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    model_path = str(config["model"]["path"])

    run_config = copy.deepcopy(config)
    cap_cfg = run_config.setdefault("evaluation", {}).setdefault("capability", {})
    cap_cfg.setdefault("only_dimensions", ["Reasoning"])
    if args.device:
        cap_cfg["device"] = args.device
        run_config.setdefault("hardware", {})["device"] = args.device

    resolved = resolve_capability_config(run_config)
    reasoning_spec = resolved["dimensions"]["Reasoning"]
    gen_kwargs = reasoning_spec.get("gen_kwargs") or {}
    if gen_kwargs.get("max_gen_toks") != 1024:
        raise SystemExit(
            f"[ERROR] expected max_gen_toks=1024 in Reasoning gen_kwargs, got {gen_kwargs!r}"
        )

    baseline = _load_e1_baseline_reasoning(Path(args.e1_checkpoint))
    print(f"[INFO] E1 baseline Reasoning (2048): {baseline:.6f}")

    print(f"[INFO] BBH1024 calib main run batch=2 limit=64 model={model_path}")
    cap_main, wall_sec, mem_peak = _run_reasoning_eval(model_path, run_config)
    score_1024 = cap_main.get("vector", {}).get("Reasoning")
    if score_1024 is None:
        raise SystemExit("[ERROR] Reasoning eval returned no score")
    score_1024_f = float(score_1024)
    delta = score_1024_f - baseline
    score_passed = abs(delta) <= float(args.max_abs_delta)
    print(f"[INFO] score_1024={score_1024_f:.6f} delta={delta:+.6f} score_passed={score_passed}")

    batch4_probe: Optional[Dict[str, Any]] = None
    if not args.skip_batch4_probe:
        print(
            f"[INFO] BBH1024 batch=4 probe limit_override={args.batch4_limit} max_gen_toks=1024"
        )
        batch4_probe = _run_batch4_probe(
            model_path,
            config,
            limit_override=int(args.batch4_limit),
        )
        print(
            f"[INFO] batch4_probe oom={batch4_probe.get('oom')} "
            f"peak={batch4_probe.get('gpu_mem_gb_peak')} passed={batch4_probe.get('passed')}"
        )

    passed = bool(score_passed)
    payload: Dict[str, Any] = {
        "experiment_id": "BBH1024_CALIB",
        "passed": passed,
        "score_passed": score_passed,
        "model_path": model_path,
        "limit": int(resolved["scan_limits"]["Reasoning"]),
        "batch_size": int(resolved["batch_size"]),
        "seed": int(resolved["seed"]),
        "gen_kwargs": dict(gen_kwargs),
        "baseline_2048": baseline,
        "baseline_source": str(args.e1_checkpoint),
        "score_1024": score_1024_f,
        "delta": delta,
        "max_abs_delta": float(args.max_abs_delta),
        "wall_sec": wall_sec,
        "gpu_mem_gb_peak": mem_peak,
        "batch4_probe": batch4_probe,
        "details": cap_main.get("details", {}).get("Reasoning"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    write_json(out / "bbh1024_calib.json", payload)
    write_report(out / "bbh1024_calib_report.md", _build_report(payload))
    flag = out / ("bbh1024_calib_passed.flag" if passed else "bbh1024_calib_failed.flag")
    flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    other = out / ("bbh1024_calib_failed.flag" if passed else "bbh1024_calib_passed.flag")
    if other.is_file():
        other.unlink()

    print(f"[INFO] BBH1024 calib passed={passed} wrote {out / 'bbh1024_calib.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
