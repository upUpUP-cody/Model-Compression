#!/usr/bin/env python3
"""E0 Code-dimension gate: re-eval HumanEval on Instruct after protocol fix.

Does not re-run other dimensions. Updates e0_summary / e0_report Code fields in place.
Writes e0_code_gate.json with pass/fail. Exit 0 only when gate passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DIM_ORDER, eval_capability_vector
from src.experiments.stage_a_common import ensure_dir, gpu_mem_gb, load_yaml, mirror_docs_report, write_report
from src.utils.qwen_squad_eval import write_json


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _merge_code_into_summary(summary: dict, code_cap: dict, *, run_id: str) -> dict:
    score = code_cap["vector"].get("Code")
    out = copy.deepcopy(summary)
    out.setdefault("outputs", {}).setdefault("P_M0", {})["Code"] = score
    cap = out.setdefault("capability", {})
    vec = dict(cap.get("vector") or {d: None for d in DIM_ORDER})
    vec["Code"] = score
    cap["vector"] = vec
    cap["vector_list"] = [vec.get(d) for d in DIM_ORDER]
    details = dict(cap.get("details") or {})
    details["Code"] = code_cap.get("details", {}).get("Code")
    cap["details"] = details
    raw = dict(cap.get("raw") or {})
    raw["Code"] = (code_cap.get("raw") or {}).get("Code")
    cap["raw"] = raw
    protocol = dict(cap.get("protocol") or {})
    protocol.update((code_cap.get("protocol") or {}))
    cap["protocol"] = protocol
    cap["only_dimensions"] = list(code_cap.get("only_dimensions") or ["Code"])
    out["capability"] = cap
    out["code_gate"] = {
        "run_id": run_id,
        "score": score,
        "protocol_note": "humaneval completion; apply_chat_template=False; gen_kwargs until/max_gen_toks",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return out


def _build_report_from_summary(summary: dict, elapsed: float) -> str:
    vector = (summary.get("capability") or {}).get("vector") or summary.get("outputs", {}).get("P_M0") or {}
    resources = summary.get("outputs", {}).get("resources") or {}
    detail = (summary.get("capability") or {}).get("details") or {}
    limit_lines = ["| 维 | 任务 | limit | 指标 |", "|----|------|-------|------|"]
    for dim in DIM_ORDER:
        d = detail.get(dim) or {}
        limit_lines.append(
            f"| {dim} | {d.get('task', 'n/a')} | {d.get('limit', 'n/a')} | {d.get('metric_key', 'n/a')} |"
        )
    code_note = ""
    gate = summary.get("code_gate") or {}
    if gate:
        code_note = (
            f"\n- Code 维已用修复后的 HumanEval 补全协议重评（gate score={_fmt(gate.get('score'))}）。\n"
        )
    vec_row = " | ".join(_fmt(vector.get(d)) for d in DIM_ORDER)
    return f"""# E0. Dense Model Baseline

## 项目 / 设置

| 项 | 设置 |
|----|------|
| Experiment ID | E0 |
| 目的 | 建立所有后续实验的 dense baseline |
| Model | {summary.get("spec")} (PDF: {summary.get("pdf_model")}) |
| Method / Compression | None |
| Evaluation | 标准小样本六维（lm_eval, mode=scan, seed={summary.get("seeds", 42)}） |
| Seeds | {summary.get("seeds", 42)} |
| GPU | {summary.get("gpu")} |
| 优先级 | P0 |
| 成功条件 | 所有 benchmark pipeline 可稳定复现 |
| status | {summary.get("status", "done")} |

## 六维协议（冻结）

{chr(10).join(limit_lines)}

## 输出

Dense performance vector P(M0) 与资源行（PDF §E0）。

## 记录表

| PPL | Math | Knowledge | Reasoning | Instruction | Code |
|-----|------|-----------|-----------|-------------|------|
| {vec_row} |

| GPU memory (GB peak) | latency_sec_eval | parameter_count | model path |
|----------------------|------------------|-----------------|------------|
| {resources.get("gpu_memory_gb_peak", "n/a")} | {elapsed:.1f} | {resources.get("parameter_count", "n/a")} | `{resources.get("model_size_path", "")}` |

## 结论（对照成功条件）

- **满足**：六维 lm_eval pipeline 已落盘；小样本协议主看后续相对 Delta 与 capability-specific cliff。
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E。
- 旧 SST-2 部分向量产物作废；以本报告为准。{code_note}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e0_dense.yaml"))
    parser.add_argument("--min-score", type=float, default=1e-6, help="pass@1 must be > this")
    parser.add_argument("--repeat", type=int, default=2, help="reproducibility runs (default 2)")
    parser.add_argument("--device", default=None, help="override capability.device (e.g. cuda:0)")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    summary_path = out / "e0_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing E0 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    run_config = copy.deepcopy(config)
    cap = run_config.setdefault("evaluation", {}).setdefault("capability", {})
    cap["only_dimensions"] = ["Code"]
    if args.device:
        cap["device"] = args.device
        run_config.setdefault("hardware", {})["device"] = args.device

    model_path = str(config["model"]["path"])
    if "Instruct" not in model_path and "instruct" not in model_path.lower():
        raise SystemExit(f"[ERROR] E0 Code gate requires Instruct model, got {model_path}")

    scores = []
    last_cap = None
    t0 = time.time()
    for i in range(max(1, int(args.repeat))):
        print(f"[INFO] E0 Code gate run {i + 1}/{args.repeat} model={model_path}")
        last_cap = eval_capability_vector(model_path, run_config)
        score = last_cap["vector"].get("Code")
        scores.append(score)
        print(f"[INFO] E0 Code gate run {i + 1} pass@1={score}")

    elapsed = time.time() - t0
    mem = gpu_mem_gb()
    assert last_cap is not None
    primary = scores[0]
    ok_nonzero = primary is not None and float(primary) > float(args.min_score)
    ok_repro = True
    if len(scores) >= 2 and scores[0] is not None and scores[1] is not None:
        # Same order of magnitude / both > 0 (scan noise allowed).
        ok_repro = float(scores[1]) > float(args.min_score) and abs(float(scores[0]) - float(scores[1])) <= 0.5
    passed = bool(ok_nonzero and ok_repro)

    summary = _merge_code_into_summary(summary, last_cap, run_id="code_gate")
    resources = summary.setdefault("outputs", {}).setdefault("resources", {})
    resources["code_gate_latency_sec"] = elapsed
    resources["code_gate_gpu_memory_gb_peak"] = mem
    write_json(summary_path, summary)

    report = _build_report_from_summary(summary, float(resources.get("latency_sec_eval") or elapsed))
    write_report(out / "e0_report.md", report)
    mirror_docs_report("E0", "dense_baseline", report)

    gate = {
        "passed": passed,
        "scores": scores,
        "min_score": args.min_score,
        "ok_nonzero": ok_nonzero,
        "ok_repro": ok_repro,
        "model_path": model_path,
        "limit": 32,
        "seed": 42,
        "elapsed_sec": elapsed,
        "chat_template": False,
        "protocol": "humaneval completion + gen_kwargs",
    }
    write_json(out / "e0_code_gate.json", gate)
    flag = out / ("e0_code_gate_passed.flag" if passed else "e0_code_gate_failed.flag")
    flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    other = out / ("e0_code_gate_failed.flag" if passed else "e0_code_gate_passed.flag")
    if other.is_file():
        other.unlink()

    print(f"[INFO] E0 Code gate passed={passed} scores={scores}")
    if not passed:
        raise SystemExit(1)
    print(f"[OK] E0 Code gate wrote {out / 'e0_code_gate.json'}")


if __name__ == "__main__":
    main()
