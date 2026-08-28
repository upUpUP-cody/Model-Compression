#!/usr/bin/env python3
"""E0 Reasoning-dimension gate: re-eval BBH on Instruct after max_gen_toks=1024.

Does not re-run other dimensions. Updates e0_summary / e0_report Reasoning fields.
Writes e0_reasoning_gate.json + passed/failed flag. Exit 0 when gate passes.

Instruct scores are NOT comparable to E1/E2 base; this gate is documentation /
protocol consistency only (J1). No E1 numeric threshold.
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

INSTRUCT_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _require_instruct_path(model_path: str) -> None:
    if model_path != INSTRUCT_PATH and "Instruct" not in model_path:
        raise SystemExit(
            f"[ERROR] E0 Reasoning gate requires Instruct model path "
            f"(expected {INSTRUCT_PATH}), got {model_path}"
        )
    if "Instruct" not in model_path:
        raise SystemExit(f"[ERROR] E0 Reasoning gate path must contain Instruct: {model_path}")


def _merge_reasoning_into_summary(summary: dict, reason_cap: dict, *, run_id: str, old_score) -> dict:
    score = reason_cap["vector"].get("Reasoning")
    out = copy.deepcopy(summary)
    out.setdefault("outputs", {}).setdefault("P_M0", {})["Reasoning"] = score
    cap = out.setdefault("capability", {})
    vec = dict(cap.get("vector") or {d: None for d in DIM_ORDER})
    vec["Reasoning"] = score
    cap["vector"] = vec
    cap["vector_list"] = [vec.get(d) for d in DIM_ORDER]
    details = dict(cap.get("details") or {})
    details["Reasoning"] = reason_cap.get("details", {}).get("Reasoning")
    cap["details"] = details
    raw = dict(cap.get("raw") or {})
    raw["Reasoning"] = (reason_cap.get("raw") or {}).get("Reasoning")
    cap["raw"] = raw
    protocol = dict(cap.get("protocol") or {})
    protocol.update((reason_cap.get("protocol") or {}))
    cap["protocol"] = protocol
    out["capability"] = cap
    out["reasoning_gate"] = {
        "run_id": run_id,
        "old_score": old_score,
        "score": score,
        "protocol_note": "bbh max_gen_toks=1024; Instruct-only; not comparable to E1 base",
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
    gate = summary.get("reasoning_gate") or {}
    reason_note = ""
    if gate:
        reason_note = (
            f"\n- Reasoning 维已用 BBH max_gen_toks=1024 重评（Instruct；"
            f"old={_fmt(gate.get('old_score'))} new={_fmt(gate.get('score'))}）。"
            f"**不可与 E1 base 对比**。\n"
        )
    code_gate = summary.get("code_gate") or {}
    code_note = ""
    if code_gate:
        code_note = (
            f"\n- Code 维已用修复后的 HumanEval 补全协议重评（gate score={_fmt(code_gate.get('score'))}）。\n"
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
- Gate：E0 为后续阈值基准；不单独触发 Gate A–E；**E0 Instruct 永不进入 Gate A 数值**。
- 旧 SST-2 部分向量产物作废；以本报告为准。{code_note}{reason_note}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e0_dense.yaml"))
    parser.add_argument("--device", default=None, help="override capability.device (e.g. cuda:0)")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    out = ensure_dir(Path(config["logging"]["output_root"]))
    summary_path = out / "e0_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing E0 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    model_path = str(config["model"]["path"])
    _require_instruct_path(model_path)

    run_config = copy.deepcopy(config)
    cap = run_config.setdefault("evaluation", {}).setdefault("capability", {})
    cap["only_dimensions"] = ["Reasoning"]
    if args.device:
        cap["device"] = args.device
        run_config.setdefault("hardware", {})["device"] = args.device

    old_score = None
    try:
        old_score = (summary.get("capability") or {}).get("vector", {}).get("Reasoning")
        if old_score is None:
            old_score = (summary.get("outputs") or {}).get("P_M0", {}).get("Reasoning")
    except Exception:
        old_score = None

    print(f"[INFO] E0 Reasoning gate model={model_path} old_score={old_score}")
    t0 = time.time()
    reason_cap = eval_capability_vector(model_path, run_config)
    score = reason_cap["vector"].get("Reasoning")
    elapsed = time.time() - t0
    mem = gpu_mem_gb()
    print(f"[INFO] E0 Reasoning gate score={score} elapsed={elapsed:.1f}s mem_gb={mem}")

    details = (reason_cap.get("details") or {}).get("Reasoning") or {}
    gk = details.get("gen_kwargs") or {}
    ok_protocol = int(gk.get("max_gen_toks") or 0) == 1024 and gk.get("do_sample") is False
    ok_score = score is not None and float(score) >= 0.0
    passed = bool(ok_protocol and ok_score)

    summary = _merge_reasoning_into_summary(summary, reason_cap, run_id="reasoning_gate", old_score=old_score)
    resources = summary.setdefault("outputs", {}).setdefault("resources", {})
    resources["reasoning_gate_latency_sec"] = elapsed
    resources["reasoning_gate_gpu_memory_gb_peak"] = mem
    write_json(summary_path, summary)

    report = _build_report_from_summary(summary, float(resources.get("latency_sec_eval") or elapsed))
    write_report(out / "e0_report.md", report)
    mirror_docs_report("E0", "dense_baseline", report)

    delta = None
    if old_score is not None and score is not None:
        delta = float(score) - float(old_score)
    gate = {
        "passed": passed,
        "ok_protocol": ok_protocol,
        "ok_score": ok_score,
        "old_score": old_score,
        "score": score,
        "delta": delta,
        "model_path": model_path,
        "limit": 64,
        "seed": 42,
        "batch_size": 1,
        "elapsed_sec": elapsed,
        "gen_kwargs": gk,
        "protocol": "bbh max_gen_toks=1024 Instruct-only (not E1-comparable)",
        "note": "J1: documentation/protocol only; do not use for E1 Delta or Gate A",
    }
    write_json(out / "e0_reasoning_gate.json", gate)
    flag = out / ("e0_reasoning_gate_passed.flag" if passed else "e0_reasoning_gate_failed.flag")
    flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    other = out / ("e0_reasoning_gate_failed.flag" if passed else "e0_reasoning_gate_passed.flag")
    if other.is_file():
        other.unlink()

    print(f"[INFO] E0 Reasoning gate passed={passed} score={score} delta={delta}")
    if not passed:
        raise SystemExit(1)
    print(f"[OK] E0 Reasoning gate wrote {out / 'e0_reasoning_gate.json'}")


if __name__ == "__main__":
    main()
