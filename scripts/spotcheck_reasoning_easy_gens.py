#!/usr/bin/env python3
"""Spotcheck Reasoning_easy generations at dense/10%/20%/30% (format collapse vs noise).

Not Gate A. Writes /mnt/data2/results/E1_soft_eval_side/reasoning_easy_gen_spotcheck.md
"""
from __future__ import annotations

import argparse
import copy
import gc
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.stage_a_common import (
    load_model,
    load_yaml,
    prepare_wanda_calibration_loader,
    stage_a_prune_mlp,
    write_json,
)

TASK = "bbh_fewshot_boolean_expressions_easy"
TASK_INCLUDE = ROOT / "configs/lm_eval_tasks/easy"
TRUE_FALSE_RE = re.compile(r"\b(True|False)\b", re.IGNORECASE)
OUT_DEFAULT = Path("/mnt/data2/results/E1_soft_eval_side/reasoning_easy_gen_spotcheck.md")
JSON_DEFAULT = Path("/mnt/data2/results/E1_soft_eval_side/reasoning_easy_gen_spotcheck.json")
JsonDict = Dict[str, Any]


def _cuda_oom_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def _flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        parts = [_flatten_text(x) for x in obj]
        return "\n".join(p for p in parts if p)
    return str(obj)


def _classify_resp(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    if TRUE_FALSE_RE.search(t):
        return "has_true_false"
    if len(t) >= 240:
        return "long_no_tf"
    return "other_no_tf"


def _extract_samples(out: JsonDict) -> List[Dict[str, Any]]:
    samples_blob = out.get("samples") or {}
    rows: List[Dict[str, Any]] = []
    for task_name, items in samples_blob.items():
        if not isinstance(items, list):
            continue
        for i, s in enumerate(items):
            if not isinstance(s, dict):
                continue
            resp = _flatten_text(s.get("resps") or s.get("filtered_resps") or "")
            doc = s.get("doc") if isinstance(s.get("doc"), dict) else {}
            raw_target = s.get("target")
            if raw_target is None and isinstance(doc, dict):
                raw_target = doc.get("target")
            target = _flatten_text(raw_target)
            filtered = _flatten_text(s.get("filtered_resps") or "")
            rows.append(
                {
                    "task": task_name,
                    "idx": i,
                    "target": target[:80],
                    "resp_preview": resp[:400].replace("\n", "\\n"),
                    "resp_len": len(resp),
                    "filtered_preview": filtered[:120].replace("\n", "\\n"),
                    "class": _classify_resp(resp),
                    "exact_match": s.get("exact_match"),
                }
            )
    return rows


def _run_one(
    model,
    tokenizer,
    model_path: str,
    device: str,
    *,
    limit: int,
    max_gen_toks: int,
    seed: int,
    batch_size: int,
    until: Optional[List[str]] = None,
) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    stop = until or ["\n", "Q:", "Q ", "</s>", "<|im_end|>", "<|endoftext|>"]
    gen_kwargs = {
        "max_gen_toks": max_gen_toks,
        "do_sample": False,
        "until": list(stop),
    }
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        dtype="float16",
        batch_size=batch_size,
        trust_remote_code=True,
    )
    task_manager = TaskManager(include_path=str(TASK_INCLUDE))
    try:
        out = simple_evaluate(
            model=lm,
            tasks=[TASK],
            num_fewshot=None,
            batch_size=batch_size,
            device=device,
            limit=limit,
            bootstrap_iters=0,
            log_samples=True,
            apply_chat_template=False,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            gen_kwargs=gen_kwargs,
            task_manager=task_manager,
        )
    finally:
        del lm

    if out is None:
        return None, [], {"error": "simple_evaluate returned None"}

    table = out.get("results") or {}
    score = None
    metric_key = None
    for key in (
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "exact_match",
        "acc,none",
        "acc",
    ):
        for _tn, payload in table.items():
            if isinstance(payload, dict) and key in payload and payload[key] is not None:
                try:
                    score = float(payload[key])
                    metric_key = key
                    break
                except (TypeError, ValueError):
                    pass
        if score is not None:
            break
    if score is None:
        # average leaf metrics
        vals = []
        for payload in table.values():
            if not isinstance(payload, dict):
                continue
            for k, v in payload.items():
                if "stderr" in k or k in ("alias", "name", "sample_len"):
                    continue
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                    break
        if vals:
            score = sum(vals) / len(vals)

    samples = _extract_samples(out)
    meta = {
        "metric_key": metric_key,
        "n_samples": len(samples),
        "results_keys": list(table.keys())[:8],
    }
    return score, samples, meta


def _class_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        c = str(s.get("class") or "unknown")
        counts[c] = counts.get(c, 0) + 1
    return counts


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def build_report(
    rows: List[Dict[str, Any]],
    *,
    limit: int,
    max_gen_toks: int,
) -> str:
    lines = [
        "# Reasoning_easy generation spotcheck",
        "",
        f"- Task: `{TASK}`",
        f"- Protocol: apply_chat_template=False, max_gen_toks={max_gen_toks}, until=short-stop, flexible-extract, seed=42",
        f"- Limit per sparsity: **{limit}** (diagnostic; ladder used 32)",
        "- Purpose: check whether 10%/30% zeros are format/extract collapse (class B), not 1-item noise",
        "",
        "## Summary",
        "",
        "| Sparsity | score@limit | n | has_true_false | empty | long_no_tf | other_no_tf |",
        "|----------|-------------|---|---------------|-------|------------|-------------|",
    ]
    for r in rows:
        cc = r["class_counts"]
        lines.append(
            f"| {r['label']} | {r['score'] if r['score'] is not None else 'n/a'} | {r['n']} | "
            f"{cc.get('has_true_false', 0)} | {cc.get('empty', 0)} | "
            f"{cc.get('long_no_tf', 0)} | {cc.get('other_no_tf', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
        ]
    )

    by_lab = {r["label"]: r for r in rows}
    dense = by_lab.get("dense")
    s10 = by_lab.get("0.10")
    s20 = by_lab.get("0.20")
    s30 = by_lab.get("0.30")

    verdict_bits: List[str] = []
    if dense and s10:
        d_tf = dense["class_counts"].get("has_true_false", 0)
        z_tf = s10["class_counts"].get("has_true_false", 0)
        if d_tf > 0 and z_tf == 0:
            verdict_bits.append(
                f"At 10%, **0/{s10['n']}** responses contain extractable True/False "
                f"(dense had {d_tf}/{dense['n']}) — consistent with **format/extract collapse**, "
                "not binomial noise of 1-2 items."
            )
        elif d_tf > 0 and z_tf < d_tf // 2:
            verdict_bits.append(
                f"At 10%, True/False-bearing gens drop sharply "
                f"({z_tf}/{s10['n']} vs dense {d_tf}/{dense['n']}) — format fragility dominates."
            )
    if s20 and s10:
        t10 = s10["class_counts"].get("has_true_false", 0)
        t20 = s20["class_counts"].get("has_true_false", 0)
        if t20 > t10:
            verdict_bits.append(
                f"At 20%, extractable True/False **recovers** to {t20}/{s20['n']} "
                f"(from {t10}/{s10['n']} at 10%) — matches ladder non-monotonic +37.5pp."
            )
    if s30:
        t30 = s30["class_counts"].get("has_true_false", 0)
        if t30 == 0:
            verdict_bits.append(
                f"At 30%, again **0/{s30['n']}** True/False-bearing — second all-or-nothing cliff."
            )
    if not verdict_bits:
        verdict_bits.append(
            "Spotcheck did not show a clean empty/format split; see per-sample previews below. "
            "Ladder zeros may still be extract-metric brittle; treat Reasoning_easy as ruler-sensitive."
        )
    for b in verdict_bits:
        lines.append(f"- {b}")

    lines.extend(["", "## Sample previews (first 4 per sparsity)", ""])
    for r in rows:
        lines.append(f"### {r['label']} (score={r['score']})")
        lines.append("")
        for s in r["samples"][:4]:
            lines.append(
                f"- class=`{s['class']}` len={s['resp_len']} target=`{_md_escape(str(s.get('target') or ''))}`"
            )
            lines.append(f"  - resp: `{_md_escape(s['resp_preview'][:300])}`")
        lines.append("")

    lines.extend(
        [
            "## Note",
            "",
            "This is a side diagnostic only; does not change Gate A / formal E1.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage_a/e1_soft_eval_side.yaml"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-gen-toks", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sparsities",
        default="0,0.10,0.20,0.30",
        help="comma-separated sparsities",
    )
    parser.add_argument("--out-md", default=str(OUT_DEFAULT))
    parser.add_argument("--out-json", default=str(JSON_DEFAULT))
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    grid = [float(x.strip()) for x in args.sparsities.split(",") if x.strip()]
    model_path = str(config["model"]["path"])

    print(f"[INFO] spotcheck Reasoning_easy limit={args.limit} grid={grid}", flush=True)
    parent, tokenizer, device = load_model(config)
    calib_loader = None
    if any(abs(s) > 1e-12 for s in grid):
        calib_loader = prepare_wanda_calibration_loader(config, tokenizer, device)

    rows: List[Dict[str, Any]] = []
    for sp in grid:
        label = "dense" if abs(sp) < 1e-12 else f"{sp:.2f}"
        print(f"[INFO] sparsity={label}", flush=True)
        if abs(sp) < 1e-12:
            child = parent
        else:
            parent.to("cpu")
            _cuda_oom_cleanup()
            child = stage_a_prune_mlp(
                copy.deepcopy(parent),
                sp,
                config,
                calib_loader=calib_loader,
                device=device,
            ).to(device)

        score, samples, meta = _run_one(
            child,
            tokenizer,
            model_path,
            device,
            limit=int(args.limit),
            max_gen_toks=int(args.max_gen_toks),
            seed=int(config.get("seed", 42)),
            batch_size=int(args.batch_size),
        )
        cc = _class_counts(samples)
        print(f"[OK] {label} score={score} classes={cc}", flush=True)
        rows.append(
            {
                "label": label,
                "sparsity": float(sp),
                "score": score,
                "n": len(samples),
                "class_counts": cc,
                "samples": samples,
                "meta": meta,
            }
        )

        if child is not parent:
            del child
            _cuda_oom_cleanup()
            parent.to(device)
            _cuda_oom_cleanup()

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(rows, limit=int(args.limit), max_gen_toks=int(args.max_gen_toks))
    out_md.write_text(report, encoding="utf-8")
    # compact json (truncate long previews already)
    write_json(
        out_json,
        {
            "task": TASK,
            "limit": int(args.limit),
            "max_gen_toks": int(args.max_gen_toks),
            "rows": [
                {
                    "label": r["label"],
                    "sparsity": r["sparsity"],
                    "score": r["score"],
                    "n": r["n"],
                    "class_counts": r["class_counts"],
                    "meta": r["meta"],
                    "samples": r["samples"],
                }
                for r in rows
            ],
        },
    )
    print(f"[OK] wrote {out_md}", flush=True)
    print(f"[OK] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
