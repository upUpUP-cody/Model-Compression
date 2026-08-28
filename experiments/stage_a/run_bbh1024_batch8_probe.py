#!/usr/bin/env python3
"""BBH1024 batch=8 VRAM probe (limit_override=32). Does not modify E1/E2 checkpoints.

Writes /mnt/data2/results/bbh_maxtok1024_calib/bbh1024_batch8_probe.json
Exit 0 always (probe result in JSON); check payload["passed"].
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.evaluation.capability import eval_capability_vector
from src.experiments.stage_a_common import ensure_dir, gpu_mem_gb, load_yaml
from src.utils.qwen_squad_eval import write_json

OUT_DIR = Path("/mnt/data2/results/bbh_maxtok1024_calib")
PEAK_GB_LIMIT = 22.0
CFG = ROOT / "configs/stage_a/e1_oneshot_curve.yaml"


def main() -> None:
    ensure_dir(OUT_DIR)
    config = load_yaml(CFG)
    model_path = str(config["model"]["path"])
    run = copy.deepcopy(config)
    cap = run.setdefault("evaluation", {}).setdefault("capability", {})
    cap["only_dimensions"] = ["Reasoning"]
    cap["skip_dimensions"] = []
    cap["batch_size"] = 8
    cap["limit_override"] = 32
    dim = cap.setdefault("dimensions", {}).setdefault("Reasoning", {})
    gk = dict(dim.get("gen_kwargs") or {})
    gk["max_gen_toks"] = 1024
    gk["do_sample"] = False
    dim["gen_kwargs"] = gk
    run.setdefault("hardware", {})["batch_size"] = 8
    run["hardware"]["device"] = "cuda:0"

    payload: Dict[str, Any] = {
        "experiment_id": "BBH1024_BATCH8_PROBE",
        "batch_size": 8,
        "limit_override": 32,
        "max_gen_toks": 1024,
        "do_sample": False,
        "model_path": model_path,
        "oom": False,
        "score": None,
        "wall_sec": None,
        "gpu_mem_gb_peak": None,
        "passed": False,
        "note": None,
    }
    print(
        f"[INFO] BBH1024 batch=8 probe limit_override=32 model={model_path}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    t0 = time.time()
    try:
        cap_out = eval_capability_vector(model_path, run)
        payload["score"] = (cap_out.get("vector") or {}).get("Reasoning")
        payload["wall_sec"] = time.time() - t0
        mem = gpu_mem_gb()
        payload["gpu_mem_gb_peak"] = mem
        if mem is not None and float(mem) > PEAK_GB_LIMIT:
            payload["note"] = f"peak {mem:.2f}GB > {PEAK_GB_LIMIT}GB limit"
        else:
            payload["passed"] = True
            payload["note"] = (
                f"peak {mem:.2f}GB <= {PEAK_GB_LIMIT}GB" if mem is not None else "ok"
            )
    except torch.cuda.OutOfMemoryError:
        payload["oom"] = True
        payload["wall_sec"] = time.time() - t0
        payload["note"] = "torch.cuda.OutOfMemoryError during batch=8 probe"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            payload["oom"] = True
            payload["wall_sec"] = time.time() - t0
            payload["note"] = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    payload["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = OUT_DIR / "bbh1024_batch8_probe.json"
    write_json(out, payload)
    print(
        f"[INFO] batch8_probe oom={payload['oom']} peak={payload['gpu_mem_gb_peak']} "
        f"passed={payload['passed']} note={payload['note']}",
        flush=True,
    )
    print(f"[OK] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
