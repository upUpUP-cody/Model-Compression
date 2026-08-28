"""E1 checkpoint resume helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.capability import resolve_capability_config
from src.experiments.stage_a_common import (
    E1_CHECKPOINT_VERSION,
    build_e1_checkpoint_payload,
    dense_capability_complete,
    e1_config_digest,
    e1_config_digest_legacy,
    e1_partial_dims_complete,
    e1_required_dimensions,
    filter_e1_sparsity_grid,
    finalize_e1_partial_capability,
    import_e1_checkpoint_seed,
    load_e1_checkpoint,
    merge_capability_dim_result,
    merge_e1_curves,
    merge_e1_main_with_shard,
    new_e1_partial,
    save_e1_checkpoint,
    sparsity_in_completed,
)


def _minimal_config():
    return {
        "model": {"path": "/mnt/data/models/Qwen2.5-3B", "spec": "formal_3B_base"},
        "sparsity_grid": [0.1, 0.2, 0.3],
        "pruning": {"method": "wanda", "wanda_batches": 4},
        "evaluation": {"capability": {"mode": "scan", "seed": 42}},
        "seed": 42,
        "recovery": "none",
    }


def test_e1_config_digest_stable():
    cfg = _minimal_config()
    assert e1_config_digest(cfg, smoke=False) == e1_config_digest(cfg, smoke=False)
    assert e1_config_digest(cfg, smoke=False) != e1_config_digest(cfg, smoke=True)


def test_dense_capability_complete():
    incomplete = {"vector": {"PPL": 1.0, "Math": None}}
    complete = {
        "vector": {
            "PPL": 1.0,
            "Math": 1.0,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    assert not dense_capability_complete(incomplete)
    assert dense_capability_complete(complete)

    skip_code = {
        "vector": {
            "PPL": 1.0,
            "Math": 1.0,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": None,
        }
    }
    assert not dense_capability_complete(skip_code)
    assert dense_capability_complete(skip_code, skip_dimensions=("Code",))


def test_e1_digest_ignores_skip_dimensions():
    cfg = _minimal_config()
    cfg_skip = {
        **cfg,
        "evaluation": {
            "capability": {
                "mode": "scan",
                "seed": 42,
                "skip_dimensions": ["Code"],
            }
        },
    }
    assert e1_config_digest(cfg, smoke=False) == e1_config_digest(cfg_skip, smoke=False)


def test_e1_digest_ignores_only_dimensions():
    cfg = _minimal_config()
    cfg_only = {
        **cfg,
        "evaluation": {
            "capability": {
                "mode": "scan",
                "seed": 42,
                "only_dimensions": ["Code"],
            }
        },
    }
    assert e1_config_digest(cfg, smoke=False) == e1_config_digest(cfg_only, smoke=False)


def test_resolve_skip_dimensions():
    cfg = {
        "evaluation": {"capability": {"mode": "scan", "skip_dimensions": ["Code", "Instruction"]}},
        "seed": 42,
    }
    resolved = resolve_capability_config(cfg)
    assert resolved["skip_dimensions"] == ("Instruction", "Code")
    with pytest.raises(ValueError, match="not in DIM_ORDER"):
        resolve_capability_config(
            {"evaluation": {"capability": {"skip_dimensions": ["Nope"]}}}
        )


def test_resolve_only_dimensions_and_code_defaults():
    cfg = {
        "evaluation": {"capability": {"mode": "scan", "only_dimensions": ["Code"]}},
        "seed": 42,
    }
    resolved = resolve_capability_config(cfg)
    assert resolved["only_dimensions"] == ("Code",)
    code = resolved["dimensions"]["Code"]
    assert code["apply_chat_template"] is False
    assert code["gen_kwargs"]["max_gen_toks"] == 512
    assert "\ndef" in code["gen_kwargs"]["until"]
    with pytest.raises(ValueError, match="not in DIM_ORDER"):
        resolve_capability_config(
            {"evaluation": {"capability": {"only_dimensions": ["Nope"]}}}
        )


def test_code_patch_must_clear_skip_when_only_code():
    """Shard YAML skip_dimensions:[Code] + only_dimensions:[Code] still skips Code.

    run_e1_code_patch clears skip_dimensions so Code actually runs.
    """
    # Broken: only ∩ skip keeps Code in skip set.
    both = {
        "evaluation": {
            "capability": {
                "mode": "scan",
                "skip_dimensions": ["Code"],
                "only_dimensions": ["Code"],
            }
        },
        "seed": 42,
    }
    broken = resolve_capability_config(both)
    assert broken["only_dimensions"] == ("Code",)
    assert broken["skip_dimensions"] == ("Code",)

    # Fixed (patch runner behavior): clear skip.
    fixed_cfg = {
        "evaluation": {
            "capability": {
                "mode": "scan",
                "skip_dimensions": [],
                "only_dimensions": ["Code"],
            }
        },
        "seed": 42,
    }
    fixed = resolve_capability_config(fixed_cfg)
    assert fixed["only_dimensions"] == ("Code",)
    assert fixed["skip_dimensions"] == ()
    assert fixed["dimensions"]["Code"]["apply_chat_template"] is False


def test_sparsity_in_completed():
    assert sparsity_in_completed(0.3, [0.1, 0.2, 0.3])
    assert not sparsity_in_completed(0.4, [0.1, 0.2, 0.3])


def test_checkpoint_roundtrip_and_digest_mismatch(tmp_path: Path):
    cfg = _minimal_config()
    digest = e1_config_digest(cfg, smoke=False)
    ckpt_path = tmp_path / "e1_checkpoint.json"
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    curve = [{"sparsity": 0.1, "vector": {}, "delta": {}, "params": 1}]
    payload = build_e1_checkpoint_payload(
        config_digest=digest,
        smoke=False,
        dense_capability=dense,
        curve=curve,
        grid=[0.1, 0.2, 0.3],
        started_at=100.0,
        elapsed_sec=50.0,
    )
    save_e1_checkpoint(ckpt_path, payload)
    loaded = load_e1_checkpoint(ckpt_path, digest)
    assert loaded is not None
    assert loaded["version"] == E1_CHECKPOINT_VERSION
    assert loaded["completed_sparsities"] == [0.1]
    assert loaded["pending_sparsities"] == [0.2, 0.3]

    with pytest.raises(ValueError, match="config mismatch"):
        load_e1_checkpoint(ckpt_path, "wrong-digest")


def test_load_missing_checkpoint_returns_none(tmp_path: Path):
    assert load_e1_checkpoint(tmp_path / "missing.json", "abc") is None


def test_e1_digest_ignores_batch_size():
    cfg = _minimal_config()
    cfg["evaluation"]["capability"]["batch_size"] = 1
    cfg_b2 = {
        **cfg,
        "evaluation": {
            "capability": {
                "mode": "scan",
                "seed": 42,
                "batch_size": 2,
            }
        },
    }
    assert e1_config_digest(cfg, smoke=False) == e1_config_digest(cfg_b2, smoke=False)
    assert e1_config_digest(cfg, smoke=False) != e1_config_digest_legacy(cfg, smoke=False)


def test_load_checkpoint_accepts_legacy_digest(tmp_path: Path):
    cfg = _minimal_config()
    legacy_digest = e1_config_digest_legacy(cfg, smoke=False)
    ckpt_path = tmp_path / "e1_checkpoint.json"
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    payload = build_e1_checkpoint_payload(
        config_digest=legacy_digest,
        smoke=False,
        dense_capability=dense,
        curve=[],
        grid=[0.1, 0.2, 0.3],
        started_at=1.0,
        elapsed_sec=1.0,
    )
    save_e1_checkpoint(ckpt_path, payload)
    loaded = load_e1_checkpoint(
        ckpt_path,
        e1_config_digest(cfg, smoke=False),
        alternate_digests=[legacy_digest],
    )
    assert loaded is not None


def test_filter_e1_sparsity_grid():
    grid = [0.1, 0.2, 0.3, 0.4]
    assert filter_e1_sparsity_grid(grid, [0.2, 0.4]) == [0.2, 0.4]
    with pytest.raises(ValueError, match="not in config"):
        filter_e1_sparsity_grid(grid, [0.5])


def test_merge_e1_curves_disjoint():
    a = [{"sparsity": 0.1, "vector": {}}]
    b = [{"sparsity": 0.2, "vector": {}}]
    merged = merge_e1_curves(a, b)
    assert [r["sparsity"] for r in merged] == [0.1, 0.2]
    with pytest.raises(ValueError, match="duplicate"):
        merge_e1_curves(a, a)


def test_merge_e1_curves_shard_novel_only():
    main = [{"sparsity": 0.1, "vector": {}}, {"sparsity": 0.2, "vector": {}}]
    shard = [
        {"sparsity": 0.1, "vector": {}},
        {"sparsity": 0.6, "vector": {}},
        {"sparsity": 0.7, "vector": {}},
    ]
    novel = [r for r in shard if float(r["sparsity"]) not in {float(x["sparsity"]) for x in main}]
    merged = merge_e1_curves(main, novel)
    assert [r["sparsity"] for r in merged] == [0.1, 0.2, 0.6, 0.7]


def test_merge_e1_main_with_shard_overwrite():
    main = [
        {"sparsity": 0.1, "vector": {"Reasoning": 0.1}},
        {"sparsity": 0.5, "vector": {"Reasoning": 0.0}},
    ]
    shard = [
        {"sparsity": 0.5, "vector": {"Reasoning": 0.9}},
        {"sparsity": 0.7, "vector": {"Reasoning": 0.7}},
    ]
    merged = merge_e1_main_with_shard(main, shard, overwrite_sparsities=[0.5, 0.6, 0.7])
    by_sp = {float(r["sparsity"]): r for r in merged}
    assert by_sp[0.1]["vector"]["Reasoning"] == 0.1
    assert by_sp[0.5]["vector"]["Reasoning"] == 0.9
    assert by_sp[0.7]["vector"]["Reasoning"] == 0.7
    assert [float(r["sparsity"]) for r in merged] == [0.1, 0.5, 0.7]


def test_import_e1_checkpoint_seed(tmp_path: Path):
    cfg = _minimal_config()
    digest = e1_config_digest_legacy(cfg, smoke=False)
    ckpt_path = tmp_path / "e1_checkpoint.json"
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    curve = [{"sparsity": 0.1, "vector": {}, "delta": {}}]
    payload = build_e1_checkpoint_payload(
        config_digest=digest,
        smoke=False,
        dense_capability=dense,
        curve=curve,
        grid=[0.1, 0.2, 0.3],
        started_at=1.0,
        elapsed_sec=1.0,
    )
    save_e1_checkpoint(ckpt_path, payload)
    dense_out, rows = import_e1_checkpoint_seed(ckpt_path, alternate_digests=[digest])
    assert dense_capability_complete(dense_out)
    assert rows[0]["sparsity"] == 0.1


def test_e1_required_dimensions_respects_skip():
    assert e1_required_dimensions() == (
        "PPL",
        "Math",
        "Knowledge",
        "Reasoning",
        "Instruction",
        "Code",
    )
    assert e1_required_dimensions(skip_dimensions=("Code",)) == (
        "PPL",
        "Math",
        "Knowledge",
        "Reasoning",
        "Instruction",
    )


def test_partial_roundtrip_and_legacy_without_partial(tmp_path: Path):
    cfg = _minimal_config()
    digest = e1_config_digest(cfg, smoke=False)
    ckpt_path = tmp_path / "e1_checkpoint.json"
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    partial = new_e1_partial(0.7)
    partial = merge_capability_dim_result(
        partial,
        "PPL",
        {"vector": {"PPL": 20.0}, "details": {"PPL": {"score": 20.0}}, "raw": {}},
    )
    partial = merge_capability_dim_result(
        partial,
        "Math",
        {"vector": {"Math": 0.1}, "details": {"Math": {"score": 0.1}}, "raw": {}},
    )
    payload = build_e1_checkpoint_payload(
        config_digest=digest,
        smoke=False,
        dense_capability=dense,
        curve=[{"sparsity": 0.5, "vector": {}}],
        grid=[0.5, 0.6, 0.7],
        started_at=1.0,
        elapsed_sec=2.0,
        partial=partial,
    )
    save_e1_checkpoint(ckpt_path, payload)
    loaded = load_e1_checkpoint(ckpt_path, digest)
    assert loaded is not None
    assert loaded["partial"]["sparsity"] == 0.7
    assert loaded["partial"]["completed_dimensions"] == ["PPL", "Math"]
    assert loaded["partial"]["vector"]["PPL"] == 20.0
    assert loaded["completed_sparsities"] == [0.5]

    # Legacy payload without partial still loads.
    legacy = build_e1_checkpoint_payload(
        config_digest=digest,
        smoke=False,
        dense_capability=dense,
        curve=[],
        grid=[0.1],
        started_at=1.0,
        elapsed_sec=1.0,
    )
    assert "partial" not in legacy
    save_e1_checkpoint(ckpt_path, legacy)
    loaded2 = load_e1_checkpoint(ckpt_path, digest)
    assert loaded2 is not None
    assert loaded2.get("partial") is None


def test_e1_partial_dims_complete_and_finalize():
    partial = new_e1_partial(0.7)
    assert not e1_partial_dims_complete(partial)
    assert not e1_partial_dims_complete(None)

    for dim, score in [
        ("PPL", 12.0),
        ("Math", 0.2),
        ("Knowledge", 0.3),
        ("Reasoning", 0.4),
        ("Instruction", 0.1),
        ("Code", 0.0),
    ]:
        partial = merge_capability_dim_result(
            partial,
            dim,
            {"vector": {dim: score}, "details": {dim: {"score": score}}, "raw": {dim: {}}},
        )
    assert e1_partial_dims_complete(partial)
    assert e1_partial_dims_complete(
        {**partial, "vector": {**partial["vector"], "Code": None}},
        skip_dimensions=("Code",),
    )
    assert not e1_partial_dims_complete(
        {**partial, "vector": {**partial["vector"], "Math": None}},
    )

    cap = finalize_e1_partial_capability(partial, model_path="/m", seed=42)
    assert cap["vector"]["PPL"] == 12.0
    assert cap["vector"]["Code"] == 0.0
    assert cap["vector_list"][0] == 12.0
    assert set(cap["details"]) == {
        "PPL",
        "Math",
        "Knowledge",
        "Reasoning",
        "Instruction",
        "Code",
    }


def test_partial_promote_clears_when_building_done_payload():
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.0,
        }
    }
    partial = new_e1_partial(0.7)
    for dim in ("PPL", "Math"):
        partial = merge_capability_dim_result(
            partial, dim, {"vector": {dim: 1.0}, "details": {}, "raw": {}}
        )
    assert not e1_partial_dims_complete(partial)

    # After promoting a finished row, final payload omits partial.
    curve = [{"sparsity": 0.7, "vector": partial["vector"]}]
    done_payload = build_e1_checkpoint_payload(
        config_digest="x",
        smoke=False,
        dense_capability=dense,
        curve=curve,
        grid=[0.7],
        started_at=1.0,
        elapsed_sec=1.0,
        partial=None,
    )
    assert "partial" not in done_payload
    assert done_payload["completed_sparsities"] == [0.7]


def test_code_patch_payload_preserves_live_partial():
    """Simulates GPU0 Code write while GPU1 holds sparsity-0.7 partial."""
    dense = {
        "vector": {
            "PPL": 10.0,
            "Math": 0.5,
            "Knowledge": 0.5,
            "Reasoning": 0.5,
            "Instruction": 0.5,
            "Code": 0.6875,
        }
    }
    curve = [
        {
            "sparsity": 0.5,
            "vector": {
                "PPL": 20.0,
                "Math": 0.4,
                "Knowledge": 0.4,
                "Reasoning": 0.4,
                "Instruction": 0.2,
                "Code": 0.5,
            },
        }
    ]
    partial = new_e1_partial(0.7)
    partial = merge_capability_dim_result(
        partial, "PPL", {"vector": {"PPL": 200.0}, "details": {}, "raw": {}}
    )
    payload = build_e1_checkpoint_payload(
        config_digest="x",
        smoke=False,
        dense_capability=dense,
        curve=curve,
        grid=[0.5, 0.6, 0.7],
        started_at=1.0,
        elapsed_sec=1.0,
        status="in_progress",
        partial=partial,
    )
    assert payload["partial"]["completed_dimensions"] == ["PPL"]
    assert payload["partial"]["vector"]["PPL"] == 200.0
    assert payload["pending_sparsities"] == [0.6, 0.7]
