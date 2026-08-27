from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.probes.train_sao_secrlenv import bind_context, load_context


def test_generic_v2_context_binds_terminal_bench_without_fake_secrlenv_fields(
    tmp_path,
):
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[],"metadata":{"task_id":"fix-git"}}\n')
    context = {
        "schema": "miles.sao-runtime.v2",
        "benchmark": "terminal-bench-2.1",
        "model": "Qwen/Qwen3.5-0.8B",
        "data": str(data),
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "base_model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "rollout_model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "data_revision": "terminal-bench-2.1@7131e437",
        "layout_hash": "1" * 64,
        "lora_config_hash": "full-parameter",
        "reward_sha256": "2" * 64,
        "dynamic_sampling_max_replacements": 0,
        "completed_groups_path": str(tmp_path / "completed.json"),
        "event_tape": str(tmp_path / "events.jsonl"),
        "learner_id": 0,
    }
    source = tmp_path / "sao-context.json"
    source.write_text(json.dumps(context, sort_keys=True, separators=(",", ":")))
    source.chmod(0o600)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    loaded = load_context(source, digest)
    args = SimpleNamespace(use_wandb=False)
    bind_context(args, loaded)

    assert args.yeto_rl_benchmark == "terminal-bench-2.1"
    assert args.yeto_rl_model == "Qwen/Qwen3.5-0.8B"
    assert not hasattr(args, "yeto_rl_secrlenv_max_infrastructure_replacements")


def test_context_source_rejects_symlink_nonregular_and_public_files(tmp_path):
    source = tmp_path / "context.json"
    source.write_text("{}")
    source.chmod(0o600)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    symlink = tmp_path / "context-link.json"
    symlink.symlink_to(source)

    with pytest.raises(ValueError, match="private regular non-symlink"):
        load_context(symlink, digest)
    with pytest.raises(ValueError, match="private regular non-symlink"):
        load_context(tmp_path, digest)

    source.chmod(0o644)
    with pytest.raises(ValueError, match="private regular non-symlink"):
        load_context(source, digest)
