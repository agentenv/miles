from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest

from tools.probes import run_tbench21_sao_streaming_island as launcher


def _write_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _rows(island_id: int) -> list[dict[str, object]]:
    return [
        {
            "messages": [{"role": "system", "content": "solve"}],
            "metadata": {
                "task_id": f"task-{index}",
                "prompt_tier": "l2",
                "max_seq_len": 8192,
                "split": "train",
                "rollout_replica": index % 4,
                "sample_id": f"train:task-{index}:r{index % 4}",
                "island_id": island_id,
                "rollout_seed": 82621 + island_id,
                "episode_timeout_seconds": 1800,
            },
        }
        for index in range(22)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_train_shard_requires_exact_unique_22_row_contract(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, _rows(3))

    assert len(launcher._load_train_rows(source, 3)) == 22

    bad = _rows(3)
    bad[-1]["metadata"]["task_id"] = bad[0]["metadata"]["task_id"]
    bad[-1]["metadata"]["sample_id"] = "train:task-0:r1"
    _write_jsonl(source, bad)
    with pytest.raises(launcher.LaunchContractError, match="22 unique"):
        launcher._load_train_rows(source, 3)


def test_preflight_binds_v2_contexts_hmac_and_fresh_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    island_id = 3
    roots = {}
    for name in ("hf", "actor", "critic", "miles", "yeto", "megatron"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    for name in ("actor", "critic"):
        (roots[name] / "latest_checkpointed_iteration.txt").write_text("1\n")
    critic_contract = roots["critic"] / "value_pretrain_contract.json"
    critic_contract.write_text("{}\n")
    critic_contract_sha = hashlib.sha256(critic_contract.read_bytes()).hexdigest()

    prompt = tmp_path / "plan" / "train" / f"island-{island_id}.jsonl"
    prompt_sha = _write_jsonl(prompt, _rows(island_id))
    run_root = tmp_path / "runs" / f"island-{island_id}"
    actor_layout = "1" * 64
    critic_layout = "2" * 64
    sao = {
        "schema": "miles.sao-runtime.v2",
        "benchmark": "terminal-bench-2.1",
        "model": launcher.MODEL,
        "data": str(prompt),
        "data_sha256": prompt_sha,
        "base_model_revision": launcher.MODEL_REVISION,
        "rollout_model_revision": launcher.MODEL_REVISION,
        "reward_sha256": launcher._sha256(launcher.SCRIPT_DIR / "openenv_generate.py"),
        "dynamic_sampling_max_replacements": 0,
        "learner_id": island_id,
        "layout_hash": actor_layout,
        "completed_groups_path": str(run_root / "completed-groups.pt"),
        "event_tape": str(run_root / "events.jsonl"),
    }
    sao_path = tmp_path / "contracts" / f"island-{island_id}" / "sao.json"
    sao_sha = _write_json(sao_path, sao)

    def role(name: str, port: int, layout: str, optimizer_steps: int) -> dict[str, object]:
        return {
            "role": name,
            "component": {
                "model_revision": launcher.MODEL_REVISION,
                "config_sha256": "3" * 64,
            },
            "syncer": {"host": "host.docker.internal", "port": port},
            "learner_id": island_id,
            "learner_generation": 0,
            "learner_generations": [0] * 8,
            "local_horizon": 1,
            "optimizer_steps_per_round": optimizer_steps,
            "expected_fragments": 2,
            "total_fragment_steps": 2,
            "parameter_layout_sha256": layout,
            "training_contract_sha256": "4" * 64,
        }

    streaming = {
        "schema": "yeto.sao-streaming-runtime.v2",
        "sao_context_sha256": sao_sha,
        "trajectory_evidence": {
            "directory": str(run_root / "trajectory-evidence"),
            "kind": "terminal-bench-2.1",
            "schema_version": 2,
        },
        "syncer_profile": {"learners": 8, "quorum": 8, "pipeline": 2, "total_steps": 2},
        "actor": role("actor", 29400, actor_layout, 1),
        "critic": role("critic", 29401, critic_layout, 2),
    }
    streaming_path = sao_path.with_name("streaming.json")
    streaming_sha = _write_json(streaming_path, streaming)
    hmac = tmp_path / "tbench.key"
    hmac.write_bytes(b"k" * 32)
    hmac.chmod(0o600)

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.delenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", raising=False)
    monkeypatch.delenv("TBENCH_REWARD_HMAC_KEY", raising=False)
    monkeypatch.setattr(launcher, "_validate_one_h200", lambda: None)
    monkeypatch.setattr(launcher, "_codex_env", lambda _args: {"codex": "exact"})
    monkeypatch.setattr(
        launcher,
        "_validated_openenv_python",
        lambda _value: tmp_path / "openenv-python",
    )
    args = launcher.ScriptArgs(
        island_id=island_id,
        prompt_data=str(prompt),
        sao_context=str(sao_path),
        sao_context_sha256=sao_sha,
        streaming_context=str(streaming_path),
        streaming_context_sha256=streaming_sha,
        run_root=str(run_root),
        hf_checkpoint=str(roots["hf"]),
        ref_load=str(roots["actor"]),
        critic_load=str(roots["critic"]),
        critic_contract_sha256=critic_contract_sha,
        tbench_hmac_key_file=str(hmac),
        miles_root=str(roots["miles"]),
        yeto_root=str(roots["yeto"]),
        megatron_path=str(roots["megatron"]),
    )

    prepared = launcher._preflight(args)

    assert len(prepared.rows) == 22
    assert prepared.actor_save == run_root / "actor-checkpoint"
    assert prepared.critic_save == run_root / "critic-checkpoint"
    assert prepared.codex_env == {"codex": "exact"}


def test_execute_emits_exact_online_island_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = launcher._Prepared(
        rows=tuple({} for _ in range(22)),
        prompt_path=tmp_path / "train.jsonl",
        sao_context_path=tmp_path / "sao.json",
        streaming_context_path=tmp_path / "streaming.json",
        run_root=tmp_path / "run",
        dump_details=tmp_path / "run" / "details",
        actor_save=tmp_path / "run" / "actor",
        critic_save=tmp_path / "run" / "critic",
        openenv_python=tmp_path / "openenv-python",
        codex_env={},
    )
    captured = {}
    monkeypatch.setattr(launcher, "_preflight", lambda _args: prepared)
    monkeypatch.setattr(
        launcher.U, "execute_train", lambda **kwargs: captured.update(kwargs)
    )
    args = launcher.ScriptArgs(
        sao_context_sha256="a" * 64,
        streaming_context_sha256="b" * 64,
        critic_contract_sha256="c" * 64,
    )

    launcher.execute(args)

    tokens = shlex.split(captured["train_args"])
    for flag, expected in {
        "--num-rollout": "1",
        "--rollout-batch-size": "22",
        "--over-sampling-batch-size": "22",
        "--num-steps-per-rollout": "1",
        "--num-critic-epochs": "2",
        "--global-batch-size": "22",
        "--num-gpus-per-node": "1",
        "--actor-num-gpus-per-node": "1",
        "--critic-num-gpus-per-node": "1",
        "--rollout-num-gpus": "1",
        "--sglang-max-running-requests": "38",
        "--sglang-mem-fraction-static": "0.15",
        "--sglang-max-total-tokens": "393216",
        "--sglang-max-mamba-cache-size": "256",
        "--rollout-weight-version-format": "counter",
    }.items():
        assert tokens[tokens.index(flag) + 1] == expected
    for flag in (
        "--sao-one-gpu-island",
        "--colocate",
        "--offload",
        "--sao-compaction",
        "--use-dynamic-global-batch-size",
        "--no-save-optim",
    ):
        assert flag in tokens
    assert tokens[tokens.index("--save") + 1] == str(prepared.actor_save)
    assert tokens[tokens.index("--critic-save") + 1] == str(prepared.critic_save)
    assert tokens[tokens.index("--rollout-seed") + 1] == "82621"
    assert captured["train_script"].endswith("train_sao_streaming_secrlenv.py")
    assert captured["num_gpus_per_node"] == 1
    assert "PYTORCH_CUDA_ALLOC_CONF" not in captured["extra_env_vars"]
