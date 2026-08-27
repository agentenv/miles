from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.probes import launch_tbench21_sao_online_wave as online


def _write(path: Path, value: object, *, private: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    if private:
        path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _plan(tmp_path: Path) -> Path:
    plan_dir = tmp_path / "plan"
    files: dict[str, str] = {}
    rows = [[] for _ in range(8)]
    for task_index in range(44):
        task_id = f"task-{task_index:02d}"
        for replica in range(4):
            island_id = (task_index * 4 + replica) % 8
            rows[island_id].append(
                {
                    "messages": [{"role": "system", "content": "solve"}],
                    "metadata": {
                        "task_id": task_id,
                        "prompt_tier": "l2",
                        "max_seq_len": 8192,
                        "split": "train",
                        "rollout_replica": replica,
                        "sample_id": f"train:{task_id}:r{replica}",
                        "island_id": island_id,
                        "rollout_seed": online.ROLLOUT_SEED_BASE + island_id,
                        "episode_timeout_seconds": 1800,
                    },
                }
            )
    for island_id, island_rows in enumerate(rows):
        path = plan_dir / f"train/island-{island_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in island_rows)
        path.write_bytes(encoded)
        files[f"train/island-{island_id}.jsonl"] = hashlib.sha256(encoded).hexdigest()
    _write(
        plan_dir / "manifest.json",
        {
            "schema": online.PLAN_SCHEMA,
            "topology": {
                "islands": 8,
                "physical_gpus": 8,
                "one_physical_gpu_per_island": True,
                "model": online.MODEL,
                "model_revision": online.MODEL_REVISION,
            },
            "rollouts": {
                "train": 176,
                "per_task": 4,
                "episode_timeout_seconds": 1800,
                "seed_base": online.ROLLOUT_SEED_BASE,
                "per_island_seeds": [online.ROLLOUT_SEED_BASE + index for index in range(8)],
            },
            "compaction": {"enabled": True, "trainer_objective": "sao"},
            "files": files,
        },
    )
    return plan_dir


def _contract_bundle(tmp_path: Path, plan_dir: Path) -> tuple[Path, Path, Path, list[dict[str, str]]]:
    miles = tmp_path / "miles"
    reward = miles / "examples" / "experimental" / "openenv" / "openenv_generate.py"
    reward.parent.mkdir(parents=True)
    reward.write_text("def reward(): return 1\n", encoding="utf-8")
    reward_sha = online._sha256(reward)

    critic = tmp_path / "critic"
    critic.mkdir()
    (critic / "latest_checkpointed_iteration.txt").write_text("1\n")
    critic_sha = _write(critic / "value_pretrain_contract.json", {"schema": "critic"})

    contracts = tmp_path / "contracts"
    contracts.mkdir()
    provisional = tmp_path / "provisional.json"
    final_template = tmp_path / "final-template.json"
    provisional_sha = _write(provisional, {"schema": "layout", "pass": 1})
    final_template_sha = _write(final_template, {"schema": "layout", "pass": 2})
    profile = {"learners": 8, "quorum": 8, "pipeline": 2, "total_steps": 2}
    profile_path = contracts / "syncer-profile.json"
    profile_file_sha = _write(profile_path, profile, private=True)
    training_path = contracts / "training-contract.json"
    training_sha = _write(training_path, {"schema": "training"}, private=True)

    plan = json.loads((plan_dir / "manifest.json").read_text())
    contexts = []
    island_records = []
    context_hashes = []
    for island_id in range(8):
        island_dir = contracts / f"island-{island_id}"
        context = {
            "schema": online.CONTEXT_SCHEMA,
            "benchmark": "terminal-bench-2.1",
            "model": online.MODEL,
            "base_model_revision": online.MODEL_REVISION,
            "rollout_model_revision": online.MODEL_REVISION,
            "learner_id": island_id,
            "data": f"/root/plan/train/island-{island_id}.jsonl",
            "data_sha256": plan["files"][f"train/island-{island_id}.jsonl"],
            "reward_sha256": reward_sha,
            "completed_groups_path": (f"/root/runs/tbench21-sao-full/online/island-{island_id}/completed-groups.pt"),
            "event_tape": (f"/root/runs/tbench21-sao-full/online/island-{island_id}/events.jsonl"),
        }
        context_path = island_dir / "sao-context.json"
        context_sha = _write(context_path, context, private=True)
        context_record = {
            "island_id": island_id,
            "host_path": str(context_path),
            "container_path": f"/root/contracts/island-{island_id}/sao-context.json",
            "sha256": context_sha,
        }
        layout_path = island_dir / "layout-attestation.json"
        layout_sha = _write(layout_path, {"schema": "layout", "island": island_id}, private=True)
        layout_record = {
            "host_path": str(layout_path),
            "container_path": f"/root/contracts/island-{island_id}/layout-attestation.json",
            "sha256": layout_sha,
        }
        streaming = {
            "schema": online.STREAMING_SCHEMA,
            "sao_context_sha256": context_sha,
            "trajectory_evidence": {
                "directory": f"/root/runs/tbench21-sao-full/online/island-{island_id}/trajectory-evidence",
                "kind": "terminal-bench-2.1",
                "schema_version": 2,
            },
            "layout_attestation": {
                "path": f"/root/contracts/island-{island_id}/layout-attestation.json",
                "sha256": layout_sha,
            },
            "actor": {
                "learner_id": island_id,
                "syncer": {"host": "host.docker.internal", "port": 29400},
            },
            "critic": {
                "learner_id": island_id,
                "syncer": {"host": "host.docker.internal", "port": 29401},
            },
        }
        stream_path = island_dir / "sao-streaming-context.json"
        stream_sha = _write(stream_path, streaming, private=True)
        stream_record = {
            "host_path": str(stream_path),
            "container_path": (f"/root/contracts/island-{island_id}/sao-streaming-context.json"),
            "sha256": stream_sha,
        }
        contexts.append(context_record)
        island_records.append(
            {
                "island_id": island_id,
                "sao_context": context_record,
                "layout_attestation": layout_record,
                "streaming_context": stream_record,
            }
        )
        context_hashes.append(
            {
                "sao_context_sha256": context_sha,
                "streaming_context_sha256": stream_sha,
            }
        )
    prepared = {
        "schema": online.CONTRACT_SCHEMA,
        "phase": "prepared",
        "plan": {
            "path": str(plan_dir / "manifest.json"),
            "sha256": online._sha256(plan_dir / "manifest.json"),
        },
        "provisional_attestation": {
            "path": str(provisional),
            "sha256": provisional_sha,
        },
        "container_paths": {
            "plan_dir": "/root/plan",
            "contract_dir": "/root/contracts",
            "run_root": "/root/runs/tbench21-sao-full/online",
        },
        "expected_fragments": 2,
        "syncer_profile": {
            "host_path": str(profile_path),
            "container_path": "/root/contracts/syncer-profile.json",
            "file_sha256": profile_file_sha,
            "semantic_sha256": "a" * 64,
            "profile": profile,
        },
        "training_contract": {
            "host_path": str(training_path),
            "sha256": training_sha,
            "contract": {"schema": "training"},
        },
        "reward_source_sha256": reward_sha,
        "critic_value_pretraining_contract_sha256": critic_sha,
        "contexts": contexts,
    }
    _write(contracts / "prepare-manifest.json", prepared, private=True)
    final = {
        **prepared,
        "phase": "final",
        "final_attestation_template": {
            "path": str(final_template),
            "sha256": final_template_sha,
        },
        "islands": island_records,
    }
    _write(contracts / "manifest.json", final, private=True)
    return contracts, miles, critic, context_hashes


def test_plan_and_contract_bundle_fail_closed_and_return_exact_hashes(tmp_path):
    plan = _plan(tmp_path)
    _, shards = online._validate_plan(plan)
    assert len(shards) == 8
    contracts, miles, critic, expected = _contract_bundle(tmp_path, plan)
    final, hashes, critic_hash = online._validate_contracts(contracts, plan, miles, critic)
    assert final["phase"] == "final"
    assert hashes == expected
    assert critic_hash == online._sha256(critic / "value_pretrain_contract.json")

    with (plan / "train/island-0.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(online.OnlineWaveLaunchError, match="unchanged shard"):
        online._validate_plan(plan)


def test_container_argv_is_one_gpu_read_only_and_hash_bound(tmp_path):
    hashes = {
        "sao_context_sha256": "a" * 64,
        "streaming_context_sha256": "b" * 64,
    }
    argv = online._container_argv(
        island_id=3,
        name="tbench21-sao-online-island-3",
        miles_root=tmp_path / "miles",
        yeto_root=tmp_path / "yeto",
        model=tmp_path / "model",
        actor_checkpoint=tmp_path / "actor",
        critic_checkpoint=tmp_path / "critic",
        plan_dir=tmp_path / "plan",
        contracts_dir=tmp_path / "contracts",
        output_dir=tmp_path / "output",
        codex_dir=tmp_path / "codex",
        hmac_key=tmp_path / "hmac",
        bridge_gateway="172.17.0.1",
        context_hashes=hashes,
        critic_contract_sha256="c" * 64,
    )
    rendered = " ".join(str(value) for value in argv)
    assert "--runtime nvidia" in rendered
    assert "--gpus" not in argv
    assert "NVIDIA_VISIBLE_DEVICES=3" in argv
    assert "/var/run/docker.sock" not in rendered
    assert "dst=/root/miles,readonly" in rendered
    assert "dst=/root/yeto,readonly" in rendered
    assert "dst=/root/model,readonly" in rendered
    assert "dst=/root/actor-checkpoint,readonly" in rendered
    assert "dst=/root/critic-checkpoint,readonly" in rendered
    assert "dst=/root/plan,readonly" in rendered
    assert "dst=/root/contracts,readonly" in rendered
    assert "dst=/root/runs/tbench21-sao-full/online/island-3" in rendered
    assert argv[argv.index("--sao-context-sha256") + 1] == "a" * 64
    assert argv[argv.index("--streaming-context-sha256") + 1] == "b" * 64
    assert argv[argv.index("--critic-contract-sha256") + 1] == "c" * 64
    assert argv[argv.index("--rollout-seed") + 1] == str(online.ROLLOUT_SEED_BASE)


def test_syncer_health_requires_bound_live_actor_and_critic(tmp_path, monkeypatch):
    contract = tmp_path / "manifest.json"
    _write(contract, {"schema": "contracts"}, private=True)
    launch = tmp_path / "syncers.json"
    _write(
        launch,
        {
            "schema": online.SYNCER_LAUNCH_SCHEMA,
            "contracts": {
                "path": str(contract),
                "sha256": online._sha256(contract),
            },
            "processes": [
                {"role": "actor", "port": 29400, "pid": 100},
                {"role": "critic", "port": 29401, "pid": 101},
            ],
        },
        private=True,
    )
    killed = []
    connected = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(online.os, "kill", lambda pid, signal: killed.append((pid, signal)))
    monkeypatch.setattr(
        online.socket,
        "create_connection",
        lambda address, timeout: connected.append((address, timeout)) or _Connection(),
    )
    result = online._validate_syncers(launch, contract)
    assert result["schema"] == online.SYNCER_LAUNCH_SCHEMA
    assert killed == [(100, 0), (101, 0)]
    assert [item[0][1] for item in connected] == [29400, 29401]
