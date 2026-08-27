from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterShardManifest,
    FullParameterTopology,
    make_full_parameter_shard_state,
)
from tools.probes.sao_streaming_layout_probe import (
    SAOStreamingLayoutProbe,
    _revision_env,
)


def _syncer_profile():
    return {
        "schema": "yeto.syncer-semantic-profile.v1",
        "learners": 2,
        "quorum": 2,
        "grace_ms": 0,
        "grace_gamma": 0.8,
        "grace_tau": 2.0,
        "pipeline": 2,
        "min_round_interval_ms": 0,
        "sync_interval_steps": 24.0,
        "delta_correction": "none",
        "quorum_timeout_s": 900,
        "final_ack_timeout_s": 900,
        "total_steps": 4,
        "policy_sweep_fragments": None,
        "outer_lr": 0.7,
        "outer_momentum": 0.9,
        "checkpoint_enabled": True,
        "checkpoint_every": 1,
        "resume": True,
        "mark_final_checkpoint": True,
        "learner_budget_steps": None,
        "max_base_lag": 0,
        "learner_weight": "equal",
        "require_profile_binding": True,
    }


def _manifest(role: str, sizes: tuple[int, ...]) -> FullParameterShardManifest:
    topology = FullParameterTopology(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    parameters = []
    for index, size in enumerate(sizes):
        parameter = torch.nn.Parameter(torch.zeros(size, dtype=torch.bfloat16))
        parameter.main_param = torch.arange(size, dtype=torch.float32)
        parameters.append((f"module.layers.{index}.weight", parameter))
    state = make_full_parameter_shard_state(
        policy_version=0,
        role=role,
        topology=topology,
        named_parameters=parameters,
    )
    return FullParameterShardManifest(
        topology=state.topology,
        role=role,
        layout_hash=state.layout_hash,
        specs=state.specs,
    )


class _Group:
    def __init__(self, manifest):
        self.manifest = manifest
        self.manifest_calls = 0

    async def full_parameter_shard_manifests(self):
        self.manifest_calls += 1
        return (self.manifest,)


def _set_probe_env(monkeypatch, evidence):
    values = {
        "YETO_SAO_STREAMING_LAYOUT_EVIDENCE": str(evidence),
        "YETO_SAO_CONTEXT_SHA256": "9" * 64,
        "YETO_SAO_ACTOR_MODEL_REVISION": "a" * 40,
        "YETO_SAO_ACTOR_CONFIG_SHA256": "1" * 64,
        "YETO_SAO_CRITIC_MODEL_REVISION": "b" * 40,
        "YETO_SAO_CRITIC_CONFIG_SHA256": "2" * 64,
        "YETO_SAO_STREAMING_MAX_FRAGMENT_BYTES": "32",
        "YETO_SAO_STREAMING_MAX_CHUNK_BYTES": "16",
        "YETO_SAO_SYNCER_PROFILE_JSON": json.dumps(
            _syncer_profile(), sort_keys=True, separators=(",", ":")
        ),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _runtime_role(role, evidence, port):
    attested = evidence[role]
    return {
        "role": role,
        "component": attested["component"],
        "syncer": {"host": "127.0.0.1", "port": port},
        "learner_id": 0,
        "learner_generation": 0,
        "learner_generations": [0, 0],
        "total_fragment_steps": 4,
        "expected_fragments": attested["expected_fragments"],
        "parameter_layout_sha256": attested["parameter_layout_sha256"],
        "local_horizon": 1,
        "optimizer_steps_per_round": 1 if role == "actor" else 2,
        "training_contract_sha256": "5" * 64,
        "wan_streams": 2,
        "wait_timeout_seconds": 30.0,
        "poll_seconds": 0.01,
        "max_fragment_bytes": 32,
        "max_chunk_bytes": 16,
    }


def test_model_revision_is_exactly_sha1_or_sha256(monkeypatch):
    name = "YETO_TEST_MODEL_REVISION"
    for length in (40, 64):
        monkeypatch.setenv(name, "a" * length)
        assert _revision_env(name) == "a" * length
    monkeypatch.setenv(name, "a" * 50)
    with pytest.raises(RuntimeError, match="not a commit"):
        _revision_env(name)


@pytest.mark.asyncio
async def test_two_role_probe_is_metadata_only_and_feeds_runtime_config(
    tmp_path,
    monkeypatch,
):
    yeto_source = Path(__file__).resolve().parents[3] / "yeto-grpo-diloco-connector"
    if not yeto_source.is_dir():
        pytest.skip("cross-repository Yeto source is unavailable")
    for name in tuple(sys.modules):
        if name == "yeto" or name.startswith("yeto."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(yeto_source))
    evidence_path = tmp_path / "sao-streaming-layouts.json"
    _set_probe_env(monkeypatch, evidence_path)
    args = SimpleNamespace(
        debug_train_only=True,
        use_critic=True,
        lora_rank=0,
        start_rollout_id=0,
        num_critic_only_steps=0,
        num_rollout=1,
    )
    # Actor fits one fragment independently; critic needs two. The probe must
    # deliberately rebuild both at the common lockstep count.
    actor = _Group(_manifest("actor", (4, 4)))
    critic = _Group(_manifest("critic", (8, 4, 2)))
    probe = SAOStreamingLayoutProbe(args)

    await probe.initialize(
        actor_model=actor,
        critic_model=critic,
        rollout_manager=object(),
    )

    assert args.num_rollout == 0
    assert actor.manifest_calls == critic.manifest_calls == 1
    assert not evidence_path.exists()
    with pytest.raises(RuntimeError, match="entered training"):
        await probe.after_local_train()

    await probe.finalize()

    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600
    assert probe.evidence_sha256 == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == "miles.sao-streaming-layouts.v2"
    assert evidence["actor"]["expected_fragments"] == 2
    assert evidence["critic"]["expected_fragments"] == 2
    assert evidence["settings"]["minimum_fragments"] == 2
    assert len(evidence["settings"]["syncer_profile_sha256"]) == 64
    assert evidence["actor"]["parameter_layout_sha256"] != evidence["critic"]["parameter_layout_sha256"]
    assert max(fragment["payload_bytes"] for fragment in evidence["actor"]["fragments"]) <= 32
    assert max(fragment["payload_bytes"] for fragment in evidence["critic"]["fragments"]) <= 32

    runtime_payload = {
        "schema": "yeto.sao-streaming-runtime.v2",
        "sao_context_sha256": "9" * 64,
        "trajectory_evidence": {
            "directory": str(tmp_path / "trajectory-evidence"),
            "kind": "secrlenv",
            "schema_version": 2,
        },
        "layout_attestation": {
            "path": str(evidence_path),
            "sha256": probe.evidence_sha256,
        },
        "syncer_profile": _syncer_profile(),
        "actor": _runtime_role("actor", evidence, 29400),
        "critic": _runtime_role("critic", evidence, 29401),
    }
    runtime_path = tmp_path / "sao-streaming-runtime.json"
    runtime_path.write_text(
        json.dumps(runtime_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    from yeto.rl.sao_streaming_runtime import load_sao_streaming_runtime

    runtime = load_sao_streaming_runtime(
        runtime_path,
        expected_sha256=runtime_sha,
        expected_sao_context_sha256="9" * 64,
    )
    assert runtime.layout_attestation_sha256 == probe.evidence_sha256
    assert runtime.streams.actor.expected_fragments == 2
    assert runtime.streams.critic.expected_fragments == 2
