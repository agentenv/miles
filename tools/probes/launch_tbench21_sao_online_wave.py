#!/usr/bin/env python3
"""Launch the eight isolated TB2.1 SAO streaming-DiLoCo islands.

This host-side launcher is deliberately fail closed.  It checks the frozen
plan, both stages of the streaming-contract bundle, the pretrained critic,
the two already-running syncers, all eight GPUs, and every mount before it
creates a container.  It never stops unrelated containers or processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from tools.probes.launch_tbench21_rollout_wave import (
        CODEX_BINARY_RELATIVE,
        ISLANDS,
        MIN_GPU_MEMORY_MIB,
        RUNTIME_IMAGE,
        RUNTIME_IMAGE_ID,
        _bridge_gateway,
    )
else:
    from launch_tbench21_rollout_wave import (  # type: ignore[no-redef]
        CODEX_BINARY_RELATIVE,
        ISLANDS,
        MIN_GPU_MEMORY_MIB,
        RUNTIME_IMAGE,
        RUNTIME_IMAGE_ID,
        _bridge_gateway,
    )


PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
CONTRACT_SCHEMA = "yeto.tbench21-sao-streaming-contracts.v1"
CONTEXT_SCHEMA = "miles.sao-runtime.v2"
STREAMING_SCHEMA = "yeto.sao-streaming-runtime.v2"
SYNCER_LAUNCH_SCHEMA = "yeto.tbench21-sao-syncer-launch.v1"
MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_CONFIG_SHA256 = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
CODEX_VERSION = "codex-cli 0.145.0"
CODEX_SHA256 = "a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14"
CODEX_SIZE_BYTES = 310730800
ROLLOUT_SEED_BASE = 82621
ACTOR_SYNCER_PORT = 29400
CRITIC_SYNCER_PORT = 29401
CONTAINER_PLAN_DIR = Path("/root/plan")
CONTAINER_CONTRACT_DIR = Path("/root/contracts")
CONTAINER_RUN_ROOT = Path("/root/runs/tbench21-sao-full/online")
CONTAINER_PREFIX = "tbench21-sao-online-island"


class OnlineWaveLaunchError(RuntimeError):
    """An immutable online-wave launch condition was not satisfied."""


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=True,
        text=True,
        capture_output=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_file(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OnlineWaveLaunchError(f"{name} must be an absolute regular non-symlink")
    resolved = path.resolve()
    if stat.S_IMODE(resolved.stat().st_mode) not in {0o400, 0o600}:
        raise OnlineWaveLaunchError(f"{name} must have mode 0400 or 0600")
    return resolved


def _load_private_json(path: Path, name: str) -> dict[str, Any]:
    resolved = _private_file(path, name)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OnlineWaveLaunchError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise OnlineWaveLaunchError(f"{name} must contain a JSON object")
    return value


def _directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise OnlineWaveLaunchError(f"{name} must be an absolute real directory")
    return path.resolve()


def _regular_file(path: Path, name: str, *, executable: bool = False) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OnlineWaveLaunchError(f"{name} must be an absolute regular non-symlink")
    path = path.resolve()
    if executable and not os.access(path, os.X_OK):
        raise OnlineWaveLaunchError(f"{name} is not executable")
    return path


def _canonical_without_final_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"phase", "final_attestation_template", "islands"}}


def _load_rows(path: Path, island_id: int) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise OnlineWaveLaunchError(f"train shard is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise OnlineWaveLaunchError(f"train shard is unreadable: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise OnlineWaveLaunchError(f"train shard has invalid JSON at line {line_number}: {path}") from error
        metadata = row.get("metadata") if isinstance(row, dict) else None
        messages = row.get("messages") if isinstance(row, dict) else None
        task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
        replica = metadata.get("rollout_replica") if isinstance(metadata, dict) else None
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(metadata, dict)
            or not isinstance(task_id, str)
            or not task_id
            or type(replica) is not int
            or replica not in range(4)
            or metadata.get("sample_id") != f"train:{task_id}:r{replica}"
            or metadata.get("split") != "train"
            or metadata.get("island_id") != island_id
            or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
            or metadata.get("prompt_tier") != "l2"
            or metadata.get("max_seq_len") != 8192
            or metadata.get("episode_timeout_seconds") != 1800
        ):
            raise OnlineWaveLaunchError(f"train row violates island {island_id}'s immutable contract")
        rows.append(row)
    sample_ids = [row["metadata"]["sample_id"] for row in rows]
    task_ids = [row["metadata"]["task_id"] for row in rows]
    if len(rows) != 22 or len(set(sample_ids)) != 22 or len(set(task_ids)) != 22:
        raise OnlineWaveLaunchError(f"island {island_id} must own 22 unique task/sample rows")
    return rows


def _validate_plan(plan_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise OnlineWaveLaunchError("plan manifest is missing or unsafe")
    try:
        plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OnlineWaveLaunchError("plan manifest is unreadable or malformed") from error
    topology = plan.get("topology") if isinstance(plan, dict) else None
    rollouts = plan.get("rollouts") if isinstance(plan, dict) else None
    compaction = plan.get("compaction") if isinstance(plan, dict) else None
    files = plan.get("files") if isinstance(plan, dict) else None
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(topology, dict)
        or topology.get("islands") != ISLANDS
        or topology.get("physical_gpus") != ISLANDS
        or topology.get("one_physical_gpu_per_island") is not True
        or topology.get("model") != MODEL
        or topology.get("model_revision") != MODEL_REVISION
        or not isinstance(rollouts, dict)
        or rollouts.get("train") != 176
        or rollouts.get("per_task") != 4
        or rollouts.get("episode_timeout_seconds") != 1800
        or rollouts.get("seed_base") != ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != [ROLLOUT_SEED_BASE + index for index in range(ISLANDS)]
        or not isinstance(compaction, dict)
        or compaction.get("enabled") is not True
        or compaction.get("trainer_objective") != "sao"
        or not isinstance(files, dict)
    ):
        raise OnlineWaveLaunchError("plan manifest differs from the online-wave contract")
    shards: list[Path] = []
    all_sample_ids: list[str] = []
    all_task_ids: list[str] = []
    for island_id in range(ISLANDS):
        relative = f"train/island-{island_id}.jsonl"
        shard = plan_dir / relative
        expected_hash = files.get(relative)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or not shard.is_file() or shard.is_symlink() or _sha256(shard) != expected_hash:
            raise OnlineWaveLaunchError(f"plan does not bind unchanged shard {relative}")
        rows = _load_rows(shard, island_id)
        all_sample_ids.extend(row["metadata"]["sample_id"] for row in rows)
        all_task_ids.extend(row["metadata"]["task_id"] for row in rows)
        shards.append(shard)
    if len(set(all_sample_ids)) != 176 or len(set(all_task_ids)) != 44:
        raise OnlineWaveLaunchError("the eight shards do not form the exact 176-row split")
    return plan, shards


def _hash_bound_file(
    record: object,
    *,
    expected_host_path: Path,
    expected_container_path: Path,
    name: str,
) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise OnlineWaveLaunchError(f"{name} record is missing")
    digest = record.get("sha256")
    if record.get("host_path") != str(expected_host_path) or record.get("container_path") != str(expected_container_path) or not isinstance(digest, str) or len(digest) != 64:
        raise OnlineWaveLaunchError(f"{name} paths or hash drifted")
    _private_file(expected_host_path, name)
    if _sha256(expected_host_path) != digest:
        raise OnlineWaveLaunchError(f"{name} content changed")
    return expected_host_path, digest


def _validate_contracts(
    contracts_dir: Path,
    plan_dir: Path,
    miles_root: Path,
    critic_checkpoint: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    prepare_path = contracts_dir / "prepare-manifest.json"
    final_path = contracts_dir / "manifest.json"
    prepared = _load_private_json(prepare_path, "prepare contract manifest")
    final = _load_private_json(final_path, "final contract manifest")
    if prepared.get("schema") != CONTRACT_SCHEMA or prepared.get("phase") != "prepared" or final.get("schema") != CONTRACT_SCHEMA or final.get("phase") != "final" or _canonical_without_final_fields(prepared) != _canonical_without_final_fields(final):
        raise OnlineWaveLaunchError("prepare/final contract manifests do not match")
    if final.get("container_paths") != {
        "plan_dir": str(CONTAINER_PLAN_DIR),
        "contract_dir": str(CONTAINER_CONTRACT_DIR),
        "run_root": str(CONTAINER_RUN_ROOT),
    }:
        raise OnlineWaveLaunchError("contract container paths differ from production mounts")
    for field, name in (
        ("provisional_attestation", "provisional layout attestation"),
        ("final_attestation_template", "final layout attestation template"),
    ):
        record = final.get(field)
        source = Path(record.get("path", "")) if isinstance(record, dict) else Path("")
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not source.is_absolute() or source.is_symlink() or not source.is_file() or not isinstance(expected, str) or _sha256(source) != expected:
            raise OnlineWaveLaunchError(f"{name} changed after contract creation")
    plan_record = final.get("plan")
    if not isinstance(plan_record, dict) or plan_record.get("path") != str(plan_dir / "manifest.json") or plan_record.get("sha256") != _sha256(plan_dir / "manifest.json"):
        raise OnlineWaveLaunchError("contract bundle is not bound to this plan")
    reward = miles_root / "examples" / "experimental" / "openenv" / "openenv_generate.py"
    if not reward.is_file() or reward.is_symlink() or final.get("reward_source_sha256") != _sha256(reward):
        raise OnlineWaveLaunchError("reward implementation changed after contract creation")
    critic_contract = critic_checkpoint / "value_pretrain_contract.json"
    if not critic_contract.is_file() or critic_contract.is_symlink() or final.get("critic_value_pretraining_contract_sha256") != _sha256(critic_contract):
        raise OnlineWaveLaunchError("critic contract changed after contract creation")
    critic_digest = _sha256(critic_contract)

    profile_record = final.get("syncer_profile")
    training_record = final.get("training_contract")
    contexts = final.get("contexts")
    islands = final.get("islands")
    fragments = final.get("expected_fragments")
    if (
        type(fragments) is not int
        or fragments not in {1, 2}
        or not isinstance(profile_record, dict)
        or not isinstance(profile_record.get("profile"), dict)
        or profile_record["profile"].get("total_steps") != fragments
        or profile_record["profile"].get("learners") != ISLANDS
        or profile_record["profile"].get("quorum") != ISLANDS
        or profile_record["profile"].get("pipeline") != 2
        or not isinstance(training_record, dict)
        or not isinstance(contexts, list)
        or len(contexts) != ISLANDS
        or not isinstance(islands, list)
        or len(islands) != ISLANDS
    ):
        raise OnlineWaveLaunchError("final contract bundle is incomplete")
    profile_path = contracts_dir / "syncer-profile.json"
    training_path = contracts_dir / "training-contract.json"
    if (
        profile_record.get("host_path") != str(profile_path)
        or profile_record.get("container_path") != str(CONTAINER_CONTRACT_DIR / "syncer-profile.json")
        or profile_record.get("file_sha256") != _sha256(profile_path)
        or training_record.get("host_path") != str(training_path)
        or training_record.get("sha256") != _sha256(training_path)
    ):
        raise OnlineWaveLaunchError("profile/training-contract files changed")
    _private_file(profile_path, "syncer profile")
    _private_file(training_path, "training contract")

    generated: list[dict[str, str]] = []
    plan_files = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))["files"]
    for island_id in range(ISLANDS):
        context_record = contexts[island_id]
        island_record = islands[island_id]
        if not isinstance(context_record, dict) or context_record.get("island_id") != island_id or not isinstance(island_record, dict) or island_record.get("island_id") != island_id or island_record.get("sao_context") != context_record:
            raise OnlineWaveLaunchError("contract island ordering or base context changed")
        host_dir = contracts_dir / f"island-{island_id}"
        context_path, context_digest = _hash_bound_file(
            context_record,
            expected_host_path=host_dir / "sao-context.json",
            expected_container_path=(CONTAINER_CONTRACT_DIR / f"island-{island_id}/sao-context.json"),
            name=f"island {island_id} SAO context",
        )
        _, _ = _hash_bound_file(
            island_record.get("layout_attestation"),
            expected_host_path=host_dir / "layout-attestation.json",
            expected_container_path=(CONTAINER_CONTRACT_DIR / f"island-{island_id}/layout-attestation.json"),
            name=f"island {island_id} layout attestation",
        )
        stream_path, stream_digest = _hash_bound_file(
            island_record.get("streaming_context"),
            expected_host_path=host_dir / "sao-streaming-context.json",
            expected_container_path=(CONTAINER_CONTRACT_DIR / f"island-{island_id}/sao-streaming-context.json"),
            name=f"island {island_id} streaming context",
        )
        context = _load_private_json(context_path, f"island {island_id} SAO context")
        streaming = _load_private_json(stream_path, f"island {island_id} streaming context")
        actor = streaming.get("actor")
        critic = streaming.get("critic")
        layout = streaming.get("layout_attestation")
        evidence = streaming.get("trajectory_evidence")
        if (
            context.get("schema") != CONTEXT_SCHEMA
            or context.get("learner_id") != island_id
            or context.get("benchmark") != "terminal-bench-2.1"
            or context.get("model") != MODEL
            or context.get("base_model_revision") != MODEL_REVISION
            or context.get("rollout_model_revision") != MODEL_REVISION
            or context.get("data") != str(CONTAINER_PLAN_DIR / f"train/island-{island_id}.jsonl")
            or context.get("data_sha256") != plan_files[f"train/island-{island_id}.jsonl"]
            or context.get("reward_sha256") != final.get("reward_source_sha256")
            or context.get("completed_groups_path") != str(CONTAINER_RUN_ROOT / f"island-{island_id}/completed-groups.pt")
            or context.get("event_tape") != str(CONTAINER_RUN_ROOT / f"island-{island_id}/events.jsonl")
            or streaming.get("schema") != STREAMING_SCHEMA
            or streaming.get("sao_context_sha256") != context_digest
            or not isinstance(layout, dict)
            or layout.get("path") != str(CONTAINER_CONTRACT_DIR / f"island-{island_id}/layout-attestation.json")
            or layout.get("sha256") != island_record["layout_attestation"]["sha256"]
            or not isinstance(evidence, dict)
            or evidence.get("directory") != str(CONTAINER_RUN_ROOT / f"island-{island_id}/trajectory-evidence")
            or evidence.get("kind") != "terminal-bench-2.1"
            or evidence.get("schema_version") != 2
            or not isinstance(actor, dict)
            or not isinstance(critic, dict)
            or actor.get("learner_id") != island_id
            or critic.get("learner_id") != island_id
            or actor.get("syncer") != {"host": "host.docker.internal", "port": ACTOR_SYNCER_PORT}
            or critic.get("syncer") != {"host": "host.docker.internal", "port": CRITIC_SYNCER_PORT}
        ):
            raise OnlineWaveLaunchError(f"island {island_id} runtime contexts drifted")
        generated.append(
            {
                "sao_context_sha256": context_digest,
                "streaming_context_sha256": stream_digest,
            }
        )
    return final, generated, critic_digest


def _validate_syncers(launch_manifest_path: Path, contract_path: Path) -> dict[str, Any]:
    launch = _load_private_json(launch_manifest_path, "syncer launch manifest")
    contracts = launch.get("contracts") if isinstance(launch, dict) else None
    processes = launch.get("processes") if isinstance(launch, dict) else None
    if launch.get("schema") != SYNCER_LAUNCH_SCHEMA or not isinstance(contracts, dict) or contracts.get("path") != str(contract_path) or contracts.get("sha256") != _sha256(contract_path) or not isinstance(processes, list) or len(processes) != 2:
        raise OnlineWaveLaunchError("syncer launch manifest is not bound to contracts")
    expected = {"actor": ACTOR_SYNCER_PORT, "critic": CRITIC_SYNCER_PORT}
    seen: set[str] = set()
    for process in processes:
        role = process.get("role") if isinstance(process, dict) else None
        pid = process.get("pid") if isinstance(process, dict) else None
        port = process.get("port") if isinstance(process, dict) else None
        if role not in expected or role in seen or port != expected[role] or type(pid) is not int or pid < 2:
            raise OnlineWaveLaunchError("syncer launch process records drifted")
        try:
            os.kill(pid, 0)
        except OSError as error:
            raise OnlineWaveLaunchError(f"{role} syncer process is not alive") from error
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                pass
        except OSError as error:
            raise OnlineWaveLaunchError(f"{role} syncer port is not reachable") from error
        seen.add(role)
    if seen != set(expected):
        raise OnlineWaveLaunchError("actor/critic syncer records are incomplete")
    return launch


def _image_gate() -> None:
    try:
        payload = json.loads(_run(["docker", "image", "inspect", RUNTIME_IMAGE]).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise OnlineWaveLaunchError("the pinned online runtime image is absent") from error
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("Id") != RUNTIME_IMAGE_ID:
        raise OnlineWaveLaunchError("the online runtime image identity drifted")


def _gpu_gate() -> list[dict[str, Any]]:
    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).stdout
    except subprocess.CalledProcessError as error:
        raise OnlineWaveLaunchError("cannot inspect the eight physical GPUs") from error
    gpus = []
    for line in output.splitlines():
        try:
            index_raw, name, memory_raw = line.split(",", 2)
            gpus.append(
                {
                    "index": int(index_raw.strip()),
                    "name": name.strip(),
                    "memory_total_mib": int(memory_raw.strip()),
                }
            )
        except ValueError as error:
            raise OnlineWaveLaunchError("nvidia-smi returned malformed GPU evidence") from error
    if len(gpus) != ISLANDS or [gpu["index"] for gpu in gpus] != list(range(ISLANDS)):
        raise OnlineWaveLaunchError("online wave requires physical GPUs 0..7")
    if any("H200" not in gpu["name"].upper() or gpu["memory_total_mib"] < MIN_GPU_MEMORY_MIB for gpu in gpus):
        raise OnlineWaveLaunchError("online wave requires eight 120-GiB-or-larger H200s")
    active = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    if active:
        raise OnlineWaveLaunchError("one or more GPUs still have active compute processes")
    return gpus


def _container_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["docker", "container", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _container_argv(
    *,
    island_id: int,
    name: str,
    miles_root: Path,
    yeto_root: Path,
    model: Path,
    actor_checkpoint: Path,
    critic_checkpoint: Path,
    plan_dir: Path,
    contracts_dir: Path,
    output_dir: Path,
    codex_dir: Path,
    hmac_key: Path,
    bridge_gateway: str,
    context_hashes: dict[str, str],
    critic_contract_sha256: str,
) -> list[str]:
    container_output = CONTAINER_RUN_ROOT / f"island-{island_id}"
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--label",
        "yeto.run=tbench21-sao-online",
        "--label",
        f"yeto.island={island_id}",
        "--runtime",
        "nvidia",
        "--env",
        f"NVIDIA_VISIBLE_DEVICES={island_id}",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--env",
        "TBENCH_REWARD_HMAC_KEY_FILE=/run/secrets/tbench-reward-hmac.key",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--add-host",
        f"host.docker.internal:{bridge_gateway}",
        "--shm-size",
        "64g",
        "--ulimit",
        "nofile=1048576:1048576",
        "--mount",
        f"type=bind,src={miles_root},dst=/root/miles,readonly",
        "--mount",
        f"type=bind,src={yeto_root},dst=/root/yeto,readonly",
        "--mount",
        f"type=bind,src={model},dst=/root/model,readonly",
        "--mount",
        f"type=bind,src={actor_checkpoint},dst=/root/actor-checkpoint,readonly",
        "--mount",
        f"type=bind,src={critic_checkpoint},dst=/root/critic-checkpoint,readonly",
        "--mount",
        f"type=bind,src={plan_dir},dst={CONTAINER_PLAN_DIR},readonly",
        "--mount",
        f"type=bind,src={contracts_dir},dst={CONTAINER_CONTRACT_DIR},readonly",
        "--mount",
        f"type=bind,src={output_dir},dst={container_output}",
        "--mount",
        f"type=bind,src={codex_dir},dst=/root/codex-cli,readonly",
        "--mount",
        f"type=bind,src={hmac_key},dst=/run/secrets/tbench-reward-hmac.key,readonly",
        "--entrypoint",
        "python3",
        RUNTIME_IMAGE_ID,
        "/root/miles/tools/probes/run_tbench21_sao_streaming_island.py",
        "--island-id",
        str(island_id),
        "--prompt-data",
        f"{CONTAINER_PLAN_DIR}/train/island-{island_id}.jsonl",
        "--sao-context",
        f"{CONTAINER_CONTRACT_DIR}/island-{island_id}/sao-context.json",
        "--sao-context-sha256",
        context_hashes["sao_context_sha256"],
        "--streaming-context",
        f"{CONTAINER_CONTRACT_DIR}/island-{island_id}/sao-streaming-context.json",
        "--streaming-context-sha256",
        context_hashes["streaming_context_sha256"],
        "--run-root",
        str(container_output),
        "--hf-checkpoint",
        "/root/model",
        "--ref-load",
        "/root/actor-checkpoint",
        "--critic-load",
        "/root/critic-checkpoint",
        "--critic-contract-sha256",
        critic_contract_sha256,
        "--codex-binary",
        f"/root/codex-cli/{CODEX_BINARY_RELATIVE}",
        "--openenv-agent-python",
        "/root/openenv-venv/bin/python",
        "--openenv-env-url",
        "http://host.docker.internal:8003",
        "--tbench-hmac-key-file",
        "/run/secrets/tbench-reward-hmac.key",
        "--miles-root",
        "/root/miles",
        "--yeto-root",
        "/root/yeto",
        "--megatron-path",
        "/root/Megatron-LM",
        "--rollout-seed",
        str(ROLLOUT_SEED_BASE),
    ]


def launch(args: argparse.Namespace) -> dict[str, Any]:
    miles_root = _directory(args.miles_root, "Miles source")
    yeto_root = _directory(args.yeto_root, "Yeto source")
    model = _directory(args.model, "Qwen model")
    actor = _directory(args.actor_checkpoint, "actor checkpoint")
    critic = _directory(args.critic_checkpoint, "critic checkpoint")
    plan_dir = _directory(args.plan_dir, "Terminal-Bench plan")
    contracts_dir = _directory(args.contracts_dir, "streaming contracts")
    codex_dir = _directory(args.codex_dir, "Codex package")
    for checkpoint, name in ((actor, "actor"), (critic, "critic")):
        if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
            raise OnlineWaveLaunchError(f"{name} checkpoint has no iteration marker")
    codex_binary = _regular_file(codex_dir / CODEX_BINARY_RELATIVE, "stock Codex binary", executable=True)
    model_config = _regular_file(model / "config.json", "Qwen model config")
    if _sha256(model_config) != MODEL_CONFIG_SHA256:
        raise OnlineWaveLaunchError("Qwen3.5-0.8B model config identity drifted")
    if codex_binary.stat().st_size != CODEX_SIZE_BYTES or _sha256(codex_binary) != CODEX_SHA256:
        raise OnlineWaveLaunchError("stock Codex binary identity drifted")
    try:
        codex_version = _run([str(codex_binary), "--version"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OnlineWaveLaunchError("stock Codex binary cannot execute") from error
    if codex_version != CODEX_VERSION:
        raise OnlineWaveLaunchError(f"stock Codex version drifted: {codex_version!r}")
    if os.getenv("TBENCH_REWARD_HMAC_KEY"):
        raise OnlineWaveLaunchError("direct Terminal-Bench HMAC keys are forbidden")
    hmac_key = _private_file(args.hmac_key, "Terminal-Bench HMAC key")
    key_size = len(hmac_key.read_bytes().rstrip(b"\r\n"))
    if not 32 <= key_size <= 4096:
        raise OnlineWaveLaunchError("Terminal-Bench HMAC key must contain 32..4096 bytes")
    _, shards = _validate_plan(plan_dir)
    final, hashes, critic_digest = _validate_contracts(contracts_dir, plan_dir, miles_root, critic)
    syncer_launch = _validate_syncers(args.syncer_launch_manifest, contracts_dir / "manifest.json")
    bridge_gateway = _bridge_gateway()
    _image_gate()
    gpus = _gpu_gate()

    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise OnlineWaveLaunchError("online output root must be fresh")
    names = [f"{CONTAINER_PREFIX}-{index}" for index in range(ISLANDS)]
    existing = [name for name in names if _container_exists(name)]
    if existing:
        raise OnlineWaveLaunchError(f"online container names already exist: {existing}")

    output_root.mkdir(mode=0o700, parents=True)
    created: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for island_id, name in enumerate(names):
            output = output_root / f"island-{island_id}"
            output.mkdir(mode=0o700)
            argv = _container_argv(
                island_id=island_id,
                name=name,
                miles_root=miles_root,
                yeto_root=yeto_root,
                model=model,
                actor_checkpoint=actor,
                critic_checkpoint=critic,
                plan_dir=plan_dir,
                contracts_dir=contracts_dir,
                output_dir=output,
                codex_dir=codex_dir,
                hmac_key=hmac_key,
                bridge_gateway=bridge_gateway,
                context_hashes=hashes[island_id],
                critic_contract_sha256=critic_digest,
            )
            container_id = _run(argv).stdout.strip()
            created.append(name)
            if len(container_id) != 64 or any(character not in "0123456789abcdef" for character in container_id):
                raise OnlineWaveLaunchError(f"Docker returned an invalid ID for {name}")
            records.append(
                {
                    "island_id": island_id,
                    "name": name,
                    "container_id": container_id,
                    "gpu": island_id,
                    "rollout_seed": ROLLOUT_SEED_BASE + island_id,
                    "plan_shard": str(shards[island_id]),
                    "plan_shard_sha256": _sha256(shards[island_id]),
                    "sao_context_sha256": hashes[island_id]["sao_context_sha256"],
                    "streaming_context_sha256": hashes[island_id]["streaming_context_sha256"],
                    "output": str(output),
                    "container_output": str(CONTAINER_RUN_ROOT / f"island-{island_id}"),
                }
            )
    except BaseException:
        for name in reversed(created):
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        raise

    manifest = {
        "schema": "miles.tbench21-sao-online-wave.v1",
        "runtime_image": {"reference": RUNTIME_IMAGE, "id": RUNTIME_IMAGE_ID},
        "gpus": gpus,
        "model": str(model),
        "actor_checkpoint": str(actor),
        "critic_checkpoint": str(critic),
        "critic_contract_sha256": critic_digest,
        "plan": {
            "path": str(plan_dir / "manifest.json"),
            "sha256": _sha256(plan_dir / "manifest.json"),
        },
        "contracts": {
            "prepare_sha256": _sha256(contracts_dir / "prepare-manifest.json"),
            "final_path": str(contracts_dir / "manifest.json"),
            "final_sha256": _sha256(contracts_dir / "manifest.json"),
            "training_contract_sha256": final["training_contract"]["sha256"],
            "semantic_profile_sha256": final["syncer_profile"]["semantic_sha256"],
        },
        "syncers": {
            "launch_manifest": str(args.syncer_launch_manifest.resolve()),
            "launch_manifest_sha256": _sha256(args.syncer_launch_manifest.resolve()),
            "processes": syncer_launch["processes"],
        },
        "codex_binary_sha256": _sha256(codex_binary),
        "hmac_key_file_mode": oct(stat.S_IMODE(hmac_key.stat().st_mode)),
        "docker_bridge_gateway": bridge_gateway,
        "containers": records,
    }
    manifest_path = output_root / "launch-manifest.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return {
        "manifest": str(manifest_path),
        "containers": names,
        "monitor": ("watch -n 10 'docker ps -a --filter label=yeto.run=tbench21-sao-online --format \"table {{.Names}}\\t{{.Status}}\"'"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miles-root", type=Path, required=True)
    parser.add_argument("--yeto-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--contracts-dir", type=Path, required=True)
    parser.add_argument("--syncer-launch-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-dir", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        result = launch(_parser().parse_args(argv))
    except Exception as error:
        print(
            f"TB2.1 online launch failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
