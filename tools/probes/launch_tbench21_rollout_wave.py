#!/usr/bin/env python3
"""Launch all eight isolated TB2.1 baseline/eval rollout containers.

This host-side launcher never stops unrelated workloads.  It requires all
eight H200s to be idle, validates every immutable input before creating the
fresh output directory, and rolls back only containers it created if one
launch fails.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNTIME_IMAGE = "yeto-sao-tbench21-runtime:20260826"
RUNTIME_IMAGE_ID = "sha256:69be75252eea4179ccae554f362351a7502f365a89c99eaca322df19f4e55572"
CODEX_BINARY_RELATIVE = "vendor/x86_64-unknown-linux-musl/bin/codex"
ISLANDS = 8
MIN_GPU_MEMORY_MIB = 120 * 1024
ROLLOUT_SEED_BASE = 82621


class LaunchError(RuntimeError):
    pass


def _run(argv: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_key_file(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise LaunchError("Terminal-Bench HMAC key must be an absolute non-symlink")
    resolved = path.resolve()
    try:
        information = resolved.stat()
        value = resolved.read_bytes().rstrip(b"\r\n")
    except OSError as error:
        raise LaunchError("Terminal-Bench HMAC key is unreadable") from error
    if not stat.S_ISREG(information.st_mode) or stat.S_IMODE(information.st_mode) not in {0o400, 0o600} or not 32 <= len(value) <= 4096:
        raise LaunchError("Terminal-Bench HMAC key is not private and bounded")
    return resolved


def _directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise LaunchError(f"{name} must be an absolute real directory: {path}")
    return path.resolve()


def _plan_shards(plan_dir: Path, phase: str) -> list[Path]:
    manifest_path = plan_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LaunchError("Terminal-Bench plan manifest is unreadable") from error
    if manifest.get("schema") != "yeto.tbench21-sao-diloco-plan.v1":
        raise LaunchError("Terminal-Bench plan schema drifted")
    topology = manifest.get("topology")
    rollouts = manifest.get("rollouts")
    files = manifest.get("files")
    if not isinstance(topology, dict) or topology.get("islands") != ISLANDS or topology.get("one_physical_gpu_per_island") is not True or not isinstance(rollouts, dict) or rollouts.get("episode_timeout_seconds") != 1800 or not isinstance(files, dict):
        raise LaunchError("Terminal-Bench plan topology drifted")
    expected_counts = {"baseline": {44, 45}, "eval": {22, 23}}
    shards = []
    for island_id in range(ISLANDS):
        relative = f"{phase}/island-{island_id}.jsonl"
        path = plan_dir / relative
        expected_hash = files.get(relative)
        if not path.is_file() or path.is_symlink() or not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise LaunchError(f"plan shard is missing or changed: {relative}")
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LaunchError(f"plan shard is malformed: {relative}") from error
        if len(rows) not in expected_counts[phase]:
            raise LaunchError(f"plan shard cardinality changed: {relative}")
        shards.append(path)
    return shards


def _gpu_gate() -> list[dict[str, int]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    gpus = []
    for line in output.splitlines():
        index, memory = (int(value.strip()) for value in line.split(",", 1))
        gpus.append({"index": index, "memory_total_mib": memory})
    if len(gpus) != ISLANDS or [gpu["index"] for gpu in gpus] != list(range(ISLANDS)):
        raise LaunchError("the rollout wave requires exactly GPUs 0..7")
    if any(gpu["memory_total_mib"] < MIN_GPU_MEMORY_MIB for gpu in gpus):
        raise LaunchError("the rollout wave requires 120-GiB-or-larger GPUs")
    active = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    if active:
        raise LaunchError("one or more GPUs still have active compute processes")
    return gpus


def _image_gate() -> None:
    try:
        payload = json.loads(_run(["docker", "image", "inspect", RUNTIME_IMAGE]).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise LaunchError("the pinned rollout runtime image is absent") from error
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("Id") != RUNTIME_IMAGE_ID:
        raise LaunchError("the rollout runtime image identity drifted")


def _bridge_gateway() -> str:
    try:
        payload = json.loads(_run(["docker", "network", "inspect", "bridge"]).stdout)
        configs = payload[0]["IPAM"]["Config"]
        gateways = [entry.get("Gateway") for entry in configs if entry.get("Gateway")]
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
        raise LaunchError("cannot resolve Docker's bridge gateway") from error
    if len(gateways) != 1:
        raise LaunchError("Docker bridge must expose exactly one gateway")
    try:
        address = ipaddress.ip_address(gateways[0])
    except ValueError as error:
        raise LaunchError("Docker bridge returned an invalid gateway") from error
    if address.version != 4 or address.is_unspecified or address.is_loopback or address.is_multicast:
        raise LaunchError("Docker bridge gateway is not a usable IPv4 address")
    return str(address)


def _container_exists(name: str) -> bool:
    completed = subprocess.run(
        ["docker", "container", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _container_argv(
    *,
    phase: str,
    island_id: int,
    name: str,
    miles_root: Path,
    yeto_root: Path,
    model: Path,
    checkpoint: Path,
    plan_dir: Path,
    output: Path,
    codex_dir: Path,
    hmac_key: Path,
    bridge_gateway: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--label",
        "yeto.run=tbench21-sao-rollout",
        "--label",
        f"yeto.phase={phase}",
        "--label",
        f"yeto.island={island_id}",
        "--runtime",
        "nvidia",
        "--env",
        f"NVIDIA_VISIBLE_DEVICES={island_id}",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--add-host",
        f"host.docker.internal:{bridge_gateway}",
        "--shm-size",
        "64g",
        "--ulimit",
        "nofile=1048576:1048576",
        "--env",
        "TBENCH_REWARD_HMAC_KEY_FILE=/run/secrets/tbench-hmac",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--mount",
        f"type=bind,src={miles_root},dst=/root/miles,readonly",
        "--mount",
        f"type=bind,src={yeto_root},dst=/root/yeto,readonly",
        "--mount",
        f"type=bind,src={model},dst=/root/model,readonly",
        "--mount",
        f"type=bind,src={checkpoint},dst=/root/input-checkpoint,readonly",
        "--mount",
        f"type=bind,src={plan_dir},dst=/root/plan,readonly",
        "--mount",
        f"type=bind,src={output},dst=/root/run-output",
        "--mount",
        f"type=bind,src={codex_dir},dst=/root/codex-cli,readonly",
        "--mount",
        f"type=bind,src={hmac_key},dst=/run/secrets/tbench-hmac,readonly",
        "--entrypoint",
        "python3",
        RUNTIME_IMAGE,
        "/root/miles/tools/probes/run_tbench21_compaction_baseline.py",
        "--island-id",
        str(island_id),
        "--phase",
        phase,
        "--prompt-data",
        f"/root/plan/{phase}/island-{island_id}.jsonl",
        "--dump-details",
        "/root/run-output/details",
        "--hf-checkpoint",
        "/root/model",
        "--ref-load",
        "/root/input-checkpoint",
        "--codex-binary",
        f"/root/codex-cli/{CODEX_BINARY_RELATIVE}",
        "--openenv-agent-python",
        "/root/openenv-venv/bin/python",
        "--openenv-env-url",
        "http://host.docker.internal:8003",
        "--miles-root",
        "/root/miles",
        "--yeto-root",
        "/root/yeto",
        "--megatron-path",
        "/root/Megatron-LM",
        "--concurrency",
        "38",
        "--rollout-seed",
        str(ROLLOUT_SEED_BASE),
    ]


def launch(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase not in {"baseline", "eval"}:
        raise LaunchError("phase must be baseline or eval")
    miles_root = _directory(args.miles_root, "Miles source")
    yeto_root = _directory(args.yeto_root, "Yeto source")
    model = _directory(args.model, "Qwen model")
    checkpoint = _directory(args.checkpoint, "Megatron checkpoint")
    plan_dir = _directory(args.plan_dir, "Terminal-Bench plan")
    codex_dir = _directory(args.codex_dir, "Codex package")
    codex_binary = codex_dir / CODEX_BINARY_RELATIVE
    if not codex_binary.is_file() or codex_binary.is_symlink():
        raise LaunchError("pinned Codex binary is missing")
    hmac_key = _private_key_file(args.hmac_key)
    bridge_gateway = _bridge_gateway()
    shards = _plan_shards(plan_dir, args.phase)
    _image_gate()
    gpus = _gpu_gate()

    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise LaunchError("rollout output root must be fresh")
    names = [f"tbench21-{args.phase}-island-{index}" for index in range(ISLANDS)]
    existing = [name for name in names if _container_exists(name)]
    if existing:
        raise LaunchError(f"rollout container names already exist: {existing}")

    output_root.mkdir(mode=0o700, parents=True)
    created: list[str] = []
    records = []
    try:
        for island_id, name in enumerate(names):
            output = output_root / f"island-{island_id}"
            output.mkdir(mode=0o700)
            argv = _container_argv(
                phase=args.phase,
                island_id=island_id,
                name=name,
                miles_root=miles_root,
                yeto_root=yeto_root,
                model=model,
                checkpoint=checkpoint,
                plan_dir=plan_dir,
                output=output,
                codex_dir=codex_dir,
                hmac_key=hmac_key,
                bridge_gateway=bridge_gateway,
            )
            container_id = _run(argv).stdout.strip()
            if len(container_id) != 64:
                raise LaunchError(f"Docker returned an invalid container ID for {name}")
            created.append(name)
            records.append(
                {
                    "island_id": island_id,
                    "name": name,
                    "container_id": container_id,
                    "plan_shard": str(shards[island_id]),
                    "plan_shard_sha256": _sha256(shards[island_id]),
                    "output": str(output),
                }
            )
    except BaseException:
        for name in reversed(created):
            subprocess.run(
                ["docker", "stop", "--time", "10", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        raise

    manifest = {
        "schema": "miles.tbench21-rollout-wave.v1",
        "phase": args.phase,
        "runtime_image": {"reference": RUNTIME_IMAGE, "id": RUNTIME_IMAGE_ID},
        "gpus": gpus,
        "model": str(model),
        "checkpoint": str(checkpoint),
        "plan_dir": str(plan_dir),
        "codex_binary_sha256": _sha256(codex_binary),
        "hmac_key_file_mode": oct(stat.S_IMODE(hmac_key.stat().st_mode)),
        "per_island_rollout_seeds": [
            ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)
        ],
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
        "monitor": ("watch -n 10 'docker ps -a --filter label=yeto.run=tbench21-sao-rollout --format \"table {{.Names}}\\t{{.Status}}\"'"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "eval"), required=True)
    parser.add_argument("--miles-root", type=Path, required=True)
    parser.add_argument("--yeto-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-dir", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        result = launch(_parser().parse_args(argv))
    except Exception as error:
        print(f"TB2.1 rollout launch failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
