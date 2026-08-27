#!/usr/bin/env python3
"""Launch and fail-closed postcheck the 8 -> 64 TB2.1 baseline ramp.

The source plan is never modified.  A gate selects the unchanged first one or
eight rows from each of its eight baseline shards, then launches one GPU island
per shard with exact one-pass/no-replacement semantics.  Postcheck authenticates
every verifier outcome and requires the managed Terminal-Bench server to have
zero sessions, containers, or cleanup debt before the next gate may start.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.probes import launch_tbench21_rollout_wave as wave
from tools.probes import postcheck_tbench21_baseline_wave as baseline_postcheck


RAMP_PLAN_SCHEMA = "miles.tbench21-baseline-ramp-plan.v1"
RAMP_LAUNCH_SCHEMA = "miles.tbench21-baseline-ramp-launch.v1"
RAMP_REPORT_SCHEMA = "miles.tbench21-baseline-ramp-report.v1"
PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
GATE_SIZES = {8: 1, 64: 8}
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")
STATUS_PATH = "/miles/managed-status"
STATUS_MAX_BYTES = 64 * 1024
STATUS_CLEAN_TIMEOUT_SECONDS = 30.0
STATUS_POLL_INTERVAL_SECONDS = 0.25
MANAGED_LABEL = "miles.tbench21.managed=true"
MANAGED_RUN_LABEL = "miles.tbench21.run_id"
OUTCOME_KEY = "tbench_trusted_outcome"
MAC_KEY = "tbench_trusted_outcome_hmac"


class RampError(RuntimeError):
    """A ramp launch or completion gate failed."""


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=True, capture_output=True, text=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str, *, private: bool = False) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RampError(f"{name} must be an absolute regular non-symlink")
    if private and stat.S_IMODE(path.stat().st_mode) not in {0o400, 0o600}:
        raise RampError(f"{name} must have mode 0400 or 0600")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RampError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise RampError(f"{name} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RampError(f"output JSON must be a fresh absolute path: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _validate_run_id(value: str, name: str) -> str:
    if RUN_ID.fullmatch(value) is None:
        raise RampError(f"{name} must be a bounded lowercase letters/digits/hyphens ID")
    return value


def _source_plan(plan_dir: Path) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    plan_dir = wave._directory(plan_dir, "Terminal-Bench source plan")
    manifest_path = plan_dir / "manifest.json"
    manifest = _load_json(manifest_path, "Terminal-Bench source plan manifest")
    topology = manifest.get("topology")
    rollouts = manifest.get("rollouts")
    files = manifest.get("files")
    if (
        manifest.get("schema") != PLAN_SCHEMA
        or not isinstance(topology, dict)
        or topology.get("islands") != 8
        or topology.get("one_physical_gpu_per_island") is not True
        or not isinstance(rollouts, dict)
        or rollouts.get("baseline") != 356
        or rollouts.get("per_task") != 4
        or rollouts.get("episode_timeout_seconds") != 1800
        or rollouts.get("seed_base") != wave.ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != [wave.ROLLOUT_SEED_BASE + island_id for island_id in range(8)]
        or not isinstance(files, dict)
    ):
        raise RampError("source plan differs from the immutable full baseline contract")
    all_ids: set[str] = set()
    rows_by_island: list[list[dict[str, Any]]] = []
    expected_counts = [45, 45, 45, 45, 44, 44, 44, 44]
    for island_id, expected_count in enumerate(expected_counts):
        relative = f"baseline/island-{island_id}.jsonl"
        shard = plan_dir / relative
        if not shard.is_file() or shard.is_symlink() or files.get(relative) != _sha256(shard):
            raise RampError(f"source plan shard changed: {relative}")
        try:
            rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RampError(f"source plan shard is malformed: {relative}") from error
        if len(rows) != expected_count:
            raise RampError(f"source plan shard cardinality changed: {relative}")
        for row in rows:
            metadata = row.get("metadata") if isinstance(row, dict) else None
            messages = row.get("messages") if isinstance(row, dict) else None
            sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in all_ids
                or not isinstance(metadata.get("task_id"), str)
                or metadata.get("island_id") != island_id
                or metadata.get("rollout_seed") != wave.ROLLOUT_SEED_BASE + island_id
                or metadata.get("split") != "baseline"
                or metadata.get("episode_timeout_seconds") != 1800
                or not isinstance(messages, list)
                or not messages
            ):
                raise RampError(f"source plan row identity changed: {relative}")
            all_ids.add(sample_id)
        rows_by_island.append(rows)
    if len(all_ids) != 356:
        raise RampError("source plan does not contain 356 unique planned identities")
    return manifest, rows_by_island


def _build_ramp_plan(
    *,
    source_plan: Path,
    ramp_root: Path,
    gate_size: int,
    run_id: str,
) -> dict[str, Any]:
    per_island = GATE_SIZES.get(gate_size)
    if per_island is None:
        raise RampError("gate size must be exactly 8 or 64")
    source_plan = source_plan.resolve()
    source_manifest, source_rows = _source_plan(source_plan)
    if ramp_root.exists() or ramp_root.is_symlink():
        raise RampError("ramp plan directory must be fresh")
    (ramp_root / "baseline").mkdir(mode=0o700, parents=True)
    files: dict[str, str] = {}
    source_files: dict[str, str] = {}
    selected: dict[str, list[str]] = {}
    for island_id, rows in enumerate(source_rows):
        relative = f"baseline/island-{island_id}.jsonl"
        chosen = rows[:per_island]
        encoded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8") for row in chosen)
        destination = ramp_root / relative
        destination.write_bytes(encoded)
        destination.chmod(0o600)
        files[relative] = hashlib.sha256(encoded).hexdigest()
        source_files[relative] = str(source_manifest["files"][relative])
        selected[str(island_id)] = [row["metadata"]["sample_id"] for row in chosen]
    if len({sample_id for ids in selected.values() for sample_id in ids}) != gate_size:
        raise RampError("ramp selection does not contain the exact unique gate size")
    manifest = {
        "schema": RAMP_PLAN_SCHEMA,
        "gate_size": gate_size,
        "per_island": per_island,
        "run_id": run_id,
        "selection": "first-n-per-island",
        "source_plan_manifest_sha256": _sha256(source_plan / "manifest.json"),
        "source_plan_shards": source_files,
        "selected_sample_ids": selected,
        "files": files,
    }
    _write_json(ramp_root / "manifest.json", manifest)
    return manifest


def _managed_status(status_url: str, expected_run_id: str, request_timeout_seconds: float) -> tuple[int, dict[str, Any]]:
    if not status_url.startswith("http://") or not status_url.endswith(STATUS_PATH):
        raise RampError(f"managed status URL must be local HTTP ending in {STATUS_PATH}")
    request = urllib.request.Request(status_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            status = int(status_value)
            raw = response.read(STATUS_MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise RampError(f"managed server status returned HTTP {error.code}") from error
        status = error.code
        try:
            raw = error.read(STATUS_MAX_BYTES + 1)
        except OSError as read_error:
            raise RampError("managed Terminal-Bench status is unavailable") from read_error
    except (OSError, urllib.error.URLError) as error:
        raise RampError("managed Terminal-Bench status is unavailable") from error
    if status not in {200, 503}:
        raise RampError(f"managed server status returned HTTP {status}")
    if len(raw) > STATUS_MAX_BYTES:
        raise RampError("managed Terminal-Bench status response is unbounded")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RampError("managed Terminal-Bench status is not JSON") from error
    if not isinstance(payload, dict):
        raise RampError("managed Terminal-Bench status is malformed")
    count_fields = (
        "active_sessions",
        "managed_containers",
        "orphan_containers",
        "failed_cleanup_count",
    )
    if payload.get("run_id") != expected_run_id or not isinstance(payload.get("ok"), bool) or not isinstance(payload.get("cleanup_blocked"), bool) or any(isinstance(payload.get(field), bool) or not isinstance(payload.get(field), int) or payload[field] < 0 for field in count_fields):
        raise RampError(f"managed Terminal-Bench status has malformed or wrong-run scope: expected={expected_run_id!r}, payload={payload!r}")
    return status, payload


def _managed_container_ids(server_run_id: str) -> list[str]:
    output = _run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={MANAGED_LABEL}",
            "--filter",
            f"label={MANAGED_RUN_LABEL}={server_run_id}",
        ]
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def _clean_server_gate(
    status_url: str,
    server_run_id: str,
    *,
    timeout_seconds: float = STATUS_CLEAN_TIMEOUT_SECONDS,
    poll_interval_seconds: float = STATUS_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Bounded-poll only valid exact-run busy status through final teardown."""

    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise RampError("managed-status wait bounds must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_status: int | None = None
    last_payload: dict[str, Any] | None = None
    count_fields = (
        "active_sessions",
        "managed_containers",
        "orphan_containers",
        "failed_cleanup_count",
    )
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RampError(f"managed Terminal-Bench server did not become clean before the {timeout_seconds:g}s deadline: HTTP {last_status}, payload={last_payload!r}")
        status, payload = _managed_status(status_url, server_run_id, remaining)
        last_status = status
        last_payload = payload
        idle = payload["ok"] is True and payload["cleanup_blocked"] is False and payload.get("last_janitor_error") in {None, ""} and all(payload[field] == 0 for field in count_fields)
        if idle:
            if status != 200:
                raise RampError("managed server returned HTTP 503 for a clean idle status")
            leaked = _managed_container_ids(server_run_id)
            if leaked:
                raise RampError(f"managed Terminal-Bench task containers remain after cleanup: {leaked[:3]!r}")
            return payload
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval_seconds, remaining))


def _container_name(gate_size: int, run_id: str, island_id: int) -> str:
    return f"tbench21-ramp{gate_size}-{run_id}-island-{island_id}"


def _container_argv(
    *,
    gate_size: int,
    run_id: str,
    island_id: int,
    name: str,
    miles_root: Path,
    yeto_root: Path,
    model: Path,
    checkpoint: Path,
    source_plan: Path,
    ramp_plan: Path,
    output: Path,
    codex_dir: Path,
    hmac_key: Path,
    bridge_gateway: str,
) -> list[str]:
    per_island = GATE_SIZES[gate_size]
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--label",
        "yeto.run=tbench21-sao-ramp",
        "--label",
        f"yeto.ramp.run_id={run_id}",
        "--label",
        f"yeto.ramp.gate_size={gate_size}",
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
        f"type=bind,src={source_plan},dst=/root/source-plan,readonly",
        "--mount",
        f"type=bind,src={ramp_plan},dst=/root/ramp-plan,readonly",
        "--mount",
        f"type=bind,src={output},dst=/root/run-output",
        "--mount",
        f"type=bind,src={codex_dir},dst=/root/codex-cli,readonly",
        "--mount",
        f"type=bind,src={hmac_key},dst=/run/secrets/tbench-hmac,readonly",
        "--entrypoint",
        "python3",
        wave.RUNTIME_IMAGE,
        "/root/miles/tools/probes/run_tbench21_compaction_ramp.py",
        "--island-id",
        str(island_id),
        "--phase",
        "baseline",
        "--gate-size",
        str(gate_size),
        "--run-id",
        run_id,
        "--ramp-manifest",
        "/root/ramp-plan/manifest.json",
        "--source-plan",
        "/root/source-plan",
        "--prompt-data",
        f"/root/ramp-plan/baseline/island-{island_id}.jsonl",
        "--dump-details",
        "/root/run-output/details",
        "--hf-checkpoint",
        "/root/model",
        "--ref-load",
        "/root/input-checkpoint",
        "--codex-binary",
        f"/root/codex-cli/{wave.CODEX_BINARY_RELATIVE}",
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
        str(per_island),
        "--rollout-seed",
        str(wave.ROLLOUT_SEED_BASE),
    ]


def launch(args: argparse.Namespace) -> dict[str, Any]:
    gate_size = args.gate_size
    if gate_size not in GATE_SIZES:
        raise RampError("gate size must be exactly 8 or 64")
    run_id = _validate_run_id(args.run_id, "run_id")
    server_run_id = _validate_run_id(args.server_run_id, "server_run_id")
    miles_root = wave._directory(args.miles_root, "Miles source")
    yeto_root = wave._directory(args.yeto_root, "Yeto source")
    model = wave._directory(args.model, "Qwen model")
    checkpoint = wave._directory(args.checkpoint, "Megatron checkpoint")
    source_plan = wave._directory(args.plan_dir, "Terminal-Bench source plan")
    codex_dir = wave._directory(args.codex_dir, "Codex package")
    codex_binary = codex_dir / wave.CODEX_BINARY_RELATIVE
    if not codex_binary.is_file() or codex_binary.is_symlink():
        raise RampError("pinned Codex binary is missing")
    hmac_key = wave._private_key_file(args.hmac_key)
    outcome_source = yeto_root / "yeto/rl/tbench_outcome.py"
    if not outcome_source.is_file() or outcome_source.is_symlink():
        raise RampError("Yeto authenticated Terminal-Bench outcome verifier is missing")

    clean_before = _clean_server_gate(args.server_status_url, server_run_id)
    wave._image_gate()
    gpus = wave._gpu_gate()
    bridge_gateway = wave._bridge_gateway()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RampError("ramp output root must be fresh")
    names = [_container_name(gate_size, run_id, island_id) for island_id in range(8)]
    existing = [name for name in names if wave._container_exists(name)]
    if existing:
        raise RampError(f"ramp container names already exist: {existing}")

    output_root.mkdir(mode=0o700, parents=True)
    ramp_plan = output_root / "ramp-plan"
    plan_manifest = _build_ramp_plan(
        source_plan=source_plan,
        ramp_root=ramp_plan,
        gate_size=gate_size,
        run_id=run_id,
    )
    created: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for island_id, name in enumerate(names):
            output = output_root / f"island-{island_id}"
            output.mkdir(mode=0o700)
            argv = _container_argv(
                gate_size=gate_size,
                run_id=run_id,
                island_id=island_id,
                name=name,
                miles_root=miles_root,
                yeto_root=yeto_root,
                model=model,
                checkpoint=checkpoint,
                source_plan=source_plan,
                ramp_plan=ramp_plan,
                output=output,
                codex_dir=codex_dir,
                hmac_key=hmac_key,
                bridge_gateway=bridge_gateway,
            )
            container_id = _run(argv).stdout.strip()
            if len(container_id) != 64 or any(character not in "0123456789abcdef" for character in container_id):
                raise RampError(f"Docker returned an invalid container ID for {name}")
            created.append(name)
            shard = ramp_plan / f"baseline/island-{island_id}.jsonl"
            records.append(
                {
                    "island_id": island_id,
                    "name": name,
                    "container_id": container_id,
                    "plan_shard": str(shard),
                    "plan_shard_sha256": _sha256(shard),
                    "selected_sample_ids": plan_manifest["selected_sample_ids"][str(island_id)],
                    "output": str(output),
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
        "schema": RAMP_LAUNCH_SCHEMA,
        "gate_size": gate_size,
        "per_island": GATE_SIZES[gate_size],
        "run_id": run_id,
        "server_run_id": server_run_id,
        "server_status_url": args.server_status_url,
        "runtime_image": {"reference": wave.RUNTIME_IMAGE, "id": wave.RUNTIME_IMAGE_ID},
        "gpus": gpus,
        "model": str(model),
        "checkpoint": str(checkpoint),
        "source_plan": str(source_plan),
        "source_plan_manifest_sha256": _sha256(source_plan / "manifest.json"),
        "ramp_plan": str(ramp_plan),
        "ramp_plan_manifest_sha256": _sha256(ramp_plan / "manifest.json"),
        "codex_binary_sha256": _sha256(codex_binary),
        "outcome_verifier_sha256": _sha256(outcome_source),
        "hmac_key_file_mode": oct(stat.S_IMODE(hmac_key.stat().st_mode)),
        "per_island_rollout_seeds": [wave.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
        "managed_status_before_launch": clean_before,
        "containers": records,
    }
    manifest_path = output_root / "launch-manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "ok": True,
        "gate_size": gate_size,
        "manifest": str(manifest_path),
        "containers": names,
        "monitor": (f"watch -n 10 'docker ps -a --filter label=yeto.ramp.run_id={run_id} --format \"table {{{{.Names}}}}\\t{{{{.Status}}}}\"'"),
    }


def _docker_inspect(names: list[str]) -> list[dict[str, Any]]:
    try:
        value = json.loads(_run(["docker", "container", "inspect", *names]).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RampError("cannot inspect ramp containers") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RampError("Docker returned malformed ramp-container evidence")
    return value


def _load_outcome_api(yeto_root: Path, expected_sha256: str):
    source = yeto_root / "yeto/rl/tbench_outcome.py"
    if not source.is_file() or source.is_symlink() or _sha256(source) != expected_sha256:
        raise RampError("authenticated outcome verifier changed since launch")
    spec = importlib.util.spec_from_file_location("_ramp_tbench_outcome", source)
    if spec is None or spec.loader is None:
        raise RampError("cannot load authenticated outcome verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_signed_samples(
    payload: dict[str, Any],
    *,
    expected_rows: list[dict[str, Any]],
    key: bytes,
    outcome_api: Any,
) -> int:
    by_id = {row["metadata"]["sample_id"]: row["metadata"]["task_id"] for row in expected_rows}
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise RampError("native rollout payload has no samples")
    outcomes: dict[str, tuple[bytes, str]] = {}
    for index, raw in enumerate(samples):
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
        outcome = metadata.get(OUTCOME_KEY) if isinstance(metadata, dict) else None
        supplied = metadata.get(MAC_KEY) if isinstance(metadata, dict) else None
        if not isinstance(sample_id, str) or sample_id not in by_id or not isinstance(outcome, dict) or not isinstance(supplied, str):
            raise RampError(f"sample segment {index} has no planned signed outcome")
        try:
            expected_mac = outcome_api.sign_outcome(outcome, key=key)
        except Exception as error:
            raise RampError(f"sample segment {index} has an invalid signed outcome") from error
        reward = float(outcome["reward"])
        reported = metadata.get("reward")
        top_level_reward = raw.get("reward")
        if (
            not hmac.compare_digest(expected_mac, supplied)
            or outcome.get("sample_id") != sample_id
            or outcome.get("task_id") != by_id[sample_id]
            or outcome.get("status") != metadata.get("exit_status")
            or isinstance(reported, bool)
            or not isinstance(reported, (int, float))
            or float(reported) != reward
            or isinstance(top_level_reward, bool)
            or not isinstance(top_level_reward, (int, float))
            or float(top_level_reward) != reward
        ):
            raise RampError(f"sample segment {index} outcome is forged or off-plan")
        canonical = outcome_api.canonical_outcome(outcome)
        previous = outcomes.setdefault(sample_id, (canonical, supplied))
        if previous != (canonical, supplied):
            raise RampError(f"signed outcome changed across segments for {sample_id}")
    if set(outcomes) != set(by_id):
        raise RampError("signed outcomes do not cover the exact ramp sample IDs")
    return len(outcomes)


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    return baseline_postcheck._load_rollout_payload(path)


def postcheck(
    *,
    launch_manifest_path: Path,
    yeto_root: Path,
    hmac_key: Path,
) -> dict[str, Any]:
    launch_manifest_path = launch_manifest_path.expanduser()
    if not launch_manifest_path.is_absolute() or launch_manifest_path.is_symlink():
        raise RampError("launch manifest must be an absolute non-symlink")
    launch_manifest_path = launch_manifest_path.resolve()
    launch = _load_json(launch_manifest_path, "ramp launch manifest", private=True)
    gate_size = launch.get("gate_size")
    per_island = GATE_SIZES.get(gate_size)
    run_id = launch.get("run_id")
    server_run_id = launch.get("server_run_id")
    records = launch.get("containers")
    runtime_image = launch.get("runtime_image")
    if (
        per_island is None
        or not isinstance(run_id, str)
        or RUN_ID.fullmatch(run_id) is None
        or not isinstance(server_run_id, str)
        or RUN_ID.fullmatch(server_run_id) is None
        or launch.get("schema") != RAMP_LAUNCH_SCHEMA
        or launch.get("per_island") != per_island
        or not isinstance(runtime_image, dict)
        or runtime_image.get("id") != wave.RUNTIME_IMAGE_ID
        or launch.get("per_island_rollout_seeds") != [wave.ROLLOUT_SEED_BASE + island_id for island_id in range(8)]
        or not isinstance(records, list)
        or len(records) != 8
    ):
        raise RampError("launch manifest differs from the exact ramp contract")
    yeto_root = wave._directory(yeto_root, "Yeto source")
    key_path = wave._private_key_file(hmac_key)
    key = key_path.read_bytes().rstrip(b"\r\n")
    outcome_api = _load_outcome_api(yeto_root, str(launch.get("outcome_verifier_sha256", "")))

    source_plan = Path(str(launch.get("source_plan", "")))
    ramp_plan = Path(str(launch.get("ramp_plan", "")))
    if (
        not source_plan.is_absolute()
        or source_plan.is_symlink()
        or not source_plan.is_dir()
        or _sha256(source_plan / "manifest.json") != launch.get("source_plan_manifest_sha256")
        or not ramp_plan.is_absolute()
        or ramp_plan.is_symlink()
        or not ramp_plan.is_dir()
        or _sha256(ramp_plan / "manifest.json") != launch.get("ramp_plan_manifest_sha256")
    ):
        raise RampError("source or derived ramp plan changed after launch")
    ramp = _load_json(ramp_plan / "manifest.json", "ramp plan manifest", private=True)
    if ramp.get("schema") != RAMP_PLAN_SCHEMA or ramp.get("gate_size") != gate_size or ramp.get("per_island") != per_island or ramp.get("run_id") != run_id or ramp.get("source_plan_manifest_sha256") != launch.get("source_plan_manifest_sha256"):
        raise RampError("derived ramp plan is not bound to this launch")

    names = [_container_name(gate_size, run_id, island_id) for island_id in range(8)]
    inspections = _docker_inspect(names)
    by_name = {str(item.get("Name", "")).removeprefix("/"): item for item in inspections}
    if len(inspections) != 8 or len(by_name) != 8:
        raise RampError("Docker did not return exactly eight ramp containers")
    completed = []
    island_reports = []
    for island_id, record in enumerate(records):
        name = names[island_id]
        item = by_name.get(name)
        state = item.get("State") if isinstance(item, dict) else None
        config = item.get("Config") if isinstance(item, dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        container_id = record.get("container_id") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("island_id") != island_id
            or record.get("name") != name
            or not isinstance(container_id, str)
            or len(container_id) != 64
            or not isinstance(item, dict)
            or item.get("Id") != container_id
            or item.get("Image") != wave.RUNTIME_IMAGE_ID
            or not isinstance(state, dict)
            or state.get("Status") != "exited"
            or state.get("Running") is not False
            or state.get("Dead") is not False
            or state.get("OOMKilled") is not False
            or state.get("ExitCode") != 0
            or state.get("Error") not in {"", None}
            or not isinstance(labels, dict)
            or labels.get("yeto.run") != "tbench21-sao-ramp"
            or labels.get("yeto.ramp.run_id") != run_id
            or labels.get("yeto.ramp.gate_size") != str(gate_size)
            or labels.get("yeto.island") != str(island_id)
        ):
            raise RampError(f"ramp container {island_id} did not exit cleanly")
        shard = ramp_plan / f"baseline/island-{island_id}.jsonl"
        if record.get("plan_shard") != str(shard) or record.get("plan_shard_sha256") != _sha256(shard) or ramp.get("files", {}).get(f"baseline/island-{island_id}.jsonl") != _sha256(shard):
            raise RampError(f"ramp island {island_id} plan binding changed")
        try:
            rows = [json.loads(line) for line in shard.read_text().splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RampError(f"ramp island {island_id} shard is malformed") from error
        selected_ids = [row["metadata"]["sample_id"] for row in rows]
        if len(rows) != per_island or selected_ids != record.get("selected_sample_ids") or selected_ids != ramp.get("selected_sample_ids", {}).get(str(island_id)):
            raise RampError(f"ramp island {island_id} selected identities changed")
        output = launch_manifest_path.parent / f"island-{island_id}"
        if record.get("output") != str(output) or not output.is_dir() or output.is_symlink():
            raise RampError(f"ramp island {island_id} output path changed")
        rollout_dir = output / "details/rollout_data"
        source = rollout_dir / "0.pt"
        if not rollout_dir.is_dir() or rollout_dir.is_symlink() or sorted(rollout_dir.iterdir(), key=lambda path: path.name) != [source] or not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise RampError(f"ramp island {island_id} must contain exactly details/rollout_data/0.pt")
        payload = _load_rollout_payload(source)
        expected = {sample_id: ordinal for ordinal, sample_id in enumerate(selected_ids)}
        try:
            base_report = baseline_postcheck._validate_payload(
                payload,
                island_id=island_id,
                expected=expected,
                source=source,
            )
        except baseline_postcheck.BaselinePostcheckError as error:
            raise RampError(f"ramp island {island_id} exact-one-pass check failed") from error
        authenticated = _verify_signed_samples(
            payload,
            expected_rows=rows,
            key=key,
            outcome_api=outcome_api,
        )
        if authenticated != per_island:
            raise RampError(f"ramp island {island_id} signed count changed")
        completed.append(
            {
                "island_id": island_id,
                "name": name,
                "container_id": container_id,
                "exit_code": 0,
                "oom_killed": False,
            }
        )
        island_reports.append({**base_report, "authenticated_outcomes": authenticated})

    clean_after = _clean_server_gate(str(launch.get("server_status_url", "")), server_run_id)
    total = sum(report["authenticated_outcomes"] for report in island_reports)
    if total != gate_size:
        raise RampError("ramp did not authenticate the exact gate trajectory count")
    return {
        "schema": RAMP_REPORT_SCHEMA,
        "ok": True,
        "gate_size": gate_size,
        "run_id": run_id,
        "server_run_id": server_run_id,
        "authenticated_outcomes": total,
        "logical_trajectories": sum(report["logical_trajectories"] for report in island_reports),
        "containers": completed,
        "islands": island_reports,
        "managed_status_after": clean_after,
        "task_container_leaks": 0,
        "infrastructure_aborted_replacements": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--gate-size", type=int, choices=(8, 64), required=True)
    launch_parser.add_argument("--run-id", required=True)
    launch_parser.add_argument("--server-run-id", required=True)
    launch_parser.add_argument(
        "--server-status-url",
        default="http://127.0.0.1:8003/miles/managed-status",
    )
    launch_parser.add_argument("--miles-root", type=Path, required=True)
    launch_parser.add_argument("--yeto-root", type=Path, required=True)
    launch_parser.add_argument("--model", type=Path, required=True)
    launch_parser.add_argument("--checkpoint", type=Path, required=True)
    launch_parser.add_argument("--plan-dir", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--codex-dir", type=Path, required=True)
    launch_parser.add_argument("--hmac-key", type=Path, required=True)

    check_parser = commands.add_parser("postcheck")
    check_parser.add_argument("--launch-manifest", type=Path, required=True)
    check_parser.add_argument("--yeto-root", type=Path, required=True)
    check_parser.add_argument("--hmac-key", type=Path, required=True)
    check_parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "launch":
            result = launch(args)
        else:
            result = postcheck(
                launch_manifest_path=args.launch_manifest,
                yeto_root=args.yeto_root,
                hmac_key=args.hmac_key,
            )
            if args.report is not None:
                _write_json(args.report.expanduser(), result)
    except Exception as error:
        print(f"TB2.1 ramp {args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
