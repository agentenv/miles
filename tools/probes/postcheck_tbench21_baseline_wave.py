#!/usr/bin/env python3
"""Fail-closed completion check for the eight-island TB2.1 baseline wave.

Run this on the trusted host (or its trusted staging container with Docker
access) immediately after all baseline containers stop.  It checks container
identity/exit state and the native Miles rollout shards without converting or
mutating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ISLANDS = 8
ROLLOUT_SEED_BASE = 82621
RUNTIME_IMAGE_ID = "sha256:69be75252eea4179ccae554f362351a7502f365a89c99eaca322df19f4e55572"
PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
LAUNCH_SCHEMA = "miles.tbench21-rollout-wave.v1"
REPORT_SCHEMA = "miles.tbench21-baseline-postcheck.v1"
TERMINAL_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})


class BaselinePostcheckError(RuntimeError):
    """The completed baseline wave is not the exact immutable run."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str, *, private: bool = False) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BaselinePostcheckError(f"{name} must be an absolute regular non-symlink")
    if private and stat.S_IMODE(path.stat().st_mode) not in {0o400, 0o600}:
        raise BaselinePostcheckError(f"{name} must have mode 0400 or 0600")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BaselinePostcheckError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise BaselinePostcheckError(f"{name} must contain a JSON object")
    return value


def _plan_rows(plan_dir: Path) -> tuple[list[dict[str, int]], str]:
    if not plan_dir.is_absolute() or plan_dir.is_symlink() or not plan_dir.is_dir():
        raise BaselinePostcheckError("plan directory must be an absolute real directory")
    plan_dir = plan_dir.resolve()
    manifest_path = plan_dir / "manifest.json"
    plan = _load_json(manifest_path, "Terminal-Bench plan")
    topology = plan.get("topology")
    rollouts = plan.get("rollouts")
    files = plan.get("files")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(topology, dict)
        or topology.get("islands") != ISLANDS
        or topology.get("one_physical_gpu_per_island") is not True
        or not isinstance(rollouts, dict)
        or rollouts.get("baseline") != 356
        or rollouts.get("per_task") != 4
        or rollouts.get("episode_timeout_seconds") != 1800
        or rollouts.get("seed_base") != ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != [ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)]
        or not isinstance(files, dict)
    ):
        raise BaselinePostcheckError("plan differs from the full baseline contract")

    expected_counts = [45, 45, 45, 45, 44, 44, 44, 44]
    all_ids: set[str] = set()
    contracts: list[dict[str, int]] = []
    for island_id, expected_count in enumerate(expected_counts):
        relative = f"baseline/island-{island_id}.jsonl"
        shard = plan_dir / relative
        digest = files.get(relative)
        if not isinstance(digest, str) or len(digest) != 64 or not shard.is_file() or shard.is_symlink() or _sha256(shard) != digest:
            raise BaselinePostcheckError(f"plan shard changed: {relative}")
        rows: list[dict[str, Any]] = []
        try:
            for line in shard.read_text(encoding="utf-8").splitlines():
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError
                    rows.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            raise BaselinePostcheckError(f"plan shard is malformed: {relative}") from error
        if len(rows) != expected_count:
            raise BaselinePostcheckError(f"island {island_id} plan must contain exactly {expected_count} rows")
        expected: dict[str, int] = {}
        for ordinal, row in enumerate(rows):
            metadata = row.get("metadata")
            sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
            if not isinstance(sample_id, str) or not sample_id or sample_id in expected or sample_id in all_ids or metadata.get("island_id") != island_id or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id or metadata.get("split") != "baseline" or metadata.get("episode_timeout_seconds") != 1800:
                raise BaselinePostcheckError(f"island {island_id} plan row identity changed")
            expected[sample_id] = ordinal
            all_ids.add(sample_id)
        contracts.append(expected)
    if len(all_ids) != 356:
        raise BaselinePostcheckError("baseline plan does not contain 356 unique IDs")
    return contracts, _sha256(manifest_path)


def _docker_inspect(names: list[str]) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["docker", "container", "inspect", *names],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise BaselinePostcheckError("cannot inspect baseline containers") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise BaselinePostcheckError("Docker returned malformed container evidence")
    return value


def _validate_containers(records: list[dict[str, Any]], inspections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(item.get("Name", "")).removeprefix("/"): item for item in inspections}
    if len(by_name) != ISLANDS or len(inspections) != ISLANDS:
        raise BaselinePostcheckError("Docker did not return exactly eight containers")
    completed: list[dict[str, Any]] = []
    for island_id, record in enumerate(records):
        if not isinstance(record, dict):
            raise BaselinePostcheckError("baseline launch container record is malformed")
        name = f"tbench21-baseline-island-{island_id}"
        expected_id = record.get("container_id")
        item = by_name.get(name)
        state = item.get("State") if isinstance(item, dict) else None
        config = item.get("Config") if isinstance(item, dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        if (
            record.get("island_id") != island_id
            or record.get("name") != name
            or not isinstance(expected_id, str)
            or len(expected_id) != 64
            or any(character not in "0123456789abcdef" for character in expected_id)
            or not isinstance(item, dict)
            or item.get("Id") != expected_id
            or item.get("Image") != RUNTIME_IMAGE_ID
            or not isinstance(state, dict)
            or state.get("Status") != "exited"
            or state.get("Running") is not False
            or state.get("Dead") is not False
            or state.get("OOMKilled") is not False
            or state.get("ExitCode") != 0
            or state.get("Error") not in {"", None}
            or not isinstance(labels, dict)
            or labels.get("yeto.run") != "tbench21-sao-rollout"
            or labels.get("yeto.phase") != "baseline"
            or labels.get("yeto.island") != str(island_id)
        ):
            raise BaselinePostcheckError(f"baseline container {island_id} did not exit cleanly with its launch identity")
        completed.append(
            {
                "island_id": island_id,
                "name": name,
                "container_id": expected_id,
                "exit_code": 0,
                "oom_killed": False,
            }
        )
    return completed


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise BaselinePostcheckError(f"cannot load native Miles shard: {path}") from error
    if not isinstance(value, dict):
        raise BaselinePostcheckError(f"native Miles shard is not an object: {path}")
    return value


def _replacement_marker(metadata: dict[str, Any]) -> str | None:
    for key, value in metadata.items():
        if "replacement" not in str(key).lower():
            continue
        inactive = value is None or value is False or (type(value) is int and value == 0) or (isinstance(value, str) and value == "") or (isinstance(value, (list, dict)) and not value)
        if not inactive:
            return str(key)
    return None


def _validate_payload(
    payload: dict[str, Any],
    *,
    island_id: int,
    expected: dict[str, int],
    source: Path,
) -> dict[str, Any]:
    samples = payload.get("samples")
    if payload.get("rollout_id") != 0 or not isinstance(payload.get("metadata"), dict) or not isinstance(samples, list) or not samples:
        raise BaselinePostcheckError(f"island {island_id} does not contain one native rollout_id=0 batch")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    sample_indices: dict[str, int] = {}
    retry_ledgers: dict[str, int] = {}
    trajectory_ids: set[str] = set()
    for ordinal, raw in enumerate(samples):
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
        index = raw.get("index") if isinstance(raw, dict) else None
        group_index = raw.get("group_index") if isinstance(raw, dict) else None
        status = raw.get("status") if isinstance(raw, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(sample_id, str)
            or sample_id not in expected
            or type(index) is not int
            or index != expected[sample_id]
            or type(group_index) is not int
            or group_index != expected[sample_id]
            or status not in {"completed", "truncated"}
            or metadata.get("island_id") != island_id
            or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
            or metadata.get("split") != "baseline"
            or metadata.get("episode_timeout_seconds") != 1800
            or metadata.get("exit_status") not in TERMINAL_STATUSES
        ):
            raise BaselinePostcheckError(f"{source}:{ordinal}: accepted sample is cycled, replaced, aborted, or off-plan")
        marker = _replacement_marker(metadata)
        if marker is not None:
            raise BaselinePostcheckError(f"{source}:{ordinal}: accepted sample carries replacement marker {marker}")
        retry_count = metadata.get("secrlenv_infrastructure_503_retries", 0)
        if type(retry_count) is not int or retry_count not in {0, 1}:
            raise BaselinePostcheckError(f"{source}:{ordinal}: invalid infrastructure retry ledger")
        prior_retry = retry_ledgers.setdefault(sample_id, retry_count)
        if prior_retry != retry_count:
            raise BaselinePostcheckError(f"{source}:{ordinal}: infrastructure retry ledger changed across segments")
        prior = sample_indices.setdefault(sample_id, index)
        if prior != index:
            raise BaselinePostcheckError(f"{source}:{ordinal}: one trajectory used multiple sampler indices")
        by_sample.setdefault(sample_id, []).append(raw)

    if set(by_sample) != set(expected) or len(sample_indices) != len(expected):
        missing = sorted(set(expected) - set(by_sample))
        extra = sorted(set(by_sample) - set(expected))
        raise BaselinePostcheckError(f"island {island_id} logical coverage changed: missing={missing[:3]!r}, extra={extra[:3]!r}")
    if set(sample_indices.values()) != set(range(len(expected))):
        raise BaselinePostcheckError(f"island {island_id} used cycled/replacement sampler indices")

    segment_count = 0
    for sample_id, rows in by_sample.items():
        rows.sort(key=lambda row: row["metadata"].get("compaction_segment_index", -1))
        metadata_rows = [row["metadata"] for row in rows]
        indices = [metadata.get("compaction_segment_index") for metadata in metadata_rows]
        types = [metadata.get("compaction_segment_type") for metadata in metadata_rows]
        identity_values = [metadata.get("compaction_trajectory_id") for metadata in metadata_rows]
        identities = set(identity_values) if all(isinstance(value, str) for value in identity_values) else set()
        if not 1 <= len(rows) <= 7 or indices != list(range(len(rows))) or types != ["execution" if index % 2 == 0 else "summary" for index in indices] or types[-1] != "execution" or len(identities) != 1 or not isinstance(next(iter(identities)), str) or not next(iter(identities)):
            raise BaselinePostcheckError(f"{source}: {sample_id} has duplicated or malformed compaction segments")
        trajectory_id = next(iter(identities))
        if trajectory_id in trajectory_ids:
            raise BaselinePostcheckError(f"{source}: two logical samples reused one trajectory ID")
        trajectory_ids.add(trajectory_id)
        segment_count += len(rows)
    return {
        "island_id": island_id,
        "rollout_path": str(source),
        "rollout_sha256": _sha256(source),
        "logical_trajectories": len(by_sample),
        "segments": segment_count,
        "sampler_indices": [0, len(expected) - 1],
        "trajectories_with_infrastructure_503_retry": sum(retry_ledgers.values()),
        "infrastructure_aborted_replacements": 0,
    }


def postcheck(launch_manifest_path: Path, plan_dir: Path) -> dict[str, Any]:
    launch_manifest_path = launch_manifest_path.expanduser()
    if not launch_manifest_path.is_absolute() or launch_manifest_path.is_symlink():
        raise BaselinePostcheckError("launch manifest path must be absolute and non-symlink")
    launch_manifest_path = launch_manifest_path.resolve()
    launch = _load_json(launch_manifest_path, "baseline launch manifest", private=True)
    records = launch.get("containers")
    runtime_image = launch.get("runtime_image")
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("phase") != "baseline"
        or not isinstance(runtime_image, dict)
        or runtime_image.get("id") != RUNTIME_IMAGE_ID
        or launch.get("per_island_rollout_seeds") != [ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)]
        or not isinstance(records, list)
        or len(records) != ISLANDS
    ):
        raise BaselinePostcheckError("launch manifest differs from the baseline wave")
    plan_dir = plan_dir.expanduser()
    if not plan_dir.is_absolute() or plan_dir.is_symlink():
        raise BaselinePostcheckError("plan directory must be absolute and non-symlink")
    plan_dir = plan_dir.resolve()
    if launch.get("plan_dir") != str(plan_dir):
        raise BaselinePostcheckError("launch manifest is not bound to this plan directory")
    expected_by_island, plan_sha256 = _plan_rows(plan_dir)
    names = [f"tbench21-baseline-island-{island_id}" for island_id in range(ISLANDS)]
    completed = _validate_containers(records, _docker_inspect(names))

    output_root = launch_manifest_path.parent
    island_reports = []
    for island_id, expected in enumerate(expected_by_island):
        record = records[island_id]
        output = output_root / f"island-{island_id}"
        if record.get("output") != str(output) or not output.is_dir() or output.is_symlink():
            raise BaselinePostcheckError(f"island {island_id} output path changed after launch")
        shard = plan_dir / f"baseline/island-{island_id}.jsonl"
        if record.get("plan_shard") != str(shard) or record.get("plan_shard_sha256") != _sha256(shard):
            raise BaselinePostcheckError(f"island {island_id} launch is not bound to its plan shard")
        rollout_dir = output / "details/rollout_data"
        source = rollout_dir / "0.pt"
        if not rollout_dir.is_dir() or rollout_dir.is_symlink() or sorted(rollout_dir.iterdir(), key=lambda path: path.name) != [source] or not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise BaselinePostcheckError(f"island {island_id} must contain exactly details/rollout_data/0.pt")
        island_reports.append(
            _validate_payload(
                _load_rollout_payload(source),
                island_id=island_id,
                expected=expected,
                source=source,
            )
        )
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "launch_manifest": {
            "path": str(launch_manifest_path),
            "sha256": _sha256(launch_manifest_path),
        },
        "plan_manifest": {
            "path": str(plan_dir / "manifest.json"),
            "sha256": plan_sha256,
        },
        "containers": completed,
        "islands": island_reports,
        "logical_trajectories": sum(report["logical_trajectories"] for report in island_reports),
        "trajectories_with_infrastructure_503_retry": sum(report["trajectories_with_infrastructure_503_retry"] for report in island_reports),
        "infrastructure_aborted_replacements": 0,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise BaselinePostcheckError("report path must be absolute and fresh")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = postcheck(args.launch_manifest, args.plan_dir)
        if args.report is not None:
            _write_report(args.report.expanduser(), report)
    except Exception as error:
        print(
            f"TB2.1 baseline postcheck failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
