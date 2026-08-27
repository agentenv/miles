#!/usr/bin/env python3
"""Fail-closed completion check for the eight-island TB2.1 held-out eval.

The checker is deliberately read-only.  It binds the exact immutable eval
split, launch/container identities, final actor checkpoint mount, native Miles
rollout shards, authenticated Terminal-Bench outcomes, and the managed Docker
server's empty post-run inventory.  Rewards are aggregated once per logical
trajectory even when learned compaction emitted multiple training segments.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import numbers
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import postcheck_tbench21_baseline_wave as baseline
else:
    import postcheck_tbench21_baseline_wave as baseline


ISLANDS = 8
ROLLOUT_SEED_BASE = baseline.ROLLOUT_SEED_BASE
RUNTIME_IMAGE_ID = baseline.RUNTIME_IMAGE_ID
PLAN_SCHEMA = baseline.PLAN_SCHEMA
LAUNCH_SCHEMA = baseline.LAUNCH_SCHEMA
REPORT_SCHEMA = "miles.tbench21-eval-postcheck.v1"
EXPECTED_COUNTS = (23, 23, 23, 23, 22, 22, 22, 22)
EXPECTED_TRAJECTORIES = 180
EXPECTED_TASKS = 45
ROLLOUTS_PER_TASK = 4
EPISODE_TIMEOUT_SECONDS = 1800
MAX_SEQ_LEN = 8192
MANAGED_STATUS_URL = "http://127.0.0.1:8003"
MANAGED_CONTAINER_URL = "http://host.docker.internal:8003"
MANAGED_SERVER_SCRIPT = "/root/miles/examples/experimental/openenv/managed_tbench21_server.py"
MANAGED_MAX_CONCURRENCY = 304
RUNNER = "/root/miles/tools/probes/run_tbench21_compaction_baseline.py"
CODEX_BINARY = "/root/codex-cli/vendor/x86_64-unknown-linux-musl/bin/codex"
PLAN_ALGORITHM = "sha256-domain-ranked-first-44"
CHECKPOINT_SCHEMA = "miles.tbench21-eval-checkpoint-inventory.v1"
TERMINAL_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})
OUTCOME_FIELDS = frozenset(
    {
        "schema",
        "benchmark",
        "task_id",
        "sample_id",
        "episode_id",
        "status",
        "reward",
        "passed",
        "verifier",
        "testsh_rc",
    }
)
OUTCOME_STATUSES = TERMINAL_STATUSES
OUTCOME_VERIFIERS = frozenset({"not_run_timeout", "openenv_native_evaluate", "terminal_bench_test_sh"})
OUTCOME_DOMAIN = b"yeto-tbench-outcome-v1\0"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+@-]{0,511}\Z")
MAC = re.compile(r"[0-9a-f]{64}\Z")
TITO_COUNT_DETAIL = re.compile(r"segment count differs: expected ([0-9]+), got ([0-9]+)\Z")


class EvalPostcheckError(RuntimeError):
    """The held-out evaluation does not satisfy its immutable contract."""


@dataclass(frozen=True)
class PlannedSample:
    task_id: str
    replica: int
    index: int


@dataclass(frozen=True)
class LogicalResult:
    sample_id: str
    task_id: str
    replica: int
    reward: float
    status: str
    verifier: str
    episode_id: str
    trajectory_id: str
    segments: int
    terminal_tito_boundary_mismatch: bool


def _load_hmac_key(path: Path) -> tuple[Path, bytes]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EvalPostcheckError("Terminal-Bench HMAC key must be an absolute regular non-symlink")
    path = path.resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode not in {0o400, 0o600}:
        raise EvalPostcheckError("Terminal-Bench HMAC key must have mode 0400 or 0600")
    try:
        key = path.read_bytes().rstrip(b"\r\n")
    except OSError as error:
        raise EvalPostcheckError("Terminal-Bench HMAC key is unreadable") from error
    if not 32 <= len(key) <= 4096:
        raise EvalPostcheckError("Terminal-Bench HMAC key must contain 32..4096 bytes")
    return path, key


def _finite_binary(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or float(value) not in {0.0, 1.0}:
        raise EvalPostcheckError(f"{name} must be a finite binary reward")
    return float(value)


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvalPostcheckError("attested value is not canonicalizable") from error
    return hashlib.sha256(encoded).hexdigest()


def _verified_outcome(
    metadata: dict[str, Any],
    *,
    key: bytes,
    sample_id: str,
    task_id: str,
) -> tuple[dict[str, Any], float]:
    if any(name in metadata for name in ("secrlenv_trusted_outcome", "secrlenv_trusted_outcome_hmac")):
        raise EvalPostcheckError("evaluation metadata mixes benchmark evidence kinds")
    outcome = metadata.get("tbench_trusted_outcome")
    supplied_mac = metadata.get("tbench_trusted_outcome_hmac")
    if not isinstance(outcome, dict) or set(outcome) != OUTCOME_FIELDS:
        raise EvalPostcheckError("evaluation trajectory has no closed signed outcome")
    if not isinstance(supplied_mac, str) or MAC.fullmatch(supplied_mac) is None:
        raise EvalPostcheckError("evaluation trajectory has no valid outcome MAC")
    if outcome.get("schema") != 1 or outcome.get("benchmark") != "terminal-bench-2.1":
        raise EvalPostcheckError("evaluation outcome schema or benchmark differs")
    for name in ("task_id", "sample_id", "episode_id"):
        value = outcome.get(name)
        if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
            raise EvalPostcheckError(f"evaluation outcome has an invalid {name}")
    if outcome["sample_id"] != sample_id or outcome["task_id"] != task_id:
        raise EvalPostcheckError("signed evaluation task/sample identity differs")
    status = outcome.get("status")
    verifier = outcome.get("verifier")
    if status not in OUTCOME_STATUSES or verifier not in OUTCOME_VERIFIERS:
        raise EvalPostcheckError("evaluation outcome status or verifier is invalid")
    reward = _finite_binary(outcome.get("reward"), name="signed outcome reward")
    if type(outcome.get("passed")) is not bool or outcome["passed"] != (reward == 1.0):
        raise EvalPostcheckError("evaluation outcome pass bit differs from reward")
    testsh_rc = outcome.get("testsh_rc")
    if status == "timeout":
        if reward != 0.0 or verifier != "not_run_timeout" or testsh_rc is not None:
            raise EvalPostcheckError("timeout outcome claims an invalid verifier verdict")
    elif verifier == "not_run_timeout":
        raise EvalPostcheckError("non-timeout outcome has no verifier verdict")
    elif verifier == "openenv_native_evaluate":
        if testsh_rc is not None:
            raise EvalPostcheckError("native verifier outcome has a test.sh status")
    elif isinstance(testsh_rc, bool) or not isinstance(testsh_rc, int) or not 0 <= testsh_rc <= 255:
        raise EvalPostcheckError("test.sh verifier outcome has an invalid exit status")
    try:
        canonical = json.dumps(
            outcome,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvalPostcheckError("evaluation outcome is not canonicalizable") from error
    expected_mac = hmac.new(key, OUTCOME_DOMAIN + canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_mac, supplied_mac):
        raise EvalPostcheckError("evaluation outcome signature mismatch")
    return outcome, reward


def _plan_rows(
    plan_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[list[dict[str, PlannedSample]], str, set[str]]:
    if not plan_dir.is_absolute() or plan_dir.is_symlink() or not plan_dir.is_dir():
        raise EvalPostcheckError("plan directory must be an absolute real directory")
    plan_dir = plan_dir.resolve()
    manifest_path = plan_dir / "manifest.json"
    if MAC.fullmatch(expected_manifest_sha256) is None:
        raise EvalPostcheckError("expected plan manifest SHA-256 is malformed")
    manifest_sha256 = baseline._sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise EvalPostcheckError("plan manifest is not the exact preapproved held-out plan")
    plan = baseline._load_json(manifest_path, "Terminal-Bench plan")
    topology = plan.get("topology")
    rollouts = plan.get("rollouts")
    split = plan.get("split")
    files = plan.get("files")
    terminal_bench = plan.get("terminal_bench")
    task_contracts = terminal_bench.get("task_contracts") if isinstance(terminal_bench, dict) else None
    train_tasks = split.get("train_task_ids") if isinstance(split, dict) else None
    eval_tasks = split.get("eval_task_ids") if isinstance(split, dict) else None
    split_payload = {
        "seed": split.get("seed") if isinstance(split, dict) else None,
        "algorithm": split.get("algorithm") if isinstance(split, dict) else None,
        "train_task_ids": train_tasks,
        "eval_task_ids": eval_tasks,
    }
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(topology, dict)
        or topology.get("islands") != ISLANDS
        or topology.get("one_physical_gpu_per_island") is not True
        or not isinstance(rollouts, dict)
        or rollouts.get("eval") != EXPECTED_TRAJECTORIES
        or rollouts.get("per_task") != ROLLOUTS_PER_TASK
        or rollouts.get("episode_timeout_seconds") != EPISODE_TIMEOUT_SECONDS
        or rollouts.get("seed_base") != ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != [ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)]
        or not isinstance(split, dict)
        or split.get("algorithm") != PLAN_ALGORITHM
        or not isinstance(split.get("seed"), str)
        or not split["seed"]
        or not isinstance(train_tasks, list)
        or not isinstance(eval_tasks, list)
        or len(train_tasks) != 44
        or len(eval_tasks) != EXPECTED_TASKS
        or any(not isinstance(task_id, str) or not task_id for task_id in train_tasks + eval_tasks)
        or len(set(train_tasks)) != 44
        or len(set(eval_tasks)) != EXPECTED_TASKS
        or bool(set(train_tasks) & set(eval_tasks))
        or split.get("train_task_count") != 44
        or split.get("eval_task_count") != EXPECTED_TASKS
        or split.get("sha256") != _canonical_sha256(split_payload)
        or not isinstance(terminal_bench, dict)
        or terminal_bench.get("version") != "2.1"
        or terminal_bench.get("task_count") != 89
        or not isinstance(task_contracts, dict)
        or len(task_contracts) != 89
        or set(train_tasks) | set(eval_tasks) != set(task_contracts)
        or terminal_bench.get("task_inventory_sha256") != _canonical_sha256(task_contracts)
        or not isinstance(files, dict)
    ):
        raise EvalPostcheckError("plan differs from the held-out evaluation contract")

    all_sample_ids: set[str] = set()
    replicas_by_task: dict[str, set[int]] = defaultdict(set)
    contracts: list[dict[str, PlannedSample]] = []
    for island_id, expected_count in enumerate(EXPECTED_COUNTS):
        relative = f"eval/island-{island_id}.jsonl"
        shard = plan_dir / relative
        digest = files.get(relative)
        if not isinstance(digest, str) or len(digest) != 64 or not shard.is_file() or shard.is_symlink() or baseline._sha256(shard) != digest:
            raise EvalPostcheckError(f"plan shard changed: {relative}")
        rows: list[dict[str, Any]] = []
        try:
            for line in shard.read_text(encoding="utf-8").splitlines():
                if not line:
                    raise TypeError
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError
                rows.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            raise EvalPostcheckError(f"plan shard is malformed: {relative}") from error
        if len(rows) != expected_count:
            raise EvalPostcheckError(f"island {island_id} eval plan must contain exactly {expected_count} rows")
        expected: dict[str, PlannedSample] = {}
        for index, row in enumerate(rows):
            metadata = row.get("metadata")
            messages = row.get("messages")
            task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
            replica = metadata.get("rollout_replica") if isinstance(metadata, dict) else None
            sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(task_id, str)
                or not task_id
                or type(replica) is not int
                or replica not in range(ROLLOUTS_PER_TASK)
                or sample_id != f"eval:{task_id}:r{replica}"
                or sample_id in expected
                or sample_id in all_sample_ids
                or metadata.get("split") != "eval"
                or metadata.get("island_id") != island_id
                or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
                or metadata.get("episode_timeout_seconds") != EPISODE_TIMEOUT_SECONDS
                or metadata.get("max_seq_len") != MAX_SEQ_LEN
                or metadata.get("prompt_tier") != "l2"
            ):
                raise EvalPostcheckError(f"island {island_id} eval plan row identity changed")
            expected[sample_id] = PlannedSample(task_id, replica, index)
            all_sample_ids.add(sample_id)
            replicas_by_task[task_id].add(replica)
        contracts.append(expected)
    if len(all_sample_ids) != EXPECTED_TRAJECTORIES:
        raise EvalPostcheckError("eval plan does not contain 180 unique sample IDs")
    if len(replicas_by_task) != EXPECTED_TASKS or any(replicas != set(range(ROLLOUTS_PER_TASK)) for replicas in replicas_by_task.values()):
        raise EvalPostcheckError("eval plan is not exactly 45 tasks with four replicas each")
    if set(replicas_by_task) != set(eval_tasks):
        raise EvalPostcheckError("eval shards do not contain exactly split.eval_task_ids")
    return contracts, manifest_sha256, set(replicas_by_task)


def _checkpoint_inventory(path: Path) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if item.is_symlink():
            raise EvalPostcheckError("actor checkpoint inventory contains a symlink")
        if item.is_file():
            entries.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": item.stat().st_size,
                    "sha256": baseline._sha256(item),
                }
            )
    if not entries:
        raise EvalPostcheckError("actor checkpoint inventory is empty")
    return _canonical_sha256({"schema": CHECKPOINT_SCHEMA, "files": entries}), entries


def _validate_training_provenance(
    path: Path,
    *,
    checkpoint: Path,
    plan_sha256: str,
    expected_policy_hash: str,
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise EvalPostcheckError("training launch manifest must be absolute and non-symlink")
    path = path.resolve()
    launch = baseline._load_json(path, "training launch manifest", private=True)
    root = path.parent
    plan = launch.get("plan")
    records = launch.get("containers")
    if launch.get("schema") != "miles.tbench21-sao-online-wave.v1" or not isinstance(plan, dict) or plan.get("sha256") != plan_sha256 or not isinstance(records, list) or len(records) != ISLANDS or MAC.fullmatch(expected_policy_hash) is None or checkpoint != root / "island-7/actor-checkpoint":
        raise EvalPostcheckError("actor checkpoint is not bound to the completed online wave")
    event_hashes: list[str] = []
    for island_id, record in enumerate(records):
        output = root / f"island-{island_id}"
        if not isinstance(record, dict) or record.get("island_id") != island_id or record.get("name") != f"tbench21-sao-online-island-{island_id}" or record.get("output") != str(output):
            raise EvalPostcheckError("training launch island identity changed")
        events_path = output / "events.jsonl"
        try:
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EvalPostcheckError("terminal actor publication evidence is unreadable") from error
        terminal = [event for event in events if isinstance(event, dict) and event.get("event") == "rl_sao_streaming_publication" and event.get("terminal") is True]
        if len(terminal) != 1 or events[-1] != terminal[0] or terminal[0].get("rollout_id") != 1 or terminal[0].get("actor_policy_hash") != expected_policy_hash or terminal[0].get("actor_fragment_versions") != [1, 2] or terminal[0].get("critic_fragment_versions") != [1, 2]:
            raise EvalPostcheckError("training island has no unique expected terminal actor publication")
        event_hashes.append(terminal[0]["actor_policy_hash"])
    return {
        "launch_manifest": str(path),
        "launch_manifest_sha256": baseline._sha256(path),
        "terminal_actor_policy_hash": expected_policy_hash,
        "islands_attesting_terminal_policy": len(event_hashes),
    }


def _validate_checkpoint(path: Path, *, expected_inventory_sha256: str) -> tuple[Path, str, int]:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise EvalPostcheckError("expected actor checkpoint must be an absolute real directory")
    path = path.resolve()
    marker = path / "latest_checkpointed_iteration.txt"
    if marker.is_symlink() or not marker.is_file():
        raise EvalPostcheckError("expected actor checkpoint has no safe iteration marker")
    try:
        iteration = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise EvalPostcheckError("actor checkpoint iteration marker is unreadable") from error
    if iteration != "0":
        raise EvalPostcheckError(f"held-out actor checkpoint marker is {iteration!r}, expected rollout 0 save")
    checkpoint_dir = path / "iter_0000000"
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise EvalPostcheckError("held-out actor checkpoint iteration is incomplete")
    if MAC.fullmatch(expected_inventory_sha256) is None:
        raise EvalPostcheckError("expected actor checkpoint inventory SHA-256 is malformed")
    inventory_sha256, entries = _checkpoint_inventory(path)
    if inventory_sha256 != expected_inventory_sha256:
        raise EvalPostcheckError("actor checkpoint bytes differ from the preverified final actor")
    return path, inventory_sha256, len(entries)


def _validate_containers(
    records: list[dict[str, Any]],
    inspections: list[dict[str, Any]],
    *,
    checkpoint: Path,
    plan_dir: Path,
    output_root: Path,
    hmac_key_path: Path,
    model: Path,
    codex_binary_sha256: str,
    bridge_gateway: str,
) -> list[dict[str, Any]]:
    by_name = {str(item.get("Name", "")).removeprefix("/"): item for item in inspections}
    if len(by_name) != ISLANDS or len(inspections) != ISLANDS:
        raise EvalPostcheckError("Docker did not return exactly eight eval containers")
    completed: list[dict[str, Any]] = []
    for island_id, record in enumerate(records):
        name = f"tbench21-eval-island-{island_id}"
        expected_id = record.get("container_id") if isinstance(record, dict) else None
        item = by_name.get(name)
        state = item.get("State") if isinstance(item, dict) else None
        config = item.get("Config") if isinstance(item, dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        environment = config.get("Env") if isinstance(config, dict) else None
        env = dict(value.split("=", 1) for value in environment if isinstance(value, str) and "=" in value) if isinstance(environment, list) else {}
        command = config.get("Cmd") if isinstance(config, dict) else None
        host_config = item.get("HostConfig") if isinstance(item, dict) else None
        mounts = item.get("Mounts") if isinstance(item, dict) else None
        mount_by_destination = {mount.get("Destination"): mount for mount in mounts if isinstance(mount, dict) and isinstance(mount.get("Destination"), str)} if isinstance(mounts, list) else {}
        checkpoint_mount = mount_by_destination.get("/root/input-checkpoint")
        plan_mount = mount_by_destination.get("/root/plan")
        hmac_mount = mount_by_destination.get("/run/secrets/tbench-hmac")
        output_mount = mount_by_destination.get("/root/run-output")
        model_mount = mount_by_destination.get("/root/model")
        codex_mount = mount_by_destination.get("/root/codex-cli")
        expected_command = [
            RUNNER,
            "--island-id",
            str(island_id),
            "--phase",
            "eval",
            "--prompt-data",
            f"/root/plan/eval/island-{island_id}.jsonl",
            "--dump-details",
            "/root/run-output/details",
            "--hf-checkpoint",
            "/root/model",
            "--ref-load",
            "/root/input-checkpoint",
            "--codex-binary",
            CODEX_BINARY,
            "--openenv-agent-python",
            "/root/openenv-venv/bin/python",
            "--openenv-env-url",
            MANAGED_CONTAINER_URL,
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
        codex_source = Path(codex_mount["Source"]) if isinstance(codex_mount, dict) and isinstance(codex_mount.get("Source"), str) else None
        codex_binary = codex_source / "vendor/x86_64-unknown-linux-musl/bin/codex" if codex_source is not None else None
        if (
            not isinstance(record, dict)
            or record.get("island_id") != island_id
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
            or labels.get("yeto.phase") != "eval"
            or labels.get("yeto.island") != str(island_id)
            or config.get("Entrypoint") != ["python3"]
            or command != expected_command
            or env.get("NVIDIA_VISIBLE_DEVICES") != str(island_id)
            or env.get("TBENCH_REWARD_HMAC_KEY_FILE") != "/run/secrets/tbench-hmac"
            or "TBENCH_REWARD_HMAC_KEY" in env
            or env.get("HF_HUB_OFFLINE") != "1"
            or env.get("TRANSFORMERS_OFFLINE") != "1"
            or not isinstance(host_config, dict)
            or host_config.get("ExtraHosts") != [f"host.docker.internal:{bridge_gateway}"]
            or not isinstance(checkpoint_mount, dict)
            or checkpoint_mount.get("Source") != str(checkpoint)
            or checkpoint_mount.get("RW") is not False
            or not isinstance(plan_mount, dict)
            or plan_mount.get("Source") != str(plan_dir)
            or plan_mount.get("RW") is not False
            or not isinstance(hmac_mount, dict)
            or hmac_mount.get("Source") != str(hmac_key_path)
            or hmac_mount.get("RW") is not False
            or not isinstance(output_mount, dict)
            or output_mount.get("Source") != str(output_root / f"island-{island_id}")
            or output_mount.get("RW") is not True
            or not isinstance(model_mount, dict)
            or model_mount.get("Source") != str(model)
            or model_mount.get("RW") is not False
            or not isinstance(codex_mount, dict)
            or codex_mount.get("RW") is not False
            or codex_binary is None
            or not codex_binary.is_file()
            or codex_binary.is_symlink()
            or baseline._sha256(codex_binary) != codex_binary_sha256
        ):
            raise EvalPostcheckError(f"eval container {island_id} did not exit cleanly with its read-only launch identity")
        completed.append(
            {
                "island_id": island_id,
                "name": name,
                "container_id": expected_id,
                "exit_code": 0,
                "oom_killed": False,
                "checkpoint_read_only": True,
            }
        )
    return completed


def _validate_payload(
    payload: dict[str, Any],
    *,
    island_id: int,
    expected: dict[str, PlannedSample],
    source: Path,
    hmac_key: bytes,
) -> tuple[dict[str, Any], list[LogicalResult]]:
    samples = payload.get("samples")
    if payload.get("rollout_id") != 0 or not isinstance(payload.get("metadata"), dict) or not isinstance(samples, list) or not samples:
        raise EvalPostcheckError(f"island {island_id} does not contain one native rollout_id=0 batch")
    by_sample: dict[str, list[tuple[dict[str, Any], dict[str, Any], float]]] = defaultdict(list)
    retry_ledgers: dict[str, int] = {}
    trajectory_ids: set[str] = set()
    for ordinal, raw in enumerate(samples):
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
        contract = expected.get(sample_id) if isinstance(sample_id, str) else None
        index = raw.get("index") if isinstance(raw, dict) else None
        group_index = raw.get("group_index") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or not isinstance(metadata, dict)
            or contract is None
            or type(index) is not int
            or index != contract.index
            or type(group_index) is not int
            or group_index != contract.index
            or raw.get("status") not in {"completed", "truncated"}
            or raw.get("remove_sample", False) is not False
            or metadata.get("task_id") != contract.task_id
            or metadata.get("rollout_replica") != contract.replica
            or metadata.get("split") != "eval"
            or metadata.get("island_id") != island_id
            or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
            or metadata.get("episode_timeout_seconds") != EPISODE_TIMEOUT_SECONDS
            or metadata.get("exit_status") not in TERMINAL_STATUSES
        ):
            raise EvalPostcheckError(f"{source}:{ordinal}: accepted sample is invalid, cycled, replaced, or off-plan")
        marker = baseline._replacement_marker(metadata)
        if marker is not None:
            raise EvalPostcheckError(f"{source}:{ordinal}: accepted sample carries replacement marker {marker}")
        retry_count = metadata.get("secrlenv_infrastructure_503_retries", 0)
        if type(retry_count) is not int or retry_count not in {0, 1}:
            raise EvalPostcheckError(f"{source}:{ordinal}: invalid infrastructure retry ledger")
        prior_retry = retry_ledgers.setdefault(sample_id, retry_count)
        if prior_retry != retry_count:
            raise EvalPostcheckError(f"{source}:{ordinal}: infrastructure retry ledger changed across segments")
        outcome, reward = _verified_outcome(
            metadata,
            key=hmac_key,
            sample_id=sample_id,
            task_id=contract.task_id,
        )
        if outcome["status"] != metadata["exit_status"] or _finite_binary(raw.get("reward"), name="sample reward") != reward or _finite_binary(metadata.get("reward"), name="metadata reward") != reward:
            raise EvalPostcheckError(f"{source}:{ordinal}: signed reward/status differs from accepted sample")
        by_sample[sample_id].append((raw, outcome, reward))

    if set(by_sample) != set(expected):
        missing = sorted(set(expected) - set(by_sample))
        extra = sorted(set(by_sample) - set(expected))
        raise EvalPostcheckError(f"island {island_id} eval coverage changed: missing={missing[:3]!r}, extra={extra[:3]!r}")
    results: list[LogicalResult] = []
    segment_count = 0
    for sample_id, rows in by_sample.items():
        rows.sort(key=lambda item: item[0]["metadata"].get("compaction_segment_index", -1))
        raw_rows = [item[0] for item in rows]
        outcomes = [item[1] for item in rows]
        rewards = [item[2] for item in rows]
        metadata_rows = [row["metadata"] for row in raw_rows]
        indices = [metadata.get("compaction_segment_index") for metadata in metadata_rows]
        types = [metadata.get("compaction_segment_type") for metadata in metadata_rows]
        identities = {metadata.get("compaction_trajectory_id") for metadata in metadata_rows}
        if (
            not 1 <= len(rows) <= 7
            or indices != list(range(len(rows)))
            or types != ["execution" if index % 2 == 0 else "summary" for index in indices]
            or types[-1] != "execution"
            or len(identities) != 1
            or not isinstance(next(iter(identities)), str)
            or not next(iter(identities))
            or any(metadata.get("compaction_schema_version") != 1 for metadata in metadata_rows)
            or any(metadata.get("compaction_context_budget") != MAX_SEQ_LEN for metadata in metadata_rows)
            or any(metadata.get("compaction_context_window") != index // 2 for index, metadata in enumerate(metadata_rows))
        ):
            raise EvalPostcheckError(f"{source}: {sample_id} has duplicated or malformed compaction segments")
        trajectory_id = next(iter(identities))
        if trajectory_id in trajectory_ids:
            raise EvalPostcheckError(f"{source}: two samples reused one trajectory ID")
        trajectory_ids.add(trajectory_id)
        if any(outcome != outcomes[0] for outcome in outcomes[1:]) or any(reward != rewards[0] for reward in rewards[1:]):
            raise EvalPostcheckError(f"{source}: {sample_id} outcome changed across compaction segments")
        terminal_metadata = metadata_rows[-1]
        orphan_dropped = terminal_metadata.get("compaction_terminal_orphan_summary_dropped")
        orphan_index = terminal_metadata.get("compaction_terminal_orphan_summary_segment_index")
        if (orphan_dropped, orphan_index) not in {
            (None, None),
            (True, len(metadata_rows)),
        }:
            raise EvalPostcheckError(f"{source}: {sample_id} has malformed terminal orphan-summary evidence")
        declared_segment_count = len(metadata_rows) + int(orphan_dropped is True)
        for segment_ordinal, metadata in enumerate(metadata_rows):
            terminal = segment_ordinal == len(metadata_rows) - 1
            if terminal:
                if metadata.get("compaction_segment_count") != declared_segment_count or metadata.get("compaction_context_window_count") != (len(metadata_rows) + 1) // 2:
                    raise EvalPostcheckError(f"{source}: {sample_id} terminal compaction totals differ")
            elif "compaction_segment_count" in metadata or "compaction_context_window_count" in metadata or "compaction_terminal_orphan_summary_dropped" in metadata or "compaction_terminal_orphan_summary_segment_index" in metadata:
                raise EvalPostcheckError(f"{source}: {sample_id} nonterminal segment carries terminal compaction fields")
        terminal_mismatch = False
        for segment_ordinal, (metadata, outcome, reward) in enumerate(zip(metadata_rows, outcomes, rewards, strict=True)):
            mismatch = metadata.get("tito_session_mismatch")
            if mismatch in (None, False, []):
                continue
            if segment_ordinal != len(metadata_rows) - 1 or not isinstance(mismatch, list) or len(mismatch) != 1:
                raise EvalPostcheckError(f"{source}: {sample_id} has a nonterminal or multi-entry TITO mismatch")
            entry = mismatch[0]
            detail_match = TITO_COUNT_DETAIL.fullmatch(entry.get("detail", "")) if isinstance(entry, dict) else None
            expected_text = entry.get("expected_text") if isinstance(entry, dict) else None
            actual_text = entry.get("actual_text") if isinstance(entry, dict) else None
            if (
                not isinstance(entry, dict)
                or set(entry) != {"type", "segment_index", "expected_text", "actual_text", "detail"}
                or entry.get("type") != "special_token_count"
                or entry.get("segment_index") != -1
                or detail_match is None
                or int(detail_match.group(2)) != int(detail_match.group(1)) - 1
                or not isinstance(expected_text, str)
                or not expected_text.endswith("[<|im_end|>]")
                or not isinstance(actual_text, str)
                or actual_text.endswith("[<|im_end|>]")
                or metadata.get("exit_status") not in {"max_seq_len", "max_turns"}
                or outcome.get("status") != metadata.get("exit_status")
                or reward != 0.0
            ):
                raise EvalPostcheckError(f"{source}: {sample_id} has an unapproved TITO mismatch shape")
            terminal_mismatch = True
        contract = expected[sample_id]
        results.append(
            LogicalResult(
                sample_id=sample_id,
                task_id=contract.task_id,
                replica=contract.replica,
                reward=rewards[0],
                status=outcomes[0]["status"],
                verifier=outcomes[0]["verifier"],
                episode_id=outcomes[0]["episode_id"],
                trajectory_id=trajectory_id,
                segments=len(rows),
                terminal_tito_boundary_mismatch=terminal_mismatch,
            )
        )
        segment_count += len(rows)
    if len({result.episode_id for result in results}) != len(results):
        raise EvalPostcheckError(f"{source}: two eval samples reused one episode ID")
    return (
        {
            "island_id": island_id,
            "rollout_path": str(source),
            "rollout_sha256": baseline._sha256(source),
            "logical_trajectories": len(results),
            "segments": segment_count,
            "reward_sum": sum(result.reward for result in results),
            "passed": sum(result.reward == 1.0 for result in results),
            "trajectories_with_infrastructure_503_retry": sum(retry_ledgers.values()),
            "infrastructure_aborted_replacements": 0,
            "signed_outcomes_verified": len(results),
            "terminal_tito_boundary_mismatch_count": sum(result.terminal_tito_boundary_mismatch for result in results),
            "terminal_tito_boundary_mismatch_sample_ids": sorted(result.sample_id for result in results if result.terminal_tito_boundary_mismatch),
        },
        results,
    )


def _aggregate(results: list[LogicalResult], expected_tasks: set[str]) -> dict[str, Any]:
    if len(results) != EXPECTED_TRAJECTORIES:
        raise EvalPostcheckError("evaluation did not produce 180 logical results")
    if len({result.sample_id for result in results}) != EXPECTED_TRAJECTORIES or len({result.episode_id for result in results}) != EXPECTED_TRAJECTORIES or len({result.trajectory_id for result in results}) != EXPECTED_TRAJECTORIES:
        raise EvalPostcheckError("evaluation reused a sample, trajectory, or managed episode identity")
    by_task: dict[str, list[LogicalResult]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    if set(by_task) != expected_tasks:
        raise EvalPostcheckError("evaluation result task roster differs from its plan")
    task_results: dict[str, Any] = {}
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda result: result.replica)
        if [result.replica for result in rows] != list(range(ROLLOUTS_PER_TASK)):
            raise EvalPostcheckError(f"task {task_id} does not have replicas 0..3")
        rewards = [result.reward for result in rows]
        task_results[task_id] = {
            "rewards_by_replica": rewards,
            "reward_sum": sum(rewards),
            "mean_reward": sum(rewards) / ROLLOUTS_PER_TASK,
            "passed_rollouts": sum(reward == 1.0 for reward in rewards),
            "any_success": any(reward == 1.0 for reward in rewards),
        }
    reward_sum = sum(result.reward for result in results)
    tasks_with_success = sum(row["any_success"] for row in task_results.values())
    mismatch_ids = sorted(result.sample_id for result in results if result.terminal_tito_boundary_mismatch)
    return {
        "rollout_reward_sum": reward_sum,
        "rollout_mean_reward": reward_sum / EXPECTED_TRAJECTORIES,
        "pass_at_1": reward_sum / EXPECTED_TRAJECTORIES,
        "passed_rollouts": int(reward_sum),
        "tasks_with_any_success": tasks_with_success,
        "task_success_rate_at_4": tasks_with_success / EXPECTED_TASKS,
        "pass_at_4": tasks_with_success / EXPECTED_TASKS,
        "status_counts": dict(sorted(Counter(result.status for result in results).items())),
        "verifier_counts": dict(sorted(Counter(result.verifier for result in results).items())),
        "terminal_tito_boundary_mismatch_count": len(mismatch_ids),
        "terminal_tito_boundary_mismatch_sample_ids": mismatch_ids,
        "tasks": task_results,
    }


def _managed_process(pid: int, *, run_id: str) -> dict[str, Any]:
    if type(pid) is not int or pid <= 1:
        raise EvalPostcheckError("managed Terminal-Bench server PID is invalid")
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        argv = [value.decode("utf-8") for value in argv if value]
    except (OSError, UnicodeError) as error:
        raise EvalPostcheckError("managed Terminal-Bench server process is not readable") from error
    expected = [
        "/root/openenv-venv/bin/python",
        MANAGED_SERVER_SCRIPT,
        "--host",
        "0.0.0.0",
        "--port",
        "8003",
        "--run-id",
        run_id,
        "--max-concurrent-envs",
        str(MANAGED_MAX_CONCURRENCY),
    ]
    if argv != expected:
        raise EvalPostcheckError("managed Terminal-Bench process identity differs from the eval endpoint")
    return {"pid": pid, "argv": argv}


def _wait_for_managed_clean(*, url: str, run_id: str, timeout_s: float) -> dict[str, Any]:
    if url != MANAGED_STATUS_URL:
        raise EvalPostcheckError("managed status URL is not the endpoint used by eval containers")
    try:
        from examples.experimental.openenv.preflight_tbench21_shared_server import (
            wait_for_idle_managed_status,
        )

        status = wait_for_idle_managed_status(
            url=url,
            expected_run_id=run_id,
            timeout_s=timeout_s,
        )
    except Exception as error:
        raise EvalPostcheckError("managed Terminal-Bench server did not drain cleanly") from error
    if (
        status.get("ok") is not True
        or status.get("run_id") != run_id
        or status.get("active_sessions") != 0
        or status.get("managed_containers") != 0
        or status.get("orphan_containers") != 0
        or status.get("cleanup_blocked") is not False
        or status.get("failed_cleanup_count") != 0
        or status.get("last_janitor_error") not in {None, ""}
    ):
        raise EvalPostcheckError("managed Terminal-Bench server retains session, container, orphan, or janitor debt")
    return status


def postcheck(
    launch_manifest_path: Path,
    plan_dir: Path,
    checkpoint: Path,
    hmac_key_path: Path,
    *,
    expected_plan_manifest_sha256: str,
    training_launch_manifest: Path,
    expected_terminal_actor_policy_hash: str,
    expected_checkpoint_inventory_sha256: str,
    managed_status_url: str,
    managed_run_id: str,
    managed_server_pid: int,
    managed_clean_timeout_s: float,
) -> dict[str, Any]:
    launch_manifest_path = launch_manifest_path.expanduser()
    if not launch_manifest_path.is_absolute() or launch_manifest_path.is_symlink():
        raise EvalPostcheckError("launch manifest path must be absolute and non-symlink")
    launch_manifest_path = launch_manifest_path.resolve()
    launch = baseline._load_json(launch_manifest_path, "eval launch manifest", private=True)
    records = launch.get("containers")
    runtime_image = launch.get("runtime_image")
    plan_source = plan_dir.expanduser()
    if not plan_source.is_absolute() or plan_source.is_symlink() or not plan_source.is_dir():
        raise EvalPostcheckError("plan directory must be an absolute real directory")
    plan_dir = plan_source.resolve()
    checkpoint, checkpoint_inventory_sha256, checkpoint_inventory_files = _validate_checkpoint(
        checkpoint.expanduser(),
        expected_inventory_sha256=expected_checkpoint_inventory_sha256,
    )
    hmac_key_path, hmac_key = _load_hmac_key(hmac_key_path.expanduser())
    model_raw = launch.get("model")
    bridge_gateway = launch.get("docker_bridge_gateway")
    codex_binary_sha256 = launch.get("codex_binary_sha256")
    model = Path(model_raw).resolve() if isinstance(model_raw, str) and Path(model_raw).is_absolute() else None
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("phase") != "eval"
        or not isinstance(runtime_image, dict)
        or runtime_image.get("id") != RUNTIME_IMAGE_ID
        or launch.get("checkpoint") != str(checkpoint)
        or launch.get("plan_dir") != str(plan_dir)
        or model is None
        or not model.is_dir()
        or model.is_symlink()
        or not isinstance(bridge_gateway, str)
        or MAC.fullmatch(codex_binary_sha256 or "") is None
        or launch.get("hmac_key_file_mode") != oct(stat.S_IMODE(hmac_key_path.stat().st_mode))
        or launch.get("per_island_rollout_seeds") != [ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)]
        or not isinstance(records, list)
        or len(records) != ISLANDS
    ):
        raise EvalPostcheckError("launch manifest differs from the held-out eval wave")
    expected_by_island, plan_sha256, expected_tasks = _plan_rows(
        plan_dir,
        expected_manifest_sha256=expected_plan_manifest_sha256,
    )
    training_provenance = _validate_training_provenance(
        training_launch_manifest.expanduser(),
        checkpoint=checkpoint,
        plan_sha256=plan_sha256,
        expected_policy_hash=expected_terminal_actor_policy_hash,
    )
    names = [f"tbench21-eval-island-{island_id}" for island_id in range(ISLANDS)]
    completed = _validate_containers(
        records,
        baseline._docker_inspect(names),
        checkpoint=checkpoint,
        plan_dir=plan_dir,
        output_root=launch_manifest_path.parent,
        hmac_key_path=hmac_key_path,
        model=model,
        codex_binary_sha256=codex_binary_sha256,
        bridge_gateway=bridge_gateway,
    )

    output_root = launch_manifest_path.parent
    island_reports: list[dict[str, Any]] = []
    logical_results: list[LogicalResult] = []
    for island_id, expected in enumerate(expected_by_island):
        record = records[island_id]
        output = output_root / f"island-{island_id}"
        if record.get("output") != str(output) or not output.is_dir() or output.is_symlink():
            raise EvalPostcheckError(f"island {island_id} output path changed after launch")
        if any((output / name).exists() for name in ("actor-checkpoint", "critic-checkpoint", "completed-groups.pt", "trajectory-evidence")) or any((output / "details" / name).exists() for name in ("train_data", "policy_loss_debug")):
            raise EvalPostcheckError(f"island {island_id} eval output contains training or optimizer artifacts")
        shard = plan_dir / f"eval/island-{island_id}.jsonl"
        if record.get("plan_shard") != str(shard) or record.get("plan_shard_sha256") != baseline._sha256(shard):
            raise EvalPostcheckError(f"island {island_id} launch is not bound to its eval shard")
        rollout_dir = output / "details/rollout_data"
        source = rollout_dir / "0.pt"
        if not rollout_dir.is_dir() or rollout_dir.is_symlink() or sorted(rollout_dir.iterdir(), key=lambda path: path.name) != [source] or not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise EvalPostcheckError(f"island {island_id} must contain exactly details/rollout_data/0.pt")
        island_report, results = _validate_payload(
            baseline._load_rollout_payload(source),
            island_id=island_id,
            expected=expected,
            source=source,
            hmac_key=hmac_key,
        )
        island_reports.append(island_report)
        logical_results.extend(results)

    managed_process = _managed_process(managed_server_pid, run_id=managed_run_id)
    managed_status = _wait_for_managed_clean(
        url=managed_status_url,
        run_id=managed_run_id,
        timeout_s=managed_clean_timeout_s,
    )
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "phase": "eval",
        "launch_manifest": {
            "path": str(launch_manifest_path),
            "sha256": baseline._sha256(launch_manifest_path),
        },
        "plan_manifest": {
            "path": str(plan_dir / "manifest.json"),
            "sha256": plan_sha256,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "iteration": 0,
            "inventory_sha256": checkpoint_inventory_sha256,
            "inventory_files": checkpoint_inventory_files,
            "mounted_read_only_by_all_islands": True,
        },
        "training_provenance": training_provenance,
        "hmac_key_file_mode": oct(stat.S_IMODE(hmac_key_path.stat().st_mode)),
        "containers": completed,
        "islands": island_reports,
        "logical_trajectories": len(logical_results),
        "task_count": len(expected_tasks),
        "rollouts_per_task": ROLLOUTS_PER_TASK,
        "signed_outcomes_verified": sum(report["signed_outcomes_verified"] for report in island_reports),
        "trajectories_with_infrastructure_503_retry": sum(report["trajectories_with_infrastructure_503_retry"] for report in island_reports),
        "infrastructure_aborted_replacements": 0,
        "terminal_tito_boundary_mismatch_count": sum(report["terminal_tito_boundary_mismatch_count"] for report in island_reports),
        "terminal_tito_boundary_mismatch_sample_ids": sorted(sample_id for report in island_reports for sample_id in report["terminal_tito_boundary_mismatch_sample_ids"]),
        "aggregate": _aggregate(logical_results, expected_tasks),
        "managed_server": {"process": managed_process, "status": managed_status},
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise EvalPostcheckError("report path must be absolute and fresh")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--expected-plan-manifest-sha256", required=True)
    parser.add_argument("--training-launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-terminal-actor-policy-hash", required=True)
    parser.add_argument("--expected-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--managed-status-url", default=MANAGED_STATUS_URL)
    parser.add_argument("--managed-run-id", required=True)
    parser.add_argument("--managed-server-pid", type=int, required=True)
    parser.add_argument("--managed-clean-timeout-s", type=float, default=60.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = postcheck(
            args.launch_manifest,
            args.plan_dir,
            args.checkpoint,
            args.hmac_key,
            expected_plan_manifest_sha256=args.expected_plan_manifest_sha256,
            training_launch_manifest=args.training_launch_manifest,
            expected_terminal_actor_policy_hash=args.expected_terminal_actor_policy_hash,
            expected_checkpoint_inventory_sha256=args.expected_checkpoint_inventory_sha256,
            managed_status_url=args.managed_status_url,
            managed_run_id=args.managed_run_id,
            managed_server_pid=args.managed_server_pid,
            managed_clean_timeout_s=args.managed_clean_timeout_s,
        )
        if args.report is not None:
            _write_report(args.report.expanduser(), report)
    except Exception as error:
        print(
            f"TB2.1 eval postcheck failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
