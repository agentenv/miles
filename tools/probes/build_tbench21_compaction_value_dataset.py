"""Build the all-trajectory SAO critic dataset from a TB2.1 baseline.

The full Terminal-Bench baseline produces one or more CompactionRL segments
for each planned trajectory.  This converter requires every planned baseline
trajectory exactly once, keeps every trainable segment, and assigns the final
binary task reward to every active token in that trajectory.  It never mixes
the actor train/eval split: the manager-requested 100% baseline selection is
made explicit in the input selection contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SCHEMA = "miles.value-pretrain.v1"
REPORT_SCHEMA = "miles.tbench21-compaction-value-conversion.v1"
MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_SEQ_LEN = 8192
VOCAB_SIZE = 248320
ISLANDS = 8
ROLLOUTS_PER_TASK = 4
ROLLOUT_SEED_BASE = 82621
HMAC_ENV = "TBENCH_REWARD_HMAC_KEY"
HMAC_FILE_ENV = "TBENCH_REWARD_HMAC_KEY_FILE"
MIN_HMAC_KEY_BYTES = 32
MAX_HMAC_KEY_BYTES = 4096


class ConversionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _outcome_api():
    try:
        from yeto.rl import tbench_outcome
    except ImportError as error:
        raise ConversionError("Yeto's authenticated Terminal-Bench outcome verifier is unavailable") from error
    return tbench_outcome


def _validate_hmac_key_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ConversionError("--hmac-key-file must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise ConversionError("Terminal-Bench HMAC key file is unreadable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ConversionError("Terminal-Bench HMAC key file must be a regular non-symlink")
    if stat.S_IMODE(info.st_mode) not in {0o400, 0o600}:
        raise ConversionError("Terminal-Bench HMAC key file must have mode 0400 or 0600")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev or opened.st_ino != info.st_ino:
                raise ConversionError("Terminal-Bench HMAC key file changed during validation")
            value = os.read(descriptor, MAX_HMAC_KEY_BYTES + 2).rstrip(b"\r\n")
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ConversionError("Terminal-Bench HMAC key file is unreadable") from error
    if not MIN_HMAC_KEY_BYTES <= len(value) <= MAX_HMAC_KEY_BYTES:
        raise ConversionError("Terminal-Bench HMAC key must contain 32..4096 bytes")
    return path


@contextmanager
def _authenticated_outcome_source(path: Path) -> Iterator[Any]:
    if HMAC_ENV in os.environ:
        raise ConversionError(f"{HMAC_ENV} must be absent; use only the explicit --hmac-key-file")
    path = _validate_hmac_key_file(path)
    prior_file = os.environ.get(HMAC_FILE_ENV)
    if prior_file is not None and prior_file != str(path):
        raise ConversionError(f"ambient {HMAC_FILE_ENV} differs from --hmac-key-file")
    api = _outcome_api()
    os.environ[HMAC_FILE_ENV] = str(path)
    try:
        api.validate_hmac_key_source()
        yield api
    except api.UntrustedTBenchOutcome as error:
        raise ConversionError("Terminal-Bench HMAC key source failed authenticated-outcome validation") from error
    finally:
        if prior_file is None:
            os.environ.pop(HMAC_FILE_ENV, None)
        else:
            os.environ[HMAC_FILE_ENV] = prior_file


def _load_selection(path: Path) -> tuple[str, ...]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ConversionError("value-pretraining selection must be an absolute regular non-symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError("value-pretraining selection is unreadable") from error
    expected_fields = {
        "schema",
        "source_phase",
        "selection",
        "sample_count",
        "task_count",
        "rollouts_per_task",
        "baseline_sample_ids",
        "critic_pretraining_exposes_actor_eval_tasks",
        "actor_eval_task_count_exposed_to_critic",
        "actor_eval_trajectory_count_exposed_to_critic",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ConversionError("value-pretraining selection schema changed")
    sample_ids = raw["baseline_sample_ids"]
    if (
        raw["schema"] != "yeto.sao-value-pretraining-selection.v1"
        or raw["source_phase"] != "baseline"
        or raw["selection"] != "all"
        or raw["sample_count"] != 356
        or raw["task_count"] != 89
        or raw["rollouts_per_task"] != 4
        or raw["critic_pretraining_exposes_actor_eval_tasks"] is not True
        or raw["actor_eval_task_count_exposed_to_critic"] != 45
        or raw["actor_eval_trajectory_count_exposed_to_critic"] != 180
        or not isinstance(sample_ids, list)
        or len(sample_ids) != 356
        or len(set(sample_ids)) != 356
        or any(not isinstance(value, str) or not value for value in sample_ids)
    ):
        raise ConversionError("value-pretraining selection is not the exact full baseline")
    return tuple(sample_ids)


def _load_plan_manifest(path: Path, *, selection_path: Path, selection_ids: tuple[str, ...]) -> dict[str, int]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ConversionError("plan manifest must be an absolute regular non-symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError("plan manifest is unreadable") from error
    if not isinstance(raw, dict):
        raise ConversionError("plan manifest must be an object")
    terminal_bench = raw.get("terminal_bench")
    rollouts = raw.get("rollouts")
    value = raw.get("value_pretraining")
    compaction = raw.get("compaction")
    topology = raw.get("topology")
    files = raw.get("files")
    if (
        raw.get("schema") != "yeto.tbench21-sao-diloco-plan.v1"
        or not isinstance(terminal_bench, dict)
        or terminal_bench.get("version") != "2.1"
        or terminal_bench.get("task_count") != 89
        or not isinstance(rollouts, dict)
        or rollouts.get("per_task") != ROLLOUTS_PER_TASK
        or rollouts.get("baseline") != 356
        or rollouts.get("episode_timeout_seconds") != 1800
        or rollouts.get("seed_base") != ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != list(range(ROLLOUT_SEED_BASE, ROLLOUT_SEED_BASE + ISLANDS))
        or not isinstance(value, dict)
        or value.get("uses_all_baseline_trajectories") is not True
        or value.get("sample_count") != 356
        or not isinstance(compaction, dict)
        or compaction.get("enabled") is not True
        or compaction.get("trainer_objective") != "sao"
        or compaction.get("max_seq_len") != MAX_SEQ_LEN
        or compaction.get("trigger_tokens") != 6144
        or compaction.get("summary_max_tokens") != 1024
        or compaction.get("max_compactions_per_episode") != 3
        or not isinstance(topology, dict)
        or topology.get("islands") != ISLANDS
        or topology.get("one_physical_gpu_per_island") is not True
        or topology.get("model") != MODEL
        or topology.get("model_revision") != MODEL_REVISION
        or not isinstance(files, dict)
    ):
        raise ConversionError("plan manifest differs from the full baseline contract")

    task_contracts = terminal_bench.get("task_contracts")
    if not isinstance(task_contracts, dict) or len(task_contracts) != 89 or any(not isinstance(task_id, str) or not task_id for task_id in task_contracts):
        raise ConversionError("plan manifest does not contain exactly 89 task IDs")
    task_ids = sorted(task_contracts)
    expected_ids = tuple(f"baseline:{task_id}:r{replica}" for task_id in task_ids for replica in range(ROLLOUTS_PER_TASK))
    if selection_ids != expected_ids:
        raise ConversionError("value selection is not exactly the plan's 89 task IDs x replicas r0..r3")

    relative = value.get("selection_path")
    expected_sha = value.get("selection_sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or not isinstance(expected_sha, str) or len(expected_sha) != 64 or files.get(relative) != expected_sha:
        raise ConversionError("plan value-selection binding is malformed")
    declared_selection = path.parent / relative
    if selection_path.is_symlink() or not selection_path.is_file() or declared_selection.resolve() != selection_path.resolve() or _sha256(selection_path) != expected_sha:
        raise ConversionError("selection SHA/path differs from the plan manifest")

    expected_islands: dict[str, int] = {}
    for task_index, task_id in enumerate(task_ids):
        for replica in range(ROLLOUTS_PER_TASK):
            sample_id = f"baseline:{task_id}:r{replica}"
            expected_islands[sample_id] = (task_index * ROLLOUTS_PER_TASK + replica) % ISLANDS
    return expected_islands


def _source_files(roots: list[Path]) -> tuple[tuple[int, Path], ...]:
    if len(roots) != ISLANDS:
        raise ConversionError("exactly eight ordered source island roots are required")
    resolved_roots: set[Path] = set()
    files: list[tuple[int, Path]] = []
    for island_id, root in enumerate(roots):
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ConversionError(f"source root is not a real directory: {root}")
        resolved = root.resolve()
        if resolved in resolved_roots:
            raise ConversionError("source island roots must be distinct")
        resolved_roots.add(resolved)
        rollout_dir = resolved / "rollout_data"
        if not rollout_dir.is_dir() or rollout_dir.is_symlink():
            raise ConversionError(f"island {island_id} has no real rollout_data directory")
        entries = sorted(rollout_dir.iterdir(), key=lambda item: item.name)
        shard = rollout_dir / "0.pt"
        if entries != [shard] or not shard.is_file() or shard.is_symlink():
            raise ConversionError(f"island {island_id} must contain exactly rollout_data/0.pt")
        files.append((island_id, shard.resolve()))
    return tuple(files)


def _integer_list(name: str, value: object, *, nonempty: bool = True) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ConversionError(f"{name} must be a non-empty list")
    if any(type(item) is not int or item < 0 for item in value):
        raise ConversionError(f"{name} must contain non-negative integers")
    return value


def _segment(
    raw: object,
    *,
    source: Path,
    ordinal: int,
    max_seq_len: int,
    outcome_api: Any,
    island_id: int,
    expected_islands: dict[str, int],
) -> dict[str, Any]:
    location = f"{source}:{ordinal}"
    if not isinstance(raw, dict):
        raise ConversionError(f"{location}: sample must be an object")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ConversionError(f"{location}: sample metadata is missing")
    sample_id = metadata.get("sample_id")
    task_id = metadata.get("task_id")
    trajectory_id = metadata.get("compaction_trajectory_id")
    context_window = metadata.get("compaction_context_window")
    segment_index = metadata.get("compaction_segment_index")
    segment_type = metadata.get("compaction_segment_type")
    context_budget = metadata.get("compaction_context_budget")
    if not isinstance(sample_id, str) or not sample_id:
        raise ConversionError(f"{location}: deterministic baseline sample_id is missing")
    if not isinstance(task_id, str) or not task_id:
        raise ConversionError(f"{location}: task_id is missing")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ConversionError(f"{location}: compaction trajectory ID is missing")
    if metadata.get("compaction_schema_version") != 1:
        raise ConversionError(f"{location}: compaction schema is not v1")
    if type(context_window) is not int or context_window < 0 or type(segment_index) is not int or segment_index < 0 or segment_index > 6 or context_window != segment_index // 2:
        raise ConversionError(f"{location}: compaction context/segment position is invalid")
    if segment_type not in {"execution", "summary"}:
        raise ConversionError(f"{location}: compaction segment type is invalid")
    if context_budget != MAX_SEQ_LEN:
        raise ConversionError(f"{location}: compaction context budget must equal {MAX_SEQ_LEN}")
    replica_prefix = f"baseline:{task_id}:r"
    replica = sample_id.removeprefix(replica_prefix)
    if not sample_id.startswith(replica_prefix) or not replica.isdigit() or int(replica) not in range(4):
        raise ConversionError(f"{location}: sample_id does not bind task_id/replica")
    if (
        expected_islands.get(sample_id) != island_id
        or metadata.get("island_id") != island_id
        or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
        or metadata.get("split") != "baseline"
        or metadata.get("rollout_replica") != int(replica)
        or metadata.get("episode_timeout_seconds") != 1800
        or metadata.get("max_seq_len") != MAX_SEQ_LEN
    ):
        raise ConversionError(f"{location}: sample island or immutable rollout seed differs from plan")
    sample_status = raw.get("status")
    if sample_status not in {"completed", "truncated"}:
        raise ConversionError(f"{location}: segment did not finish with trainable status")

    try:
        outcome, signed_reward = outcome_api.verified_outcome(metadata)
    except outcome_api.UntrustedTBenchOutcome as error:
        raise ConversionError(f"{location}: segment has no valid authenticated Terminal-Bench outcome") from error
    reported_reward = metadata.get("reward")
    exit_status = metadata.get("exit_status")
    if outcome["task_id"] != task_id:
        raise ConversionError(f"{location}: signed task_id differs from row metadata")
    if outcome["sample_id"] != sample_id:
        raise ConversionError(f"{location}: signed sample_id differs from row metadata")
    if isinstance(reported_reward, bool) or not isinstance(reported_reward, (int, float)) or not math.isfinite(float(reported_reward)) or float(reported_reward) != signed_reward:
        raise ConversionError(f"{location}: signed reward differs from row metadata")
    if exit_status != outcome["status"]:
        raise ConversionError(f"{location}: signed status differs from row metadata")

    tokens = _integer_list(f"{location}.tokens", raw.get("tokens"))
    response_length = raw.get("response_length")
    if type(response_length) is not int or response_length < 1 or response_length >= len(tokens) or len(tokens) > max_seq_len:
        raise ConversionError(f"{location}: segment token lengths are invalid")
    loss_mask = _integer_list(f"{location}.loss_mask", raw.get("loss_mask"))
    if len(loss_mask) != response_length or any(value not in {0, 1} for value in loss_mask):
        raise ConversionError(f"{location}: loss mask is not response-aligned and binary")
    if not any(loss_mask):
        raise ConversionError(f"{location}: segment has no trainable tokens")
    reward = raw.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)) or float(reward) != signed_reward:
        raise ConversionError(f"{location}: signed reward differs from the top-level sample reward")
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "context_window": context_window,
        "segment_index": segment_index,
        "segment_type": segment_type,
        "context_budget": context_budget,
        "island_id": island_id,
        "tokens": tokens,
        "response_length": response_length,
        "loss_mask": loss_mask,
        "reward": signed_reward,
        "outcome_status": outcome["status"],
        "outcome_episode_id": outcome["episode_id"],
        "outcome_mac": metadata[outcome_api.MAC_KEY],
        "sample_status": sample_status,
        "source": str(source),
        "source_ordinal": ordinal,
    }


def _load_segments(
    files: tuple[tuple[int, Path], ...],
    *,
    max_seq_len: int,
    hmac_key_file: Path,
    expected_islands: dict[str, int],
) -> list[dict[str, Any]]:
    import torch

    segments: list[dict[str, Any]] = []
    with _authenticated_outcome_source(hmac_key_file) as outcome_api:
        for island_id, source in files:
            payload = torch.load(source, map_location="cpu", weights_only=False)
            rows = payload.get("samples") if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or payload.get("rollout_id") != 0 or not isinstance(payload.get("metadata"), dict) or not isinstance(rows, list):
                raise ConversionError(f"{source}: expected the native rollout_id=0 Miles shard")
            island_segments = [
                _segment(
                    row,
                    source=source,
                    ordinal=ordinal,
                    max_seq_len=max_seq_len,
                    outcome_api=outcome_api,
                    island_id=island_id,
                    expected_islands=expected_islands,
                )
                for ordinal, row in enumerate(rows)
            ]
            expected_count = 45 if island_id < 4 else 44
            if len({row["sample_id"] for row in island_segments}) != expected_count:
                raise ConversionError(f"island {island_id} must contain exactly {expected_count} planned trajectories")
            segments.extend(island_segments)
    return segments


def _validate_and_order(segments: list[dict[str, Any]], expected_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        by_sample.setdefault(segment["sample_id"], []).append(segment)
    expected = set(expected_ids)
    if set(by_sample) != expected:
        missing = sorted(expected - set(by_sample))
        extra = sorted(set(by_sample) - expected)
        raise ConversionError(f"baseline trajectory coverage changed: missing={missing[:5]!r}, extra={extra[:5]!r}")

    ordered: list[dict[str, Any]] = []
    seen_trajectory_ids: set[str] = set()
    for sample_id in expected_ids:
        rows = sorted(by_sample[sample_id], key=lambda item: item["segment_index"])
        indices = [row["segment_index"] for row in rows]
        types = [row["segment_type"] for row in rows]
        trajectory_ids = {row["trajectory_id"] for row in rows}
        task_ids = {row["task_id"] for row in rows}
        rewards = {row["reward"] for row in rows}
        outcomes = {
            (
                row["outcome_status"],
                row["outcome_episode_id"],
                row["outcome_mac"],
            )
            for row in rows
        }
        if indices != list(range(len(rows))):
            raise ConversionError(f"{sample_id}: segment indices are not contiguous")
        if len(rows) > 7:
            raise ConversionError(f"{sample_id}: trajectory exceeds three compactions/seven segments")
        expected_types = ["execution" if index % 2 == 0 else "summary" for index in indices]
        if types != expected_types or types[-1] != "execution":
            raise ConversionError(f"{sample_id}: segment sequence is invalid")
        if len(trajectory_ids) != 1 or len(task_ids) != 1 or len(rewards) != 1 or len(outcomes) != 1:
            raise ConversionError(f"{sample_id}: segment identity/authenticated outcome changed")
        trajectory_id = next(iter(trajectory_ids))
        if trajectory_id in seen_trajectory_ids:
            raise ConversionError("two planned trajectories reused one compaction trajectory ID")
        seen_trajectory_ids.add(trajectory_id)
        ordered.extend(rows)
    return ordered


def build(
    *,
    source_roots: list[Path],
    selection_path: Path,
    plan_manifest_path: Path,
    hmac_key_file: Path,
    output_dir: Path,
    model: str,
    revision: str,
    max_seq_len: int,
    vocab_size: int,
) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise ConversionError("output directory must be fresh")
    if model != MODEL or revision != MODEL_REVISION or max_seq_len != MAX_SEQ_LEN or vocab_size != VOCAB_SIZE:
        raise ConversionError("pinned Qwen3.5-0.8B model contract changed")
    expected_ids = _load_selection(selection_path)
    expected_islands = _load_plan_manifest(
        plan_manifest_path,
        selection_path=selection_path,
        selection_ids=expected_ids,
    )
    source_files = _source_files(source_roots)
    segments = _validate_and_order(
        _load_segments(
            source_files,
            max_seq_len=max_seq_len,
            hmac_key_file=hmac_key_file,
            expected_islands=expected_islands,
        ),
        expected_ids,
    )
    if any(max(segment["tokens"]) >= vocab_size for segment in segments):
        raise ConversionError("baseline contains a token outside the pinned model vocabulary")

    output_dir.mkdir(mode=0o700, parents=True)
    train_path = output_dir / "train.jsonl"
    rows = []
    for segment in segments:
        rows.append(
            {
                "sample_id": (f"tb21:{segment['sample_id']}:segment-{segment['segment_index']}"),
                "tokens": segment["tokens"],
                "response_length": segment["response_length"],
                "returns": [segment["reward"]] * segment["response_length"],
                "loss_mask": segment["loss_mask"],
            }
        )
    _exclusive_write(train_path, b"".join(_canonical(row) for row in rows))
    train_sha256 = _sha256(train_path)
    manifest = {
        "schema": SCHEMA,
        "train": {
            "path": train_path.name,
            "sha256": train_sha256,
            "num_samples": len(rows),
        },
        "heldout": None,
        "objective": {
            "loss_type": "classification",
            "num_bins": 51,
            "reward_range": [0.0, 1.0],
            "target_type": "hl_gauss",
            "hl_gauss_sigma_ratio": 0.75,
        },
        "provenance": {
            "schema": "miles.tbench21-compaction-value-provenance.v1",
            "model": model,
            "revision": revision,
            "plan_manifest_sha256": _sha256(plan_manifest_path),
            "selection_sha256": _sha256(selection_path),
            "planned_trajectories": len(expected_ids),
            "included_trajectories": len({row["sample_id"] for row in segments}),
            "included_segments": len(segments),
            "all_baseline_segments_exactly_once": True,
            "all_terminal_bench_outcomes_authenticated": True,
            "authenticated_outcomes_bound_to_rows": True,
            "critic_pretraining_exposes_actor_eval_tasks": True,
            "source_shards": {str(path): _sha256(path) for _, path in source_files},
        },
    }
    manifest_path = output_dir / "manifest.json"
    _exclusive_write(manifest_path, _canonical(manifest))
    report = {
        "schema": REPORT_SCHEMA,
        "model": model,
        "revision": revision,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "plan_manifest_path": str(plan_manifest_path),
        "plan_manifest_sha256": _sha256(plan_manifest_path),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "planned_trajectories": len(expected_ids),
        "included_trajectories": len({row["sample_id"] for row in segments}),
        "included_segments": len(segments),
        "all_terminal_bench_outcomes_authenticated": True,
        "authenticated_outcomes_bound_to_rows": True,
        "segment_types": dict(sorted(Counter(row["segment_type"] for row in segments).items())),
        "outcome_statuses": dict(sorted(Counter(row["outcome_status"] for row in segments).items())),
        "rewards_by_trajectory": dict(sorted(Counter(next(row["reward"] for row in segments if row["sample_id"] == sample_id) for sample_id in expected_ids).items())),
        "active_tokens": sum(sum(row["loss_mask"]) for row in segments),
        "source_shards": {str(path): _sha256(path) for _, path in source_files},
        "train_sha256": train_sha256,
        "manifest_sha256": _sha256(manifest_path),
    }
    _exclusive_write(output_dir / "report.json", _canonical(report))
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument(
        "--hmac-key-file",
        type=Path,
        required=True,
        help=("absolute 0400/0600 non-symlink key used to authenticate every Terminal-Bench outcome"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args(argv)
    manifest = build(
        source_roots=args.source_root,
        selection_path=args.selection,
        plan_manifest_path=args.plan_manifest,
        hmac_key_file=args.hmac_key_file,
        output_dir=args.output_dir.resolve(),
        model=args.model,
        revision=args.revision,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
