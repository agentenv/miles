"""Plan exact-coverage offline value pretraining for the TB2.1 baseline.

Compaction makes the number of critic rows data-dependent: each trajectory
has one execution segment plus two rows for every compaction event.  Miles'
offline value dataset intentionally drops an incomplete final global batch,
so this preflight chooses only a DP/GBS pair that divides the observed
manifest count.  The selected plan therefore consumes every baseline segment
exactly once in the single requested epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from miles.value_pretraining import ValuePretrainDataset, load_value_pretrain_manifest


SCHEMA = "miles.tbench21-value-pretrain-plan.v1"
BASELINE_PROVENANCE_SCHEMA = "miles.tbench21-compaction-value-provenance.v1"
TEACHER_PROVENANCE_SCHEMA = "miles.tbench21-qwen38-teacher-value-provenance.v1"


@dataclass(frozen=True)
class BatchPlan:
    num_samples: int
    dp_size: int
    global_batch_size: int
    optimizer_steps: int
    epochs: int = 1

    @property
    def consumed_rows(self) -> int:
        return self.optimizer_steps * self.global_batch_size


def choose_batch_plan(
    num_samples: int,
    *,
    available_gpus: int,
    max_global_batch_size: int,
) -> BatchPlan:
    """Prefer maximum exact DP, then the largest bounded exact global batch.

    More DP ranks reduce the dominant transformer work.  Within that DP
    choice, the largest divisor up to ``max_global_batch_size`` reduces
    optimizer/all-reduce overhead without turning the whole corpus into one
    giant optimization step.  Both choices are divisors of ``num_samples``;
    no padding, repetition, or incomplete-batch drop is permitted.
    """

    for name, value in {
        "num_samples": num_samples,
        "available_gpus": available_gpus,
        "max_global_batch_size": max_global_batch_size,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    dp_size = next(candidate for candidate in range(min(num_samples, available_gpus, max_global_batch_size), 0, -1) if num_samples % candidate == 0)
    global_batch_size = max(candidate for candidate in range(dp_size, min(num_samples, max_global_batch_size) + 1) if candidate % dp_size == 0 and num_samples % candidate == 0)
    plan = BatchPlan(
        num_samples=num_samples,
        dp_size=dp_size,
        global_batch_size=global_batch_size,
        optimizer_steps=num_samples // global_batch_size,
    )
    if plan.consumed_rows != num_samples:
        raise AssertionError("exact-coverage value batch planning failed")
    return plan


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_baseline_provenance(
    provenance: object,
    *,
    expected_model: str,
    expected_revision: str,
    num_samples: int,
) -> None:
    expected = {
        "schema": BASELINE_PROVENANCE_SCHEMA,
        "model": expected_model,
        "revision": expected_revision,
        "planned_trajectories": 356,
        "included_trajectories": 356,
        "included_segments": num_samples,
        "all_baseline_segments_exactly_once": True,
        "all_terminal_bench_outcomes_authenticated": True,
        "authenticated_outcomes_bound_to_rows": True,
        "critic_pretraining_exposes_actor_eval_tasks": True,
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("value manifest does not prove the exact pinned TB2.1 baseline")
    source_shards = provenance.get("source_shards")
    if (
        not _is_sha256(provenance.get("plan_manifest_sha256"))
        or not _is_sha256(provenance.get("selection_sha256"))
        or not isinstance(source_shards, dict)
        or len(source_shards) != 8
        or any(
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or Path(path).parts[-2:] != ("rollout_data", "0.pt")
            or not _is_sha256(digest)
            for path, digest in source_shards.items()
        )
    ):
        raise ValueError("value manifest has incomplete source provenance")


def _validate_teacher_provenance(
    provenance: object,
    *,
    manifest_path: Path,
    expected_model: str,
    expected_revision: str,
    num_samples: int,
) -> None:
    expected = {
        "schema": TEACHER_PROVENANCE_SCHEMA,
        "model": expected_model,
        "revision": expected_revision,
        "max_seq_len": 8192,
        "source_model": "Qwen/Qwen3.8-27B",
        "source_agent": "codex",
        "source_agent_version": "0.146.0",
        "source_trajectory_schema": "ATIF-v1.7",
        "reasoning_effort": "xhigh",
        "terminal_bench_version": "2.1",
        "plan_manifest_sha256": "48f74ac79a9de776cfc36c370fcc04411e827c7837747192943771282ecd1ad7",
        "task_inventory_sha256": "f3bbf6f7a5eae0505bcaf8b9587b7691dd7cfbd98e5b2c1cb60b83f1fc86f8df",
        "split_sha256": "c3e5abde12dfae025f161dff927dfcc16b8f04e3aa77f7b95256ebbac20b85a8",
        "planned_tasks": 89,
        "included_tasks": 89,
        "included_trajectories": 89,
        "included_samples": num_samples,
        "one_trace_per_task": True,
        "all_tasks_exactly_once": True,
        "reward_counts": {"0": 54, "1": 35},
        "known_timeout_zero_mappings": 1,
        "missing_reward_policy": "map-known-agent-timeout-to-zero",
        "native_session_fallback_count": 2,
        "native_session_fallback_tasks": [
            "terminal-bench/gpt2-codegolf",
            "terminal-bench/polyglot-c-py",
        ],
        "legacy_unsigned_outcomes": True,
        "all_terminal_bench_outcomes_authenticated": False,
        "assistant_only_loss_mask": True,
        "conversion_report_path": "report.json",
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("value manifest does not prove the exact pinned Qwen3.8-27B TB2.1 teacher corpus")
    if not _is_sha256(provenance.get("tokenizer_digest")) or not _is_sha256(
        provenance.get("chat_template_sha256")
    ):
        raise ValueError("teacher value manifest has incomplete target-tokenizer provenance")
    inventories = provenance.get("source_inventories")
    if not isinstance(inventories, dict) or set(inventories) != {
        "result",
        "session",
        "trajectory",
    }:
        raise ValueError("teacher value manifest has incomplete source inventories")
    for name, inventory in inventories.items():
        if (
            not isinstance(inventory, dict)
            or inventory.get("count") != 89
            or isinstance(inventory.get("total_bytes"), bool)
            or not isinstance(inventory.get("total_bytes"), int)
            or inventory["total_bytes"] < 1
            or not _is_sha256(inventory.get("aggregate_sha256"))
        ):
            raise ValueError(f"teacher value manifest has malformed {name} inventory")
    report_digest = provenance.get("conversion_report_sha256")
    report_path = manifest_path.parent / "report.json"
    if (
        not _is_sha256(report_digest)
        or not report_path.is_file()
        or report_path.is_symlink()
        or _sha256_file(report_path) != report_digest
    ):
        raise ValueError("teacher value manifest conversion report is missing or changed")


def build_plan(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_model: str,
    expected_revision: str,
    available_gpus: int,
    max_global_batch_size: int,
) -> dict[str, object]:
    manifest = load_value_pretrain_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    # Materializing this index validates the dataset hash, row count, unique
    # IDs, token/return/mask shapes, and objective range before Ray reserves a
    # GPU placement group.
    dataset = ValuePretrainDataset(manifest.train, manifest.objective)
    raw_manifest = json.loads(manifest.path.read_text(encoding="utf-8"))
    provenance = raw_manifest.get("provenance")
    provenance_schema = provenance.get("schema") if isinstance(provenance, dict) else None
    if provenance_schema == BASELINE_PROVENANCE_SCHEMA:
        _validate_baseline_provenance(
            provenance,
            expected_model=expected_model,
            expected_revision=expected_revision,
            num_samples=len(dataset),
        )
    elif provenance_schema == TEACHER_PROVENANCE_SCHEMA:
        _validate_teacher_provenance(
            provenance,
            manifest_path=manifest.path,
            expected_model=expected_model,
            expected_revision=expected_revision,
            num_samples=len(dataset),
        )
    else:
        raise ValueError("value manifest provenance schema is not an approved TB2.1 corpus")
    batch = choose_batch_plan(
        len(dataset),
        available_gpus=available_gpus,
        max_global_batch_size=max_global_batch_size,
    )
    return {
        "schema": SCHEMA,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "train_dataset_sha256": manifest.train.sha256,
        "conversion_provenance": provenance,
        "objective": manifest.objective.to_dict(),
        "batch": {
            **asdict(batch),
            "consumed_rows": batch.consumed_rows,
            "dropped_rows": 0,
            "repeated_rows": 0,
        },
        "selection": {
            "available_gpus": available_gpus,
            "max_global_batch_size": max_global_batch_size,
            "policy": "max_exact_dp_then_largest_bounded_exact_gbs",
        },
    }


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--available-gpus", type=int, required=True)
    parser.add_argument("--max-global-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = build_plan(
        args.manifest.resolve(),
        expected_manifest_sha256=args.manifest_sha256,
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
        available_gpus=args.available_gpus,
        max_global_batch_size=args.max_global_batch_size,
    )
    encoded = _canonical(payload)
    if args.output is not None:
        _exclusive_write(args.output.resolve(), encoded)
    print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
