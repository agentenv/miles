"""Fail-closed post-run validator for the two-rollout SAO TB2.1 gate."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


EXPECTED_ROLLOUTS = (0, 1)
EXPECTED_CHECKPOINT_ITERATION = "1"
MIN_CRITIC_UPDATES_PER_ROLLOUT = 2
FAILURE_SIGNATURES = {
    "python_traceback": re.compile(r"Traceback \(most recent call last\):"),
    "cuda_oom": re.compile(r"(?:CUDA out of memory|torch\.(?:cuda\.)?OutOfMemoryError|OutOfMemoryError:)"),
    "ray_task_error": re.compile(r"\bRayTaskError\b"),
    "structured_failure": re.compile(r"\bphase=end ok=false\b"),
    "failed_ray_job": re.compile(r"\bJob ['\"][^'\"]+['\"] failed\b"),
    "nonzero_container_exit": re.compile(r"\bcontainer=exited exit=[1-9][0-9]*\b"),
    "oom_exit": re.compile(r"\boom=true\b", re.IGNORECASE),
}


class EvidenceError(RuntimeError):
    """The captured artifacts do not prove a valid E2E gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(float(value))


def _binary_reward(value: object, *, field: str) -> float:
    _require(_finite_number(value), f"{field} is not an explicit finite number")
    result = float(value)
    _require(result in (0.0, 1.0), f"{field} is not binary: {result!r}")
    return result


def _load_rollout(path: Path, rollout_id: int) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing safe rollout artifact: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EvidenceError(f"cannot load rollout artifact {path}: {error}") from error
    _require(isinstance(payload, dict), f"rollout {rollout_id} payload is not an object")
    _require(payload.get("rollout_id") == rollout_id, f"rollout {rollout_id} has the wrong embedded ID")
    samples = payload.get("samples")
    _require(isinstance(samples, list), f"rollout {rollout_id} samples are missing")
    _require(len(samples) == 1, f"rollout {rollout_id} has {len(samples)} accepted samples, expected 1")
    sample = samples[0]
    _require(isinstance(sample, dict), f"rollout {rollout_id} sample is not an object")
    return sample


def _validate_sample(sample: dict[str, Any], rollout_id: int) -> dict[str, object]:
    prefix = f"rollout {rollout_id} sample"
    _require(sample.get("status") in {"completed", "truncated"}, f"{prefix} was not accepted")
    _require(not sample.get("remove_sample", False), f"{prefix} is marked for removal")

    metadata = sample.get("metadata")
    _require(isinstance(metadata, dict), f"{prefix} metadata is missing")
    _require(metadata.get("exit_status") == "completed", f"{prefix} has no completed canonical verdict")
    sample_reward = _binary_reward(sample.get("reward"), field=f"{prefix}.reward")
    metadata_reward = _binary_reward(metadata.get("reward"), field=f"{prefix}.metadata.reward")
    _require(sample_reward == metadata_reward, f"{prefix} reward disagrees with its verifier metadata")

    response_length = sample.get("response_length")
    _require(
        isinstance(response_length, int) and not isinstance(response_length, bool) and response_length > 0,
        f"{prefix} has no positive response length",
    )
    tokens = sample.get("tokens")
    _require(isinstance(tokens, list) and len(tokens) >= response_length, f"{prefix} tokens are misaligned")

    loss_mask = sample.get("loss_mask")
    _require(isinstance(loss_mask, list), f"{prefix} loss mask is missing")
    _require(len(loss_mask) == response_length, f"{prefix} loss mask is misaligned")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value in (0, 1) for value in loss_mask),
        f"{prefix} loss mask is not binary",
    )
    _require(any(loss_mask), f"{prefix} has no trainable response tokens")

    log_probs = sample.get("rollout_log_probs")
    _require(isinstance(log_probs, list), f"{prefix} behavior logprobs are missing")
    _require(len(log_probs) == response_length, f"{prefix} behavior logprobs are misaligned")
    _require(all(_finite_number(value) for value in log_probs), f"{prefix} behavior logprobs are non-finite")

    versions = sample.get("weight_versions")
    _require(isinstance(versions, list) and versions, f"{prefix} weight versions are missing")
    normalized_versions = {str(value) for value in versions}
    expected_version = str(rollout_id + 1)
    _require(len(normalized_versions) == 1, f"{prefix} mixes weight versions: {sorted(normalized_versions)!r}")
    _require(normalized_versions == {expected_version}, f"{prefix} did not use weight version {expected_version}")

    return {
        "reward": sample_reward,
        "response_length": response_length,
        "active_tokens": sum(loss_mask),
        "weight_version": expected_version,
    }


def _read_events(events_dir: Path) -> list[dict[str, Any]]:
    _require(events_dir.is_dir() and not events_dir.is_symlink(), f"missing safe event directory: {events_dir}")
    paths = sorted(events_dir.glob("*.jsonl"))
    _require(bool(paths), f"no event logs found in {events_dir}")
    events: list[dict[str, Any]] = []
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), f"unsafe event log: {path}")
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"malformed event at {path}:{line_number}") from error
            _require(isinstance(event, dict), f"event at {path}:{line_number} is not an object")
            events.append(event)
    _require(bool(events), f"event logs in {events_dir} are empty")
    return events


def _validate_update_metrics(metrics: object, *, required: tuple[str, ...], label: str) -> None:
    _require(isinstance(metrics, dict), f"{label} metrics are missing")
    for key in required:
        _require(key in metrics, f"{label} is missing {key}")
        _require(_finite_number(metrics[key]), f"{label} has non-finite {key}")
    for key, value in metrics.items():
        if str(key).startswith("train/"):
            _require(_finite_number(value), f"{label} has non-finite training metric {key}")


def _validate_updates(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    for event in events:
        if event.get("type") == "train_group_step_end":
            outcomes = event.get("cell_outcomes")
            if isinstance(outcomes, dict):
                _require("error" not in outcomes.values(), "training event records a failed cell")

    report: dict[str, dict[str, int]] = {}
    role_specs = {
        "actor": (("train/loss", "train/grad_norm"), 1, "train/grad_norm"),
        "critic": (
            ("train/critic-value_loss", "train/critic-grad_norm"),
            MIN_CRITIC_UPDATES_PER_ROLLOUT,
            "train/critic-grad_norm",
        ),
    }
    for role, (required, minimum_count, discriminator) in role_specs.items():
        per_rollout: dict[str, int] = {}
        for rollout_id in EXPECTED_ROLLOUTS:
            matching = []
            for event in events:
                source = event.get("source")
                metrics = event.get("metrics")
                if event.get("type") == "metric" and event.get("rollout_id") == rollout_id and isinstance(source, dict) and source.get("component") == role and isinstance(metrics, dict) and discriminator in metrics:
                    matching.append(event)
            _require(
                len(matching) >= minimum_count,
                f"rollout {rollout_id} has {len(matching)} {role} updates, expected at least {minimum_count}",
            )
            for index, event in enumerate(matching):
                _validate_update_metrics(
                    event.get("metrics"),
                    required=required,
                    label=f"rollout {rollout_id} {role} update {index}",
                )
                metrics = event["metrics"]
                _require(metrics.get("train/step") == rollout_id, f"rollout {rollout_id} {role} update has the wrong step")
            per_rollout[str(rollout_id)] = len(matching)
        report[role] = per_rollout
    return report


def _checkpoint_marker(root: Path, role: str) -> str:
    marker_path = root / "latest_checkpointed_iteration.txt"
    _require(marker_path.is_file() and not marker_path.is_symlink(), f"missing safe {role} checkpoint marker")
    marker = marker_path.read_text(encoding="utf-8").strip()
    _require(marker == EXPECTED_CHECKPOINT_ITERATION, f"{role} checkpoint marker is {marker!r}, expected '1'")
    return marker


def _verify_checkpoint_layout(actor: Path, critic: Path, hf: Path) -> dict[str, int]:
    # Delegate tensor coverage, shapes, dtypes, and actor/critic backbone layout
    # to the existing metadata-only Qwen3.5 checkpoint validator.
    if __package__:
        from .check_sao_qwen35_raw_checkpoints import verify
    else:
        from check_sao_qwen35_raw_checkpoints import verify

    return verify(actor, critic, hf)


def _scan_console_logs(paths: list[Path]) -> int:
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), f"missing safe console log: {path}")
        saw_content = False
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                saw_content = True
                for name, pattern in FAILURE_SIGNATURES.items():
                    _require(not pattern.search(line), f"console log {path}:{line_number} contains {name}")
        _require(saw_content, f"console log is empty: {path}")
    return len(paths)


def validate(
    *,
    details: Path,
    actor: Path,
    critic: Path,
    hf: Path,
    logs: list[Path] | None = None,
    checkpoint_verifier: Callable[[Path, Path, Path], dict[str, int]] | None = None,
) -> dict[str, object]:
    rollout_dir = details / "rollout_data"
    _require(rollout_dir.is_dir() and not rollout_dir.is_symlink(), f"missing safe rollout directory: {rollout_dir}")
    numeric_rollouts = {path.stem for path in rollout_dir.glob("*.pt") if path.stem.isdecimal()}
    _require(numeric_rollouts == {"0", "1"}, f"unexpected numeric rollout evidence: {sorted(numeric_rollouts)!r}")

    rollouts = {
        str(rollout_id): _validate_sample(
            _load_rollout(rollout_dir / f"{rollout_id}.pt", rollout_id),
            rollout_id,
        )
        for rollout_id in EXPECTED_ROLLOUTS
    }
    updates = _validate_updates(_read_events(details / "events"))
    markers = {
        "actor": _checkpoint_marker(actor, "actor"),
        "critic": _checkpoint_marker(critic, "critic"),
    }
    verifier = checkpoint_verifier or _verify_checkpoint_layout
    checkpoint_layout = verifier(actor, critic, hf)
    _require(isinstance(checkpoint_layout, dict), "checkpoint verifier returned no report")
    log_count = _scan_console_logs(logs or [])

    return {
        "rollouts": rollouts,
        "updates": updates,
        "checkpoint_markers": markers,
        "checkpoint_layout": checkpoint_layout,
        "console_logs_checked": log_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = validate(
        details=args.details,
        actor=args.actor,
        critic=args.critic,
        hf=args.hf,
        logs=args.log,
    )
    print(
        "SAO_TBENCH21_E2E_OK",
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()
