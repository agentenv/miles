"""Strict, dependency-free contracts for offline SAO value pretraining.

The training entry point imports this module before Ray or CUDA is initialized so
that malformed data fails without reserving GPUs.  Keeping the contract free of
Torch also makes dataset production and checkpoint handoff independently
testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALUE_PRETRAIN_SCHEMA = "miles.value-pretrain.v1"
VALUE_PRETRAIN_CONTRACT_SCHEMA = "miles.value-pretrain-checkpoint.v1"
VALUE_PRETRAIN_CONTRACT_NAME = "value_pretrain_contract.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True)
class ValueObjective:
    loss_type: str
    num_bins: int
    reward_low: float
    reward_high: float
    target_type: str
    hl_gauss_sigma_ratio: float

    @classmethod
    def from_dict(cls, raw: object) -> ValueObjective:
        if not isinstance(raw, dict):
            raise ValueError("objective must be an object")
        loss_type = raw.get("loss_type", "classification")
        if loss_type not in {"mse", "classification"}:
            raise ValueError("objective.loss_type must be 'mse' or 'classification'")

        num_bins = _require_positive_int(raw.get("num_bins", 51), field="objective.num_bins")
        if loss_type == "classification" and num_bins < 2:
            raise ValueError("classification value pretraining requires at least two bins")

        reward_range = raw.get("reward_range", [0.0, 1.0])
        if not isinstance(reward_range, list) or len(reward_range) != 2:
            raise ValueError("objective.reward_range must contain [LOW, HIGH]")
        reward_low = _require_finite_float(reward_range[0], field="objective.reward_range[0]")
        reward_high = _require_finite_float(reward_range[1], field="objective.reward_range[1]")
        if reward_low >= reward_high:
            raise ValueError("objective.reward_range must have LOW < HIGH")

        target_type = raw.get("target_type", "hl_gauss")
        if target_type not in {"two_hot", "hl_gauss"}:
            raise ValueError("objective.target_type must be 'two_hot' or 'hl_gauss'")
        sigma_ratio = _require_finite_float(
            raw.get("hl_gauss_sigma_ratio", 0.75),
            field="objective.hl_gauss_sigma_ratio",
        )
        if sigma_ratio <= 0:
            raise ValueError("objective.hl_gauss_sigma_ratio must be positive")

        return cls(
            loss_type=loss_type,
            num_bins=num_bins,
            reward_low=reward_low,
            reward_high=reward_high,
            target_type=target_type,
            hl_gauss_sigma_ratio=sigma_ratio,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "loss_type": self.loss_type,
            "num_bins": self.num_bins,
            "reward_range": [self.reward_low, self.reward_high],
            "target_type": self.target_type,
            "hl_gauss_sigma_ratio": self.hl_gauss_sigma_ratio,
        }


@dataclass(frozen=True)
class ValueDatasetSpec:
    path: Path
    sha256: str
    num_samples: int


@dataclass(frozen=True)
class ValuePretrainManifest:
    path: Path
    sha256: str
    train: ValueDatasetSpec
    heldout: ValueDatasetSpec | None
    objective: ValueObjective


def _dataset_spec(raw: object, *, field: str, manifest_dir: Path) -> ValueDatasetSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    path_value = raw.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{field}.path must be a non-empty string")
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_dir / path
    return ValueDatasetSpec(
        path=path.resolve(),
        sha256=_require_sha256(raw.get("sha256"), field=f"{field}.sha256"),
        num_samples=_require_positive_int(raw.get("num_samples"), field=f"{field}.num_samples"),
    )


def load_value_pretrain_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> ValuePretrainManifest:
    manifest_path = Path(path).resolve()
    raw_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(raw_bytes)
    if expected_sha256 is not None:
        expected_sha256 = _require_sha256(expected_sha256, field="expected manifest SHA-256")
        if manifest_sha256 != expected_sha256:
            raise ValueError(f"value-pretraining manifest SHA-256 mismatch: expected {expected_sha256}, got {manifest_sha256}")
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid value-pretraining manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("value-pretraining manifest must be an object")
    if raw.get("schema") != VALUE_PRETRAIN_SCHEMA:
        raise ValueError(f"manifest.schema must be {VALUE_PRETRAIN_SCHEMA!r}")

    heldout_raw = raw.get("heldout")
    return ValuePretrainManifest(
        path=manifest_path,
        sha256=manifest_sha256,
        train=_dataset_spec(raw.get("train"), field="train", manifest_dir=manifest_path.parent),
        heldout=(_dataset_spec(heldout_raw, field="heldout", manifest_dir=manifest_path.parent) if heldout_raw is not None else None),
        objective=ValueObjective.from_dict(raw.get("objective")),
    )


@dataclass(frozen=True)
class ValueSample:
    sample_id: str
    tokens: tuple[int, ...]
    response_length: int
    returns: tuple[float, ...]
    loss_mask: tuple[int, ...]

    def to_rollout_row(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "tokens": list(self.tokens),
            "response_length": self.response_length,
            "returns": list(self.returns),
            "loss_mask": list(self.loss_mask),
        }


def parse_value_sample(raw: object, *, location: str, objective: ValueObjective) -> ValueSample:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: sample must be an object")
    sample_id = raw.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{location}: sample_id must be a non-empty string")

    tokens_raw = raw.get("tokens")
    if not isinstance(tokens_raw, list) or len(tokens_raw) < 2:
        raise ValueError(f"{location}: tokens must contain at least two token IDs")
    tokens: list[int] = []
    for index, token in enumerate(tokens_raw):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{location}: tokens[{index}] must be a non-negative integer")
        tokens.append(token)

    response_length = _require_positive_int(raw.get("response_length"), field=f"{location}.response_length")
    if response_length >= len(tokens):
        raise ValueError(f"{location}: response_length must leave at least one prompt token")

    returns_raw = raw.get("returns")
    if not isinstance(returns_raw, list) or len(returns_raw) != response_length:
        raise ValueError(f"{location}: returns length must equal response_length")
    returns = tuple(_require_finite_float(value, field=f"{location}.returns[{index}]") for index, value in enumerate(returns_raw))
    if objective.loss_type == "classification" and any(value < objective.reward_low or value > objective.reward_high for value in returns):
        raise ValueError(f"{location}: classification returns must stay inside [{objective.reward_low}, {objective.reward_high}]")

    mask_raw = raw.get("loss_mask", [1] * response_length)
    if not isinstance(mask_raw, list) or len(mask_raw) != response_length:
        raise ValueError(f"{location}: loss_mask length must equal response_length")
    mask: list[int] = []
    for index, value in enumerate(mask_raw):
        if value not in (0, 1, False, True):
            raise ValueError(f"{location}: loss_mask[{index}] must be 0 or 1")
        mask.append(int(value))
    if not any(mask):
        raise ValueError(f"{location}: loss_mask must select at least one target token")

    return ValueSample(
        sample_id=sample_id,
        tokens=tuple(tokens),
        response_length=response_length,
        returns=returns,
        loss_mask=tuple(mask),
    )


class ValuePretrainDataset:
    """Validated JSONL dataset with deterministic, random-access batching."""

    def __init__(self, spec: ValueDatasetSpec, objective: ValueObjective) -> None:
        self.spec = spec
        self.objective = objective
        if not spec.path.is_file():
            raise FileNotFoundError(f"value-pretraining dataset does not exist: {spec.path}")
        actual_sha256 = sha256_file(spec.path)
        if actual_sha256 != spec.sha256:
            raise ValueError(f"dataset SHA-256 mismatch for {spec.path}: expected {spec.sha256}, got {actual_sha256}")

        offsets: list[int] = []
        sample_ids: set[str] = set()
        with spec.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                try:
                    raw = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{spec.path}:{line_number}: invalid JSON: {exc}") from exc
                sample = parse_value_sample(raw, location=f"{spec.path}:{line_number}", objective=objective)
                if sample.sample_id in sample_ids:
                    raise ValueError(f"{spec.path}:{line_number}: duplicate sample_id {sample.sample_id!r}")
                sample_ids.add(sample.sample_id)
                offsets.append(offset)

        if len(offsets) != spec.num_samples:
            raise ValueError(f"dataset row-count mismatch for {spec.path}: expected {spec.num_samples}, got {len(offsets)}")
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return len(self._offsets)

    def get(self, index: int) -> ValueSample:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        with self.spec.path.open("rb") as stream:
            return self._read_at(stream, index)

    def _read_at(self, stream, index: int) -> ValueSample:
        stream.seek(self._offsets[index])
        raw = json.loads(stream.readline())
        return parse_value_sample(raw, location=f"{self.spec.path}@{index}", objective=self.objective)

    def batch_indices(self, *, global_batch_size: int, epochs: int, seed: int) -> Iterator[tuple[int, ...]]:
        global_batch_size = _require_positive_int(global_batch_size, field="global_batch_size")
        epochs = _require_positive_int(epochs, field="epochs")
        if len(self) < global_batch_size:
            raise ValueError(f"dataset has {len(self)} samples, fewer than global_batch_size={global_batch_size}")
        for epoch in range(epochs):
            indices = list(range(len(self)))
            random.Random(seed + epoch).shuffle(indices)
            usable = len(indices) - (len(indices) % global_batch_size)
            for start in range(0, usable, global_batch_size):
                yield tuple(indices[start : start + global_batch_size])

    def build_rollout_batch(self, indices: tuple[int, ...]) -> dict[str, list[object]]:
        with self.spec.path.open("rb") as stream:
            samples = [self._read_at(stream, index) for index in indices]
        return {
            "tokens": [list(sample.tokens) for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "returns": [list(sample.returns) for sample in samples],
            "loss_masks": [list(sample.loss_mask) for sample in samples],
            "sample_indices": list(indices),
            "sample_ids": [sample.sample_id for sample in samples],
        }


def value_pretrain_num_steps(*, num_samples: int, global_batch_size: int, epochs: int) -> int:
    num_samples = _require_positive_int(num_samples, field="num_samples")
    global_batch_size = _require_positive_int(global_batch_size, field="global_batch_size")
    epochs = _require_positive_int(epochs, field="epochs")
    steps = (num_samples // global_batch_size) * epochs
    if steps < 1:
        raise ValueError("value pretraining requires at least one complete global batch")
    return steps


def apply_objective_to_args(args: Any, objective: ValueObjective) -> None:
    args.value_loss_type = objective.loss_type
    args.value_num_bins = objective.num_bins
    args.value_reward_range = [objective.reward_low, objective.reward_high]
    args.value_target_type = objective.target_type
    args.hl_gauss_sigma_ratio = objective.hl_gauss_sigma_ratio


def write_value_pretrain_contract(
    checkpoint_dir: str | os.PathLike[str],
    *,
    manifest: ValuePretrainManifest,
    completed_steps: int,
    model_identity: str,
    seed: int,
    global_batch_size: int,
) -> tuple[Path, str]:
    completed_steps = _require_positive_int(completed_steps, field="completed_steps")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    global_batch_size = _require_positive_int(global_batch_size, field="global_batch_size")
    if not isinstance(model_identity, str) or not model_identity:
        raise ValueError("model_identity must be a non-empty string")
    payload = {
        "schema": VALUE_PRETRAIN_CONTRACT_SCHEMA,
        "source_manifest_sha256": manifest.sha256,
        "train_dataset_sha256": manifest.train.sha256,
        "objective": manifest.objective.to_dict(),
        "batch_plan": {
            "seed": seed,
            "global_batch_size": global_batch_size,
            "drop_incomplete_batch": True,
        },
        "completed_steps": completed_steps,
        "model_identity": model_identity,
    }
    encoded = _canonical_json_bytes(payload)
    digest = _sha256_bytes(encoded)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / VALUE_PRETRAIN_CONTRACT_NAME
    temporary = directory / f".{VALUE_PRETRAIN_CONTRACT_NAME}.tmp"
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination, digest


def load_value_pretrain_contract(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = Path(checkpoint_dir) / VALUE_PRETRAIN_CONTRACT_NAME
    raw_bytes = path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if expected_sha256 is not None:
        expected_sha256 = _require_sha256(expected_sha256, field="expected checkpoint contract SHA-256")
        if digest != expected_sha256:
            raise ValueError(f"value-pretraining checkpoint contract SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schema") != VALUE_PRETRAIN_CONTRACT_SCHEMA:
        raise ValueError(f"checkpoint contract schema must be {VALUE_PRETRAIN_CONTRACT_SCHEMA!r}")
    _require_sha256(raw.get("source_manifest_sha256"), field="source_manifest_sha256")
    _require_sha256(raw.get("train_dataset_sha256"), field="train_dataset_sha256")
    ValueObjective.from_dict(raw.get("objective"))
    batch_plan = raw.get("batch_plan")
    if not isinstance(batch_plan, dict):
        raise ValueError("batch_plan must be an object")
    if isinstance(batch_plan.get("seed"), bool) or not isinstance(batch_plan.get("seed"), int):
        raise ValueError("batch_plan.seed must be an integer")
    _require_positive_int(batch_plan.get("global_batch_size"), field="batch_plan.global_batch_size")
    if batch_plan.get("drop_incomplete_batch") is not True:
        raise ValueError("batch_plan.drop_incomplete_batch must be true")
    _require_positive_int(raw.get("completed_steps"), field="completed_steps")
    if not isinstance(raw.get("model_identity"), str) or not raw["model_identity"]:
        raise ValueError("model_identity must be a non-empty string")
    return raw
