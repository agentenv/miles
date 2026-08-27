"""Auditable explained-variance gates for offline value checkpoints.

This module intentionally has no Torch or Ray dependency.  GPU workers return
only the five sufficient statistics below; the driver validates rank/sample
coverage and computes both batch-local and aggregate explained variance.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miles.value_pretraining import ValuePretrainDataset, ValueSample


VALUE_EVAL_REPORT_SCHEMA = "miles.value-pretrain-eval.v2"
_MIN_TARGET_VARIANCE = 1e-8


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def scalar_active_label(sample: ValueSample) -> float:
    """Return the row's single active target, rejecting ambiguous rows."""

    labels = {float(target) for target, active in zip(sample.returns, sample.loss_mask, strict=True) if active}
    if len(labels) != 1:
        raise ValueError(
            f"value sample {sample.sample_id!r} must have one scalar active label; " f"got {sorted(labels)!r}"
        )
    return next(iter(labels))


def select_mixed_label_batches(
    dataset: ValuePretrainDataset,
    *,
    global_batch_size: int,
    num_batches: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Select deterministic, disjoint batches containing at least two labels.

    Two differently labelled rows are reserved for every batch before the
    remaining slots are filled.  This makes nonzero target variance a property
    of the batch plan rather than a lucky consequence of a shuffle.
    """

    global_batch_size = _positive_int(global_batch_size, field="global_batch_size")
    num_batches = _positive_int(num_batches, field="num_batches")
    if global_batch_size < 2:
        raise ValueError("mixed-label value evaluation requires global_batch_size >= 2")
    if global_batch_size * num_batches > len(dataset):
        raise ValueError(
            "mixed-label value evaluation requires disjoint rows: "
            f"requested {global_batch_size * num_batches}, dataset has {len(dataset)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    groups: dict[float, list[int]] = {}
    for index in range(len(dataset)):
        groups.setdefault(scalar_active_label(dataset.get(index)), []).append(index)
    if len(groups) < 2:
        raise ValueError("mixed-label value evaluation requires at least two target labels")

    rng = random.Random(seed)
    for label in sorted(groups):
        rng.shuffle(groups[label])

    batches: list[list[int]] = [[] for _ in range(num_batches)]
    # Reserve two distinct labels for each batch.  Choosing the two largest
    # remaining groups preserves scarce labels for as many batches as possible.
    for batch in batches:
        available = sorted(
            ((len(indices), label) for label, indices in groups.items() if indices),
            key=lambda item: (-item[0], item[1]),
        )
        if len(available) < 2:
            raise ValueError(
                "dataset label counts cannot supply two distinct labels to every "
                f"requested batch (num_batches={num_batches})"
            )
        first_label = available[0][1]
        second_label = available[1][1]
        batch.extend((groups[first_label].pop(), groups[second_label].pop()))

    remaining = [index for indices in groups.values() for index in indices]
    rng.shuffle(remaining)
    cursor = 0
    for batch in batches:
        needed = global_batch_size - len(batch)
        batch.extend(remaining[cursor : cursor + needed])
        cursor += needed
        rng.shuffle(batch)

    planned = tuple(tuple(batch) for batch in batches)
    flat = [index for batch in planned for index in batch]
    if len(flat) != len(set(flat)):
        raise AssertionError("mixed-label value evaluation reused a dataset row")
    for batch in planned:
        labels = {scalar_active_label(dataset.get(index)) for index in batch}
        if len(labels) < 2:
            raise AssertionError("mixed-label value evaluation produced a homogeneous batch")
    return planned


@dataclass(frozen=True)
class ValueEvalStats:
    n: int
    returns_sum: float
    returns_sq_sum: float
    residual_sum: float
    residual_sq_sum: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ValueEvalStats:
        n = raw.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("critic evaluation stats.n must be a positive integer")
        values: dict[str, float] = {}
        for field in (
            "returns_sum",
            "returns_sq_sum",
            "residual_sum",
            "residual_sq_sum",
        ):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"critic evaluation stats.{field} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"critic evaluation stats.{field} must be finite")
            values[field] = value
        return cls(n=n, **values)

    def __add__(self, other: ValueEvalStats) -> ValueEvalStats:
        return ValueEvalStats(
            n=self.n + other.n,
            returns_sum=self.returns_sum + other.returns_sum,
            returns_sq_sum=self.returns_sq_sum + other.returns_sq_sum,
            residual_sum=self.residual_sum + other.residual_sum,
            residual_sq_sum=self.residual_sq_sum + other.residual_sq_sum,
        )

    def metrics(self) -> dict[str, float | int]:
        if self.n <= 1:
            raise ValueError("explained variance requires at least two active targets")
        target_variance = self.returns_sq_sum / self.n - (self.returns_sum / self.n) ** 2
        residual_variance = self.residual_sq_sum / self.n - (self.residual_sum / self.n) ** 2
        # Small negative values can arise from floating-point cancellation.
        if -1e-12 < target_variance < 0:
            target_variance = 0.0
        if -1e-12 < residual_variance < 0:
            residual_variance = 0.0
        if target_variance <= _MIN_TARGET_VARIANCE:
            raise ValueError("critic evaluation batch has zero target variance; explained variance is undefined")
        explained_variance = 1.0 - residual_variance / target_variance
        if not all(math.isfinite(value) for value in (target_variance, residual_variance, explained_variance)):
            raise ValueError("critic evaluation produced non-finite variance metrics")
        return {
            "active_targets": self.n,
            "target_mean": self.returns_sum / self.n,
            "target_variance": target_variance,
            "residual_mean": self.residual_sum / self.n,
            "residual_variance": residual_variance,
            "explained_variance": explained_variance,
        }


def combine_rank_stats(
    reports: Sequence[Mapping[str, object] | None],
    *,
    expected_dp_size: int,
    expected_sample_indices: Sequence[int],
    stats_field: str | None = None,
    expected_active_targets: int | None = None,
) -> ValueEvalStats:
    """Validate one report per DP rank and exact sample coverage, then sum."""

    expected_dp_size = _positive_int(expected_dp_size, field="expected_dp_size")
    concrete = [report for report in reports if report is not None]
    if len(concrete) != expected_dp_size:
        raise RuntimeError(f"critic evaluation expected {expected_dp_size} DP reports, got {len(concrete)}")
    ranks: list[int] = []
    observed_indices: list[int] = []
    combined: ValueEvalStats | None = None
    for report in concrete:
        rank = report.get("dp_rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError("critic evaluation report has an invalid dp_rank")
        ranks.append(rank)
        sample_indices = report.get("sample_indices")
        if not isinstance(sample_indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in sample_indices
        ):
            raise RuntimeError("critic evaluation report has invalid sample_indices")
        observed_indices.extend(sample_indices)
        raw_stats: object = report if stats_field is None else report.get(stats_field)
        if not isinstance(raw_stats, Mapping):
            label = "top-level" if stats_field is None else stats_field
            raise RuntimeError(f"critic evaluation report is missing {label} sufficient statistics")
        stats = ValueEvalStats.from_mapping(raw_stats)
        combined = stats if combined is None else combined + stats

    if sorted(ranks) != list(range(expected_dp_size)):
        raise RuntimeError(f"critic evaluation DP rank coverage is invalid: {sorted(ranks)!r}")
    if Counter(observed_indices) != Counter(expected_sample_indices):
        raise RuntimeError("critic evaluation sample coverage differs from the deterministic batch plan")
    assert combined is not None
    if expected_active_targets is not None:
        expected_active_targets = _positive_int(
            expected_active_targets,
            field="expected_active_targets",
        )
        if combined.n != expected_active_targets:
            raise RuntimeError(
                "critic evaluation active-target coverage differs from the "
                f"validated dataset: expected {expected_active_targets}, got {combined.n}"
            )
    return combined


def gate_metrics(batch_metrics: Sequence[Mapping[str, object]]) -> tuple[ValueEvalStats, dict[str, object]]:
    """Apply the aggregate-positive and strict-majority-positive EV gate."""

    if not batch_metrics:
        raise ValueError("critic evaluation gate requires at least one batch")
    stats = [ValueEvalStats.from_mapping(batch["sufficient_stats"]) for batch in batch_metrics]
    aggregate = stats[0]
    for item in stats[1:]:
        aggregate += item
    aggregate_metrics = aggregate.metrics()
    positive_batches = sum(float(batch["metrics"]["explained_variance"]) > 0 for batch in batch_metrics)
    majority_positive = positive_batches * 2 > len(batch_metrics)
    aggregate_positive = float(aggregate_metrics["explained_variance"]) > 0
    return aggregate, {
        "passed": aggregate_positive and majority_positive,
        "aggregate_ev_positive": aggregate_positive,
        "strict_majority_batch_ev_positive": majority_positive,
        "positive_batches": positive_batches,
        "evaluable_batches": len(batch_metrics),
        "required_positive_batches": len(batch_metrics) // 2 + 1,
        "aggregate_metrics": aggregate_metrics,
    }
