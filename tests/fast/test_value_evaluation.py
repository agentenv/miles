from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from miles.value_evaluation import (
    ValueEvalStats,
    combine_rank_stats,
    gate_metrics,
    scalar_active_label,
    select_mixed_label_batches,
)
from miles.value_pretraining import VALUE_PRETRAIN_SCHEMA, ValuePretrainDataset, load_value_pretrain_manifest


def _dataset(tmp_path: Path, labels: list[float]) -> ValuePretrainDataset:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "tokens": [10, 20, 30 + index, 40 + index],
            "response_length": 2,
            "returns": [label, label],
            "loss_mask": [1, 1],
        }
        for index, label in enumerate(labels)
    ]
    encoded = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(encoded)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": VALUE_PRETRAIN_SCHEMA,
                "train": {
                    "path": data_path.name,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "num_samples": len(rows),
                },
                "objective": {
                    "loss_type": "classification",
                    "num_bins": 51,
                    "reward_range": [0.0, 1.0],
                    "target_type": "hl_gauss",
                    "hl_gauss_sigma_ratio": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = load_value_pretrain_manifest(manifest_path)
    return ValuePretrainDataset(manifest.train, manifest.objective)


def _batch(stats: ValueEvalStats) -> dict[str, object]:
    return {
        "sufficient_stats": {
            "n": stats.n,
            "returns_sum": stats.returns_sum,
            "returns_sq_sum": stats.returns_sq_sum,
            "residual_sum": stats.residual_sum,
            "residual_sq_sum": stats.residual_sq_sum,
        },
        "metrics": stats.metrics(),
    }


def test_mixed_label_batches_are_deterministic_disjoint_and_mixed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [0.0, 1.0] * 6)

    first = select_mixed_label_batches(
        dataset,
        global_batch_size=4,
        num_batches=3,
        seed=17,
    )
    second = select_mixed_label_batches(
        dataset,
        global_batch_size=4,
        num_batches=3,
        seed=17,
    )

    assert first == second
    flat = [index for batch in first for index in batch]
    assert len(flat) == len(set(flat)) == 12
    assert all(len({scalar_active_label(dataset.get(index)) for index in batch}) >= 2 for batch in first)


def test_mixed_label_batches_reject_insufficient_minority_rows(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [0.0] * 7 + [1.0])

    with pytest.raises(ValueError, match="two distinct labels"):
        select_mixed_label_batches(
            dataset,
            global_batch_size=4,
            num_batches=2,
            seed=17,
        )


def test_rank_stats_require_exact_dp_and_sample_coverage() -> None:
    reports = [
        {
            "dp_rank": 0,
            "sample_indices": [3],
            "n": 2,
            "returns_sum": 0.0,
            "returns_sq_sum": 0.0,
            "residual_sum": 0.0,
            "residual_sq_sum": 0.0,
        },
        {
            "dp_rank": 1,
            "sample_indices": [9],
            "n": 2,
            "returns_sum": 2.0,
            "returns_sq_sum": 2.0,
            "residual_sum": 0.0,
            "residual_sq_sum": 0.0,
        },
    ]

    combined = combine_rank_stats(
        reports,
        expected_dp_size=2,
        expected_sample_indices=(9, 3),
        expected_active_targets=4,
    )
    assert combined.metrics()["explained_variance"] == pytest.approx(1.0)

    reports[1]["sample_indices"] = [4]
    with pytest.raises(RuntimeError, match="sample coverage"):
        combine_rank_stats(
            reports,
            expected_dp_size=2,
            expected_sample_indices=(9, 3),
        )


def test_rank_stats_require_exact_active_target_coverage() -> None:
    reports = [
        {
            "dp_rank": 0,
            "sample_indices": [3, 9],
            "trajectory_weighted": {
                "n": 2,
                "returns_sum": 1.0,
                "returns_sq_sum": 1.0,
                "residual_sum": 0.0,
                "residual_sq_sum": 0.0,
            },
        }
    ]

    combined = combine_rank_stats(
        reports,
        expected_dp_size=1,
        expected_sample_indices=(3, 9),
        stats_field="trajectory_weighted",
        expected_active_targets=2,
    )
    assert combined.metrics()["explained_variance"] == pytest.approx(1.0)

    with pytest.raises(RuntimeError, match="active-target coverage"):
        combine_rank_stats(
            reports,
            expected_dp_size=1,
            expected_sample_indices=(3, 9),
            stats_field="trajectory_weighted",
            expected_active_targets=3,
        )


def test_zero_target_variance_is_not_an_evaluable_zero_ev() -> None:
    homogeneous = ValueEvalStats(
        n=4,
        returns_sum=0.0,
        returns_sq_sum=0.0,
        residual_sum=0.0,
        residual_sq_sum=0.0,
    )

    with pytest.raises(ValueError, match="zero target variance"):
        homogeneous.metrics()


def test_gate_requires_aggregate_positive_and_strict_batch_majority() -> None:
    perfect = ValueEvalStats(
        n=2,
        returns_sum=1.0,
        returns_sq_sum=1.0,
        residual_sum=0.0,
        residual_sq_sum=0.0,
    )
    weak_negative = ValueEvalStats(
        n=2,
        returns_sum=1.0,
        returns_sq_sum=1.0,
        residual_sum=0.0,
        residual_sq_sum=0.72,
    )

    _aggregate, passing = gate_metrics([_batch(perfect), _batch(perfect), _batch(weak_negative)])
    assert passing["passed"] is True
    assert passing["positive_batches"] == 2

    _aggregate, tied = gate_metrics([_batch(perfect), _batch(weak_negative)])
    assert tied["aggregate_ev_positive"] is True
    assert tied["strict_majority_batch_ev_positive"] is False
    assert tied["passed"] is False
