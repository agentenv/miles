from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[2] / "tools" / "probes" / "plan_tbench21_value_pretrain.py"
SPEC = importlib.util.spec_from_file_location("plan_tbench21_value_pretrain", TOOL)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


@pytest.mark.parametrize(
    "samples,gpus,max_gbs,expected",
    [
        (356, 8, 64, (4, 4, 89)),
        (358, 8, 64, (2, 2, 179)),
        (360, 8, 64, (8, 40, 9)),
        (384, 8, 64, (8, 64, 6)),
        (420, 8, 64, (7, 42, 10)),
        (356, 1, 32, (1, 4, 89)),
        (360, 8, 4, (4, 4, 90)),
    ],
)
def test_batch_plan_has_exact_coverage_and_maximizes_dp_then_gbs(
    samples: int,
    gpus: int,
    max_gbs: int,
    expected: tuple[int, int, int],
) -> None:
    plan = planner.choose_batch_plan(
        samples,
        available_gpus=gpus,
        max_global_batch_size=max_gbs,
    )

    assert (plan.dp_size, plan.global_batch_size, plan.optimizer_steps) == expected
    assert plan.consumed_rows == samples
    assert samples % plan.global_batch_size == 0
    assert plan.global_batch_size % plan.dp_size == 0


@pytest.mark.parametrize("field", ["num_samples", "available_gpus", "max_global_batch_size"])
def test_batch_plan_rejects_non_positive_inputs(field: str) -> None:
    values = {
        "num_samples": 356,
        "available_gpus": 8,
        "max_global_batch_size": 64,
    }
    values[field] = 0
    with pytest.raises(ValueError, match=field):
        planner.choose_batch_plan(
            values["num_samples"],
            available_gpus=values["available_gpus"],
            max_global_batch_size=values["max_global_batch_size"],
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_plan_binds_full_baseline_provenance_and_dataset(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"tb21:baseline:task-{index // 4:03d}:r{index % 4}:segment-0",
                    "tokens": [1, 2],
                    "response_length": 1,
                    "returns": [float(index % 2)],
                    "loss_mask": [1],
                },
                separators=(",", ":"),
            )
            + "\n"
            for index in range(356)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "miles.value-pretrain.v1",
                "train": {
                    "path": train.name,
                    "sha256": _sha256(train),
                    "num_samples": 356,
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
                    "model": "Qwen/Qwen3.5-0.8B",
                    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                    "plan_manifest_sha256": "c" * 64,
                    "selection_sha256": "a" * 64,
                    "planned_trajectories": 356,
                    "included_trajectories": 356,
                    "included_segments": 356,
                    "all_baseline_segments_exactly_once": True,
                    "all_terminal_bench_outcomes_authenticated": True,
                    "authenticated_outcomes_bound_to_rows": True,
                    "critic_pretraining_exposes_actor_eval_tasks": True,
                    "source_shards": {f"/root/run/island-{index}/rollout_data/0.pt": "b" * 64 for index in range(8)},
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    payload = planner.build_plan(
        manifest,
        expected_manifest_sha256=_sha256(manifest),
        expected_model="Qwen/Qwen3.5-0.8B",
        expected_revision="2fc06364715b967f1860aea9cf38778875588b17",
        available_gpus=8,
        max_global_batch_size=64,
    )

    assert payload["batch"] == {
        "num_samples": 356,
        "dp_size": 4,
        "global_batch_size": 4,
        "optimizer_steps": 89,
        "epochs": 1,
        "consumed_rows": 356,
        "dropped_rows": 0,
        "repeated_rows": 0,
    }
    loaded = planner.load_value_pretrain_manifest(
        manifest,
        expected_sha256=_sha256(manifest),
    )
    dataset = planner.ValuePretrainDataset(loaded.train, loaded.objective)
    batches = list(dataset.batch_indices(global_batch_size=4, epochs=1, seed=82621))
    assert len(batches) == 89
    assert sorted(index for batch in batches for index in batch) == list(range(356))
    with pytest.raises(ValueError, match="exact pinned TB2.1 baseline"):
        planner.build_plan(
            manifest,
            expected_manifest_sha256=_sha256(manifest),
            expected_model="wrong/model",
            expected_revision="2fc06364715b967f1860aea9cf38778875588b17",
            available_gpus=8,
            max_global_batch_size=64,
        )

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["provenance"]["all_terminal_bench_outcomes_authenticated"] = False
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exact pinned TB2.1 baseline"):
        planner.build_plan(
            manifest,
            expected_manifest_sha256=_sha256(manifest),
            expected_model="Qwen/Qwen3.5-0.8B",
            expected_revision="2fc06364715b967f1860aea9cf38778875588b17",
            available_gpus=8,
            max_global_batch_size=64,
        )

    raw["provenance"]["all_terminal_bench_outcomes_authenticated"] = True
    raw["provenance"]["source_shards"].pop("/root/run/island-7/rollout_data/0.pt")
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete source provenance"):
        planner.build_plan(
            manifest,
            expected_manifest_sha256=_sha256(manifest),
            expected_model="Qwen/Qwen3.5-0.8B",
            expected_revision="2fc06364715b967f1860aea9cf38778875588b17",
            available_gpus=8,
            max_global_batch_size=64,
        )
