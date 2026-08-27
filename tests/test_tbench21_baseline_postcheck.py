from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.probes import postcheck_tbench21_baseline_wave as postcheck


def _write_json(path: Path, value: object, *, private: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    if private:
        path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _plan(tmp_path: Path) -> tuple[Path, list[dict[str, int]]]:
    plan_dir = tmp_path / "plan"
    rows = [[] for _ in range(8)]
    for task_index in range(89):
        task_id = f"task-{task_index:02d}"
        for replica in range(4):
            island_id = (task_index * 4 + replica) % 8
            rows[island_id].append(
                {
                    "messages": [{"role": "system", "content": "solve"}],
                    "metadata": {
                        "task_id": task_id,
                        "sample_id": f"baseline:{task_id}:r{replica}",
                        "split": "baseline",
                        "island_id": island_id,
                        "rollout_seed": postcheck.ROLLOUT_SEED_BASE + island_id,
                        "episode_timeout_seconds": 1800,
                    },
                }
            )
    files = {}
    for island_id, island_rows in enumerate(rows):
        path = plan_dir / f"baseline/island-{island_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in island_rows)
        path.write_bytes(encoded)
        files[f"baseline/island-{island_id}.jsonl"] = hashlib.sha256(encoded).hexdigest()
    _write_json(
        plan_dir / "manifest.json",
        {
            "schema": postcheck.PLAN_SCHEMA,
            "topology": {"islands": 8, "one_physical_gpu_per_island": True},
            "rollouts": {
                "baseline": 356,
                "per_task": 4,
                "episode_timeout_seconds": 1800,
                "seed_base": postcheck.ROLLOUT_SEED_BASE,
                "per_island_seeds": [postcheck.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            },
            "files": files,
        },
    )
    expected = [{row["metadata"]["sample_id"]: index for index, row in enumerate(island)} for island in rows]
    return plan_dir, expected


def _sample(
    *,
    island_id: int,
    sample_id: str,
    index: int,
    segment_index: int = 0,
    trajectory_id: str | None = None,
) -> dict:
    return {
        "index": index,
        "group_index": index,
        "status": "completed",
        "metadata": {
            "sample_id": sample_id,
            "split": "baseline",
            "island_id": island_id,
            "rollout_seed": postcheck.ROLLOUT_SEED_BASE + island_id,
            "episode_timeout_seconds": 1800,
            "exit_status": "completed",
            "compaction_trajectory_id": trajectory_id or f"trajectory-{island_id}-{index}",
            "compaction_segment_index": segment_index,
            "compaction_segment_type": ("execution" if segment_index % 2 == 0 else "summary"),
            "secrlenv_infrastructure_503_retries": 0,
        },
    }


def _payload(island_id: int, expected: dict[str, int]) -> dict:
    samples = [
        _sample(
            island_id=island_id,
            sample_id=sample_id,
            index=index,
        )
        for sample_id, index in expected.items()
    ]
    first_id = next(iter(expected))
    samples.pop(0)
    samples[:0] = [
        _sample(
            island_id=island_id,
            sample_id=first_id,
            index=0,
            segment_index=segment,
            trajectory_id=f"trajectory-{island_id}-0",
        )
        for segment in range(3)
    ]
    return {"rollout_id": 0, "metadata": {}, "samples": samples}


def _inspection(island_id: int, container_id: str, *, exit_code: int = 0) -> dict:
    return {
        "Id": container_id,
        "Name": f"/tbench21-baseline-island-{island_id}",
        "Image": postcheck.RUNTIME_IMAGE_ID,
        "State": {
            "Status": "exited",
            "Running": False,
            "Dead": False,
            "OOMKilled": False,
            "ExitCode": exit_code,
            "Error": "",
        },
        "Config": {
            "Labels": {
                "yeto.run": "tbench21-sao-rollout",
                "yeto.phase": "baseline",
                "yeto.island": str(island_id),
            }
        },
    }


def test_payload_accepts_compaction_segments_without_cycling(tmp_path):
    _, expected = _plan(tmp_path)
    source = tmp_path / "0.pt"
    source.write_bytes(b"pt")
    report = postcheck._validate_payload(
        _payload(0, expected[0]),
        island_id=0,
        expected=expected[0],
        source=source,
    )
    assert report["logical_trajectories"] == 45
    assert report["segments"] == 47
    assert report["infrastructure_aborted_replacements"] == 0


def test_payload_rejects_cycled_index_and_replacement_marker(tmp_path):
    _, expected = _plan(tmp_path)
    source = tmp_path / "0.pt"
    source.write_bytes(b"pt")
    cycled = _payload(0, expected[0])
    cycled["samples"][-1]["index"] = 45
    cycled["samples"][-1]["group_index"] = 45
    with pytest.raises(postcheck.BaselinePostcheckError, match="cycled, replaced"):
        postcheck._validate_payload(cycled, island_id=0, expected=expected[0], source=source)

    replaced = _payload(0, expected[0])
    replaced["samples"][-1]["metadata"]["replacement_attempts"] = 1
    with pytest.raises(postcheck.BaselinePostcheckError, match="replacement marker"):
        postcheck._validate_payload(replaced, island_id=0, expected=expected[0], source=source)


def test_full_postcheck_requires_exit_zero_and_exact_one_pt(tmp_path, monkeypatch):
    plan_dir, expected = _plan(tmp_path)
    output = tmp_path / "baseline"
    output.mkdir()
    records = []
    payloads = {}
    inspections = []
    for island_id in range(8):
        island = output / f"island-{island_id}"
        source = island / "details/rollout_data/0.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"pt-{island_id}".encode())
        payloads[source] = _payload(island_id, expected[island_id])
        shard = plan_dir / f"baseline/island-{island_id}.jsonl"
        container_id = str(island_id) * 64
        records.append(
            {
                "island_id": island_id,
                "name": f"tbench21-baseline-island-{island_id}",
                "container_id": container_id,
                "plan_shard": str(shard),
                "plan_shard_sha256": postcheck._sha256(shard),
                "output": str(island),
            }
        )
        inspections.append(_inspection(island_id, container_id))
    launch_path = output / "launch-manifest.json"
    _write_json(
        launch_path,
        {
            "schema": postcheck.LAUNCH_SCHEMA,
            "phase": "baseline",
            "runtime_image": {"id": postcheck.RUNTIME_IMAGE_ID},
            "plan_dir": str(plan_dir),
            "per_island_rollout_seeds": [postcheck.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            "containers": records,
        },
        private=True,
    )
    monkeypatch.setattr(postcheck, "_docker_inspect", lambda names: inspections)
    monkeypatch.setattr(postcheck, "_load_rollout_payload", lambda path: payloads[path])
    report = postcheck.postcheck(launch_path, plan_dir)
    assert report["ok"] is True
    assert report["logical_trajectories"] == 356
    assert [item["logical_trajectories"] for item in report["islands"]] == [
        45,
        45,
        45,
        45,
        44,
        44,
        44,
        44,
    ]

    inspections[5] = _inspection(5, "5" * 64, exit_code=1)
    with pytest.raises(postcheck.BaselinePostcheckError, match="did not exit cleanly"):
        postcheck.postcheck(launch_path, plan_dir)
