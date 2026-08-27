from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from tools.probes import launch_tbench21_baseline_ramp as host
from tools.probes import run_tbench21_compaction_ramp as island


def _write_json(path: Path, value: object, *, private: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    if private:
        path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _source_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan-v2"
    rows = [[] for _ in range(8)]
    for task_index in range(89):
        for replica in range(4):
            island_id = (task_index * 4 + replica) % 8
            rows[island_id].append(
                {
                    "messages": [{"role": "system", "content": "solve"}],
                    "metadata": {
                        "task_id": f"task-{task_index:02d}",
                        "sample_id": f"baseline:task-{task_index:02d}:r{replica}",
                        "split": "baseline",
                        "island_id": island_id,
                        "rollout_seed": host.wave.ROLLOUT_SEED_BASE + island_id,
                        "episode_timeout_seconds": 1800,
                    },
                }
            )
    files = {}
    for island_id, shard_rows in enumerate(rows):
        relative = f"baseline/island-{island_id}.jsonl"
        files[relative] = _write_json(plan / relative, shard_rows[0])
        encoded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in shard_rows)
        (plan / relative).write_bytes(encoded)
        files[relative] = hashlib.sha256(encoded).hexdigest()
    _write_json(
        plan / "manifest.json",
        {
            "schema": host.PLAN_SCHEMA,
            "topology": {"islands": 8, "one_physical_gpu_per_island": True},
            "rollouts": {
                "baseline": 356,
                "per_task": 4,
                "episode_timeout_seconds": 1800,
                "seed_base": host.wave.ROLLOUT_SEED_BASE,
                "per_island_seeds": [host.wave.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            },
            "files": files,
        },
    )
    return plan


@pytest.mark.parametrize(("gate_size", "per_island"), [(8, 1), (64, 8)])
def test_ramp_plan_is_exact_prefix_without_mutating_source(tmp_path: Path, gate_size: int, per_island: int) -> None:
    source = _source_plan(tmp_path)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    ramp_root = tmp_path / f"ramp-{gate_size}"
    manifest = host._build_ramp_plan(
        source_plan=source,
        ramp_root=ramp_root,
        gate_size=gate_size,
        run_id=f"gate{gate_size}-v1",
    )

    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == before
    assert sum(len(ids) for ids in manifest["selected_sample_ids"].values()) == gate_size
    expected_island3: list[dict] | None = None
    for island_id in range(8):
        ramp_rows = [json.loads(line) for line in (ramp_root / f"baseline/island-{island_id}.jsonl").read_text().splitlines()]
        source_rows = [json.loads(line) for line in (source / f"baseline/island-{island_id}.jsonl").read_text().splitlines()]
        assert ramp_rows == source_rows[:per_island]
        if island_id == 3:
            expected_island3 = source_rows[:per_island]

    args = island.RampArgs(
        island_id=3,
        gate_size=gate_size,
        run_id=f"gate{gate_size}-v1",
        source_plan=str(source),
        ramp_manifest=str(ramp_root / "manifest.json"),
        prompt_data=str(ramp_root / "baseline/island-3.jsonl"),
        concurrency=per_island,
    )
    assert island._load_bound_rows(args) == expected_island3


@pytest.mark.parametrize(("gate_size", "count"), [(8, 1), (64, 8)])
def test_island_launch_is_exact_one_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate_size: int,
    count: int,
) -> None:
    captured: dict[str, object] = {}
    args = island.RampArgs(
        gate_size=gate_size,
        run_id=f"gate{gate_size}-v1",
        concurrency=count,
        prompt_data=str(tmp_path / "ramp.jsonl"),
        dump_details=str(tmp_path / "details"),
    )
    monkeypatch.setattr(
        island,
        "_preflight",
        lambda _args: ([{} for _ in range(count)], {"CODEX_IDENTITY": "pinned"}),
    )
    monkeypatch.setattr(island.baseline, "_hmac_key_file", lambda: tmp_path / "hmac")
    monkeypatch.setattr(
        island.baseline,
        "_validated_openenv_python",
        lambda _value: tmp_path / "openenv-python",
    )
    monkeypatch.setattr(island.U, "execute_train", lambda **kwargs: captured.update(kwargs))

    island.execute(args)
    train_args = str(captured["train_args"])
    assert train_args.count("--rollout-one-pass-no-replacement") == 1
    assert f"--rollout-batch-size {count}" in train_args
    assert f"--global-batch-size {count}" in train_args
    assert f"--async-max-concurrent-samples {count}" in train_args
    assert train_args.count("--session-server-startup-timeout-secs 120") == 1


def test_clean_server_gate_rejects_any_run_scoped_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = {
        "ok": True,
        "run_id": "baseline-v3",
        "active_sessions": 0,
        "managed_containers": 0,
        "orphan_containers": 0,
        "cleanup_blocked": False,
        "failed_cleanup_count": 0,
        "last_janitor_error": None,
    }
    monkeypatch.setattr(host, "_managed_status", lambda *_args: (200, healthy))
    monkeypatch.setattr(host, "_managed_container_ids", lambda _run_id: ["a" * 64])
    with pytest.raises(host.RampError, match="task containers remain"):
        host._clean_server_gate("http://127.0.0.1:8003/miles/managed-status", "baseline-v3")


def test_clean_server_gate_polls_valid_busy_status_without_replaying_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = {
        "ok": True,
        "run_id": "baseline-v3",
        "active_sessions": 1,
        "managed_containers": 1,
        "orphan_containers": 0,
        "cleanup_blocked": False,
        "failed_cleanup_count": 0,
        "last_janitor_error": None,
    }
    idle = {**busy, "active_sessions": 0, "managed_containers": 0}
    responses = iter([(503, busy), (200, idle)])
    calls = []
    monkeypatch.setattr(
        host,
        "_managed_status",
        lambda *_args: calls.append("status") or next(responses),
    )
    monkeypatch.setattr(host, "_managed_container_ids", lambda _run_id: [])
    monkeypatch.setattr(host.time, "sleep", lambda _seconds: None)

    result = host._clean_server_gate(
        "http://127.0.0.1:8003/miles/managed-status",
        "baseline-v3",
    )
    assert result == idle
    assert calls == ["status", "status"]


def test_managed_container_filter_uses_two_complete_label_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class _Completed:
        stdout = ""

    monkeypatch.setattr(
        host,
        "_run",
        lambda argv: captured.extend(argv) or _Completed(),
    )
    assert host._managed_container_ids("baseline-v3") == []
    assert captured == [
        "docker",
        "ps",
        "-aq",
        "--filter",
        "label=miles.tbench21.managed=true",
        "--filter",
        "label=miles.tbench21.run_id=baseline-v3",
    ]


class _OutcomeApi:
    @staticmethod
    def canonical_outcome(outcome: dict) -> bytes:
        return json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def sign_outcome(cls, outcome: dict, *, key: bytes) -> str:
        return hmac.new(key, b"test\0" + cls.canonical_outcome(outcome), hashlib.sha256).hexdigest()


def _payload(row: dict, island_id: int, key: bytes) -> dict:
    metadata = row["metadata"]
    outcome = {
        "schema": 1,
        "benchmark": "terminal-bench-2.1",
        "task_id": metadata["task_id"],
        "sample_id": metadata["sample_id"],
        "episode_id": f"episode-{island_id}",
        "status": "completed",
        "reward": 0.0,
        "passed": False,
        "verifier": "openenv_native_evaluate",
        "testsh_rc": None,
    }
    mac = _OutcomeApi.sign_outcome(outcome, key=key)
    return {
        "rollout_id": 0,
        "metadata": {},
        "samples": [
            {
                "index": 0,
                "group_index": 0,
                "status": "completed",
                "reward": 0.0,
                "metadata": {
                    **metadata,
                    "exit_status": "completed",
                    "reward": 0.0,
                    "compaction_trajectory_id": f"trajectory-{island_id}",
                    "compaction_segment_index": 0,
                    "compaction_segment_type": "execution",
                    "secrlenv_infrastructure_503_retries": 0,
                    host.OUTCOME_KEY: outcome,
                    host.MAC_KEY: mac,
                },
            }
        ],
    }


def test_postcheck_authenticates_exact_gate_and_rejects_forgery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_plan(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    ramp_root = output / "ramp-plan"
    ramp = host._build_ramp_plan(
        source_plan=source,
        ramp_root=ramp_root,
        gate_size=8,
        run_id="gate8-v1",
    )
    yeto = tmp_path / "yeto"
    verifier = yeto / "yeto/rl/tbench_outcome.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("# pinned\n")
    key = b"k" * 48
    key_file = tmp_path / "hmac.key"
    key_file.write_bytes(key)
    key_file.chmod(0o600)

    records = []
    inspections = []
    payloads = {}
    for island_id in range(8):
        island_output = output / f"island-{island_id}"
        pt = island_output / "details/rollout_data/0.pt"
        pt.parent.mkdir(parents=True)
        pt.write_bytes(b"pt")
        row = json.loads((ramp_root / f"baseline/island-{island_id}.jsonl").read_text())
        payloads[pt] = _payload(row, island_id, key)
        container_id = f"{island_id:x}" * 64
        name = host._container_name(8, "gate8-v1", island_id)
        shard = ramp_root / f"baseline/island-{island_id}.jsonl"
        records.append(
            {
                "island_id": island_id,
                "name": name,
                "container_id": container_id,
                "plan_shard": str(shard),
                "plan_shard_sha256": host._sha256(shard),
                "selected_sample_ids": ramp["selected_sample_ids"][str(island_id)],
                "output": str(island_output),
            }
        )
        inspections.append(
            {
                "Id": container_id,
                "Name": f"/{name}",
                "Image": host.wave.RUNTIME_IMAGE_ID,
                "State": {
                    "Status": "exited",
                    "Running": False,
                    "Dead": False,
                    "OOMKilled": False,
                    "ExitCode": 0,
                    "Error": "",
                },
                "Config": {
                    "Labels": {
                        "yeto.run": "tbench21-sao-ramp",
                        "yeto.ramp.run_id": "gate8-v1",
                        "yeto.ramp.gate_size": "8",
                        "yeto.island": str(island_id),
                    }
                },
            }
        )
    launch_path = output / "launch-manifest.json"
    _write_json(
        launch_path,
        {
            "schema": host.RAMP_LAUNCH_SCHEMA,
            "gate_size": 8,
            "per_island": 1,
            "run_id": "gate8-v1",
            "server_run_id": "baseline-v3",
            "server_status_url": "http://127.0.0.1:8003/miles/managed-status",
            "runtime_image": {"id": host.wave.RUNTIME_IMAGE_ID},
            "source_plan": str(source),
            "source_plan_manifest_sha256": host._sha256(source / "manifest.json"),
            "ramp_plan": str(ramp_root),
            "ramp_plan_manifest_sha256": host._sha256(ramp_root / "manifest.json"),
            "outcome_verifier_sha256": host._sha256(verifier),
            "per_island_rollout_seeds": [host.wave.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            "containers": records,
        },
        private=True,
    )
    clean = {
        "ok": True,
        "run_id": "baseline-v3",
        "active_sessions": 0,
        "managed_containers": 0,
        "orphan_containers": 0,
        "cleanup_blocked": False,
        "failed_cleanup_count": 0,
        "last_janitor_error": None,
    }
    monkeypatch.setattr(host, "_docker_inspect", lambda _names: inspections)
    monkeypatch.setattr(host, "_load_rollout_payload", lambda path: payloads[path])
    monkeypatch.setattr(host, "_load_outcome_api", lambda *_args: _OutcomeApi)
    monkeypatch.setattr(host, "_clean_server_gate", lambda *_args: clean)

    report = host.postcheck(
        launch_manifest_path=launch_path,
        yeto_root=yeto,
        hmac_key=key_file,
    )
    assert report["ok"] is True
    assert report["authenticated_outcomes"] == 8
    assert report["logical_trajectories"] == 8
    assert report["task_container_leaks"] == 0

    payloads[next(iter(payloads))]["samples"][0]["metadata"][host.MAC_KEY] = "0" * 64
    with pytest.raises(host.RampError, match="forged or off-plan"):
        host.postcheck(
            launch_manifest_path=launch_path,
            yeto_root=yeto,
            hmac_key=key_file,
        )
