from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from tools.probes import postcheck_tbench21_eval_wave as postcheck


KEY = b"held-out-eval-test-key-that-is-at-least-32-bytes"


def _write_json(path: Path, value: object, *, private: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    if private:
        path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _plan(tmp_path: Path) -> tuple[Path, list[dict[str, postcheck.PlannedSample]]]:
    plan_dir = tmp_path / "plan"
    rows = [[] for _ in range(postcheck.ISLANDS)]
    for task_index in range(postcheck.EXPECTED_TASKS):
        task_id = f"task-{task_index:02d}"
        for replica in range(postcheck.ROLLOUTS_PER_TASK):
            island_id = (task_index * postcheck.ROLLOUTS_PER_TASK + replica) % 8
            rows[island_id].append(
                {
                    "messages": [{"role": "system", "content": "solve"}],
                    "metadata": {
                        "task_id": task_id,
                        "sample_id": f"eval:{task_id}:r{replica}",
                        "split": "eval",
                        "rollout_replica": replica,
                        "island_id": island_id,
                        "rollout_seed": postcheck.ROLLOUT_SEED_BASE + island_id,
                        "episode_timeout_seconds": postcheck.EPISODE_TIMEOUT_SECONDS,
                        "max_seq_len": postcheck.MAX_SEQ_LEN,
                        "prompt_tier": "l2",
                    },
                }
            )
    assert tuple(map(len, rows)) == postcheck.EXPECTED_COUNTS
    files = {}
    for island_id, island_rows in enumerate(rows):
        path = plan_dir / f"eval/island-{island_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in island_rows)
        path.write_bytes(encoded)
        files[f"eval/island-{island_id}.jsonl"] = hashlib.sha256(encoded).hexdigest()
    eval_tasks = [f"task-{index:02d}" for index in range(45)]
    train_tasks = [f"train-{index:02d}" for index in range(44)]
    task_contracts = {task_id: {"task_name": f"terminal-bench/{task_id}"} for task_id in sorted(eval_tasks + train_tasks)}
    split_payload = {
        "seed": "test-split",
        "algorithm": postcheck.PLAN_ALGORITHM,
        "train_task_ids": train_tasks,
        "eval_task_ids": eval_tasks,
    }
    _write_json(
        plan_dir / "manifest.json",
        {
            "schema": postcheck.PLAN_SCHEMA,
            "terminal_bench": {
                "version": "2.1",
                "task_count": 89,
                "task_contracts": task_contracts,
                "task_inventory_sha256": postcheck._canonical_sha256(task_contracts),
            },
            "topology": {"islands": 8, "one_physical_gpu_per_island": True},
            "rollouts": {
                "eval": 180,
                "per_task": 4,
                "episode_timeout_seconds": 1800,
                "seed_base": postcheck.ROLLOUT_SEED_BASE,
                "per_island_seeds": [postcheck.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            },
            "split": {
                **split_payload,
                "sha256": postcheck._canonical_sha256(split_payload),
                "train_task_count": 44,
                "eval_task_count": 45,
            },
            "files": files,
        },
    )
    expected = [
        {
            row["metadata"]["sample_id"]: postcheck.PlannedSample(
                row["metadata"]["task_id"],
                row["metadata"]["rollout_replica"],
                index,
            )
            for index, row in enumerate(island)
        }
        for island in rows
    ]
    return plan_dir, expected


def _signed_outcome(*, task_id: str, sample_id: str, reward: float) -> tuple[dict, str]:
    outcome = {
        "schema": 1,
        "benchmark": "terminal-bench-2.1",
        "task_id": task_id,
        "sample_id": sample_id,
        "episode_id": f"episode-{task_id}-{sample_id.rsplit(':', 1)[-1]}",
        "status": "completed",
        "reward": reward,
        "passed": reward == 1.0,
        "verifier": "openenv_native_evaluate",
        "testsh_rc": None,
    }
    canonical = json.dumps(
        outcome,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    mac = hmac.new(KEY, postcheck.OUTCOME_DOMAIN + canonical, hashlib.sha256).hexdigest()
    return outcome, mac


def _sample(
    *,
    island_id: int,
    sample_id: str,
    contract: postcheck.PlannedSample,
    segment_index: int = 0,
    reward: float = 0.0,
) -> dict:
    outcome, mac = _signed_outcome(
        task_id=contract.task_id,
        sample_id=sample_id,
        reward=reward,
    )
    return {
        "index": contract.index,
        "group_index": contract.index,
        "status": "completed",
        "remove_sample": False,
        "reward": reward,
        "metadata": {
            "task_id": contract.task_id,
            "sample_id": sample_id,
            "split": "eval",
            "rollout_replica": contract.replica,
            "island_id": island_id,
            "rollout_seed": postcheck.ROLLOUT_SEED_BASE + island_id,
            "episode_timeout_seconds": 1800,
            "exit_status": "completed",
            "reward": reward,
            "compaction_trajectory_id": f"trajectory-{island_id}-{contract.index}",
            "compaction_segment_index": segment_index,
            "compaction_segment_type": ("execution" if segment_index % 2 == 0 else "summary"),
            "compaction_schema_version": 1,
            "compaction_context_window": segment_index // 2,
            "compaction_context_budget": postcheck.MAX_SEQ_LEN,
            "secrlenv_infrastructure_503_retries": 0,
            "tito_session_mismatch": [],
            "tbench_trusted_outcome": outcome,
            "tbench_trusted_outcome_hmac": mac,
        },
    }


def _payload(island_id: int, expected: dict[str, postcheck.PlannedSample]) -> dict:
    samples = []
    for sample_id, contract in expected.items():
        reward = float(contract.task_id in {"task-00", "task-01"} and contract.replica == 0)
        samples.append(
            _sample(
                island_id=island_id,
                sample_id=sample_id,
                contract=contract,
                reward=reward,
            )
        )
    first_id = next(iter(expected))
    first = expected[first_id]
    first_reward = float(first.task_id in {"task-00", "task-01"} and first.replica == 0)
    samples.pop(0)
    samples[:0] = [
        _sample(
            island_id=island_id,
            sample_id=first_id,
            contract=first,
            segment_index=segment,
            reward=first_reward,
        )
        for segment in range(3)
    ]
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["metadata"]["sample_id"], []).append(sample)
    for rows in grouped.values():
        terminal = max(
            rows,
            key=lambda sample: sample["metadata"]["compaction_segment_index"],
        )
        terminal["metadata"]["compaction_segment_count"] = len(rows)
        terminal["metadata"]["compaction_context_window_count"] = (len(rows) + 1) // 2
    return {"rollout_id": 0, "metadata": {}, "samples": samples}


def _inspection(
    island_id: int,
    container_id: str,
    *,
    checkpoint: Path,
    plan_dir: Path,
    output: Path,
    key_path: Path,
    model: Path,
    codex: Path,
    exit_code: int = 0,
) -> dict:
    return {
        "Id": container_id,
        "Name": f"/tbench21-eval-island-{island_id}",
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
            "Entrypoint": ["python3"],
            "Cmd": [
                postcheck.RUNNER,
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
                postcheck.CODEX_BINARY,
                "--openenv-agent-python",
                "/root/openenv-venv/bin/python",
                "--openenv-env-url",
                postcheck.MANAGED_CONTAINER_URL,
                "--miles-root",
                "/root/miles",
                "--yeto-root",
                "/root/yeto",
                "--megatron-path",
                "/root/Megatron-LM",
                "--concurrency",
                "38",
                "--rollout-seed",
                str(postcheck.ROLLOUT_SEED_BASE),
            ],
            "Env": [
                f"NVIDIA_VISIBLE_DEVICES={island_id}",
                "TBENCH_REWARD_HMAC_KEY_FILE=/run/secrets/tbench-hmac",
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
            ],
            "Labels": {
                "yeto.run": "tbench21-sao-rollout",
                "yeto.phase": "eval",
                "yeto.island": str(island_id),
            },
        },
        "HostConfig": {"ExtraHosts": ["host.docker.internal:172.17.0.1"]},
        "Mounts": [
            {
                "Source": str(checkpoint),
                "Destination": "/root/input-checkpoint",
                "RW": False,
            },
            {"Source": str(plan_dir), "Destination": "/root/plan", "RW": False},
            {"Source": str(output), "Destination": "/root/run-output", "RW": True},
            {"Source": str(key_path), "Destination": "/run/secrets/tbench-hmac", "RW": False},
            {"Source": str(model), "Destination": "/root/model", "RW": False},
            {"Source": str(codex), "Destination": "/root/codex-cli", "RW": False},
        ],
    }


def _full_fixture(tmp_path: Path, monkeypatch):
    plan_dir, expected = _plan(tmp_path)
    output = tmp_path / "eval"
    output.mkdir()
    online = tmp_path / "online"
    checkpoint = online / "island-7/actor-checkpoint"
    (checkpoint / "iter_0000000").mkdir(parents=True)
    (checkpoint / "iter_0000000/__0_0.distcp").write_bytes(b"final-actor")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n")
    key_path = tmp_path / "reward.key"
    key_path.write_bytes(KEY)
    key_path.chmod(0o600)
    model = tmp_path / "Qwen3.5-0.8B"
    model.mkdir()
    codex = tmp_path / "codex"
    codex_binary = codex / "vendor/x86_64-unknown-linux-musl/bin/codex"
    codex_binary.parent.mkdir(parents=True)
    codex_binary.write_bytes(b"codex")
    records = []
    payloads = {}
    inspections = []
    for island_id in range(8):
        island = output / f"island-{island_id}"
        source = island / "details/rollout_data/0.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"pt-{island_id}".encode())
        payloads[source] = _payload(island_id, expected[island_id])
        shard = plan_dir / f"eval/island-{island_id}.jsonl"
        container_id = f"{island_id:x}" * 64
        records.append(
            {
                "island_id": island_id,
                "name": f"tbench21-eval-island-{island_id}",
                "container_id": container_id,
                "plan_shard": str(shard),
                "plan_shard_sha256": postcheck.baseline._sha256(shard),
                "output": str(island),
            }
        )
        inspections.append(
            _inspection(
                island_id,
                container_id,
                checkpoint=checkpoint,
                plan_dir=plan_dir,
                output=island,
                key_path=key_path,
                model=model,
                codex=codex,
            )
        )
    policy_hash = "a" * 64
    training_records = []
    for island_id in range(8):
        island = online / f"island-{island_id}"
        island.mkdir(parents=True, exist_ok=True)
        (island / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": "rl_sao_streaming_publication",
                    "rollout_id": 1,
                    "terminal": True,
                    "actor_policy_hash": policy_hash,
                    "actor_fragment_versions": [1, 2],
                    "critic_fragment_versions": [1, 2],
                }
            )
            + "\n"
        )
        training_records.append(
            {
                "island_id": island_id,
                "name": f"tbench21-sao-online-island-{island_id}",
                "output": str(island),
            }
        )
    training_launch = online / "launch-manifest.json"
    plan_sha = postcheck.baseline._sha256(plan_dir / "manifest.json")
    _write_json(
        training_launch,
        {
            "schema": "miles.tbench21-sao-online-wave.v1",
            "plan": {"sha256": plan_sha},
            "containers": training_records,
        },
        private=True,
    )
    launch_path = output / "launch-manifest.json"
    _write_json(
        launch_path,
        {
            "schema": postcheck.LAUNCH_SCHEMA,
            "phase": "eval",
            "runtime_image": {"id": postcheck.RUNTIME_IMAGE_ID},
            "checkpoint": str(checkpoint),
            "plan_dir": str(plan_dir),
            "model": str(model),
            "docker_bridge_gateway": "172.17.0.1",
            "codex_binary_sha256": postcheck.baseline._sha256(codex_binary),
            "hmac_key_file_mode": "0o600",
            "per_island_rollout_seeds": [postcheck.ROLLOUT_SEED_BASE + island_id for island_id in range(8)],
            "containers": records,
        },
        private=True,
    )
    monkeypatch.setattr(postcheck.baseline, "_docker_inspect", lambda names: inspections)
    monkeypatch.setattr(postcheck.baseline, "_load_rollout_payload", lambda path: payloads[path])
    managed = {
        "ok": True,
        "run_id": "test-run",
        "active_sessions": 0,
        "managed_containers": 0,
        "orphan_containers": 0,
        "cleanup_blocked": False,
        "failed_cleanup_count": 0,
        "last_janitor_error": None,
    }
    monkeypatch.setattr(postcheck, "_wait_for_managed_clean", lambda **kwargs: managed)
    monkeypatch.setattr(postcheck, "_managed_process", lambda *args, **kwargs: {"pid": 123})
    checkpoint_sha, _ = postcheck._checkpoint_inventory(checkpoint)
    attestations = {
        "expected_plan_manifest_sha256": plan_sha,
        "training_launch_manifest": training_launch,
        "expected_terminal_actor_policy_hash": policy_hash,
        "expected_checkpoint_inventory_sha256": checkpoint_sha,
        "managed_server_pid": 123,
    }
    return launch_path, plan_dir, checkpoint, key_path, inspections, payloads, attestations


def test_full_eval_postcheck_authenticates_and_aggregates(tmp_path, monkeypatch):
    launch, plan, checkpoint, key, _inspections, _payloads, attestations = _full_fixture(tmp_path, monkeypatch)
    report = postcheck.postcheck(
        launch,
        plan,
        checkpoint,
        key,
        **attestations,
        managed_status_url="http://127.0.0.1:8003",
        managed_run_id="test-run",
        managed_clean_timeout_s=1.0,
    )
    assert report["ok"] is True
    assert report["logical_trajectories"] == 180
    assert report["task_count"] == 45
    assert report["signed_outcomes_verified"] == 180
    assert [row["logical_trajectories"] for row in report["islands"]] == list(postcheck.EXPECTED_COUNTS)
    assert report["aggregate"]["passed_rollouts"] == 2
    assert report["aggregate"]["pass_at_1"] == 2 / 180
    assert report["aggregate"]["tasks_with_any_success"] == 2
    assert report["aggregate"]["task_success_rate_at_4"] == 2 / 45
    assert report["aggregate"]["pass_at_4"] == 2 / 45
    assert report["managed_server"]["status"]["managed_containers"] == 0


def test_eval_postcheck_rejects_unsigned_or_tampered_outcome(tmp_path, monkeypatch):
    launch, plan, checkpoint, key, _inspections, payloads, attestations = _full_fixture(tmp_path, monkeypatch)
    first_payload = next(iter(payloads.values()))
    first_payload["samples"][0]["metadata"]["tbench_trusted_outcome_hmac"] = "0" * 64
    with pytest.raises(postcheck.EvalPostcheckError, match="signature mismatch"):
        postcheck.postcheck(
            launch,
            plan,
            checkpoint,
            key,
            **attestations,
            managed_status_url="http://127.0.0.1:8003",
            managed_run_id="test-run",
            managed_clean_timeout_s=1.0,
        )


def test_eval_postcheck_rejects_nonzero_container_and_cleanup_debt(tmp_path, monkeypatch):
    launch, plan, checkpoint, key, inspections, _payloads, attestations = _full_fixture(tmp_path, monkeypatch)
    inspections[5]["State"]["ExitCode"] = 1
    with pytest.raises(postcheck.EvalPostcheckError, match="did not exit cleanly"):
        postcheck.postcheck(
            launch,
            plan,
            checkpoint,
            key,
            **attestations,
            managed_status_url="http://127.0.0.1:8003",
            managed_run_id="test-run",
            managed_clean_timeout_s=1.0,
        )
    inspections[5]["State"]["ExitCode"] = 0
    monkeypatch.setattr(
        postcheck,
        "_wait_for_managed_clean",
        lambda **kwargs: (_ for _ in ()).throw(postcheck.EvalPostcheckError("managed cleanup debt")),
    )
    with pytest.raises(postcheck.EvalPostcheckError, match="managed cleanup debt"):
        postcheck.postcheck(
            launch,
            plan,
            checkpoint,
            key,
            **attestations,
            managed_status_url="http://127.0.0.1:8003",
            managed_run_id="test-run",
            managed_clean_timeout_s=1.0,
        )


def test_eval_postcheck_reports_only_exact_terminal_boundary_mismatch(tmp_path, monkeypatch):
    launch, plan, checkpoint, key, _inspections, payloads, attestations = _full_fixture(tmp_path, monkeypatch)
    payload = next(iter(payloads.values()))
    sample_id = payload["samples"][0]["metadata"]["sample_id"]
    rows = [sample for sample in payload["samples"] if sample["metadata"]["sample_id"] == sample_id]
    for sample in rows:
        metadata = sample["metadata"]
        outcome = dict(metadata["tbench_trusted_outcome"])
        outcome["status"] = "max_seq_len"
        outcome["reward"] = 0.0
        outcome["passed"] = False
        canonical = json.dumps(outcome, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        metadata["exit_status"] = "max_seq_len"
        metadata["reward"] = 0.0
        sample["reward"] = 0.0
        metadata["tbench_trusted_outcome"] = outcome
        metadata["tbench_trusted_outcome_hmac"] = hmac.new(KEY, postcheck.OUTCOME_DOMAIN + canonical, hashlib.sha256).hexdigest()
    terminal = max(rows, key=lambda sample: sample["metadata"]["compaction_segment_index"])
    terminal["metadata"]["tito_session_mismatch"] = [
        {
            "type": "special_token_count",
            "segment_index": -1,
            "expected_text": "[content][<|im_end|>]",
            "actual_text": "[content]",
            "detail": "segment count differs: expected 7, got 6",
        }
    ]
    report = postcheck.postcheck(
        launch,
        plan,
        checkpoint,
        key,
        **attestations,
        managed_status_url=postcheck.MANAGED_STATUS_URL,
        managed_run_id="test-run",
        managed_clean_timeout_s=1.0,
    )
    assert report["terminal_tito_boundary_mismatch_count"] == 1
    assert report["terminal_tito_boundary_mismatch_sample_ids"] == [sample_id]

    terminal["metadata"]["tito_session_mismatch"][0]["type"] = "assistant_text"
    with pytest.raises(postcheck.EvalPostcheckError, match="unapproved TITO mismatch"):
        postcheck.postcheck(
            launch,
            plan,
            checkpoint,
            key,
            **attestations,
            managed_status_url=postcheck.MANAGED_STATUS_URL,
            managed_run_id="test-run",
            managed_clean_timeout_s=1.0,
        )
