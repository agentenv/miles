from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.probes import validate_sao_tbench21_e2e as validator


def _sample(rollout_id: int) -> dict[str, object]:
    response_length = 3
    reward = float(rollout_id % 2)
    return {
        "status": "completed",
        "remove_sample": False,
        "tokens": [9, 8, 7, 6],
        "response_length": response_length,
        "loss_mask": [1, 0, 1],
        "rollout_log_probs": [-0.1, -0.2, -0.3],
        "weight_versions": [str(rollout_id + 1), str(rollout_id + 1)],
        "reward": reward,
        "metadata": {"reward": reward, "exit_status": "completed"},
    }


def _write_event(path: Path, *, role: str, rollout_id: int, update_index: int) -> None:
    if role == "actor":
        metrics = {
            "train/loss": -0.1,
            "train/grad_norm": 2.0,
            "train/step": rollout_id,
        }
    else:
        metrics = {
            "train/critic-value_loss": 0.5 + update_index,
            "train/critic-grad_norm": 3.0,
            "train/step": rollout_id,
        }
    event = {
        "timestamp": "2026-08-26T00:00:00Z",
        "source": {"component": role, "cell_index": 0, "rank_within_cell": 0},
        "type": "metric",
        "rollout_id": rollout_id,
        "attempt": 0,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event) + "\n")


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    details = tmp_path / "details"
    rollout_dir = details / "rollout_data"
    events_dir = details / "events"
    rollout_dir.mkdir(parents=True)
    events_dir.mkdir()
    for rollout_id in validator.EXPECTED_ROLLOUTS:
        torch.save(
            {"rollout_id": rollout_id, "metadata": {}, "samples": [_sample(rollout_id)]},
            rollout_dir / f"{rollout_id}.pt",
        )
        _write_event(events_dir / "actor_cell0_rank0.jsonl", role="actor", rollout_id=rollout_id, update_index=0)
        for update_index in range(validator.MIN_CRITIC_UPDATES_PER_ROLLOUT):
            _write_event(
                events_dir / "critic_cell0_rank0.jsonl",
                role="critic",
                rollout_id=rollout_id,
                update_index=update_index,
            )
    actor = tmp_path / "actor"
    critic = tmp_path / "critic"
    hf = tmp_path / "hf"
    for root in (actor, critic, hf):
        root.mkdir()
    for root in (actor, critic):
        (root / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    return details, actor, critic, hf


def _validate(tmp_path: Path, *, logs: list[Path] | None = None):
    details, actor, critic, hf = _evidence(tmp_path)
    return validator.validate(
        details=details,
        actor=actor,
        critic=critic,
        hf=hf,
        logs=logs,
        checkpoint_verifier=lambda *_args: {"actor_tensors": 200},
    )


def test_accepts_complete_two_rollout_gate(tmp_path: Path):
    report = _validate(tmp_path)

    assert report["rollouts"]["0"]["weight_version"] == "1"
    assert report["rollouts"]["1"]["weight_version"] == "2"
    assert report["updates"] == {
        "actor": {"0": 1, "1": 1},
        "critic": {"0": 2, "1": 2},
    }
    assert report["checkpoint_markers"] == {"actor": "1", "critic": "1"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sample: sample["metadata"].pop("reward"), "metadata.reward"),
        (lambda sample: sample["metadata"].update(exit_status="timeout"), "completed canonical verdict"),
        (lambda sample: sample.update(loss_mask=[1, 1]), "loss mask is misaligned"),
        (lambda sample: sample["rollout_log_probs"].__setitem__(0, float("nan")), "logprobs are non-finite"),
        (lambda sample: sample.update(weight_versions=["1", "2"]), "mixes weight versions"),
    ],
)
def test_rejects_untrainable_or_unverifiable_rollout(tmp_path: Path, mutation, message: str):
    details, actor, critic, hf = _evidence(tmp_path)
    path = details / "rollout_data" / "0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutation(payload["samples"][0])
    torch.save(payload, path)

    with pytest.raises(validator.EvidenceError, match=message):
        validator.validate(
            details=details,
            actor=actor,
            critic=critic,
            hf=hf,
            checkpoint_verifier=lambda *_args: {},
        )


def test_rejects_nonfinite_update_and_wrong_checkpoint_marker(tmp_path: Path):
    details, actor, critic, hf = _evidence(tmp_path)
    actor_events = details / "events" / "actor_cell0_rank0.jsonl"
    rows = [json.loads(line) for line in actor_events.read_text(encoding="utf-8").splitlines()]
    rows[1]["metrics"]["train/grad_norm"] = float("nan")
    actor_events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(validator.EvidenceError, match="non-finite train/grad_norm"):
        validator.validate(
            details=details,
            actor=actor,
            critic=critic,
            hf=hf,
            checkpoint_verifier=lambda *_args: {},
        )

    rows[1]["metrics"]["train/grad_norm"] = 2.0
    actor_events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (critic / "latest_checkpointed_iteration.txt").write_text("0\n", encoding="utf-8")
    with pytest.raises(validator.EvidenceError, match="critic checkpoint marker"):
        validator.validate(
            details=details,
            actor=actor,
            critic=critic,
            hf=hf,
            checkpoint_verifier=lambda *_args: {},
        )


def test_optional_console_log_scan_rejects_failure_signature(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("ray job running\nTraceback (most recent call last):\n", encoding="utf-8")

    with pytest.raises(validator.EvidenceError, match="python_traceback"):
        _validate(tmp_path, logs=[log])
