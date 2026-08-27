from __future__ import annotations

from pathlib import Path

import pytest

from tools.probes import run_tbench21_compaction_baseline as baseline


def test_baseline_runner_enables_one_pass_no_replacement(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    args = baseline.ScriptArgs(
        prompt_data=str(tmp_path / "island-0.jsonl"),
        dump_details=str(tmp_path / "details"),
    )
    monkeypatch.setattr(
        baseline,
        "_preflight",
        lambda _args: ([{} for _ in range(45)], {"CODEX_IDENTITY": "pinned"}),
    )
    monkeypatch.setattr(baseline, "_hmac_key_file", lambda: tmp_path / "hmac.key")
    monkeypatch.setattr(
        baseline,
        "_validated_openenv_python",
        lambda _value: tmp_path / "openenv-python",
    )
    monkeypatch.setattr(
        baseline.U,
        "execute_train",
        lambda **kwargs: captured.update(kwargs),
    )

    baseline.execute(args)

    train_args = str(captured["train_args"])
    assert train_args.count("--rollout-one-pass-no-replacement") == 1
    assert "--num-rollout 1 --rollout-batch-size 45" in train_args
    assert train_args.count("--session-server-startup-timeout-secs 120") == 1


def test_eval_runner_loads_and_publishes_checkpoint_without_training(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    args = baseline.ScriptArgs(
        phase="eval",
        serve_ref_checkpoint=True,
        prompt_data=str(tmp_path / "island-0.jsonl"),
        dump_details=str(tmp_path / "details"),
    )
    monkeypatch.setattr(
        baseline,
        "_preflight",
        lambda _args: ([{} for _ in range(23)], {"CODEX_IDENTITY": "pinned"}),
    )
    monkeypatch.setattr(baseline, "_hmac_key_file", lambda: tmp_path / "hmac.key")
    monkeypatch.setattr(
        baseline,
        "_validated_openenv_python",
        lambda _value: tmp_path / "openenv-python",
    )
    monkeypatch.setattr(
        baseline.U,
        "execute_train",
        lambda **kwargs: captured.update(kwargs),
    )

    baseline.execute(args)

    train_args = str(captured["train_args"])
    assert "--rollout-only-from-checkpoint" in train_args
    assert "--train-backend megatron" in train_args
    assert "--dist-ckpt-strictness raise_unexpected" in train_args
    assert "--rollout-weight-version-format counter" in train_args
    assert "--debug-disable-optimizer" in train_args
    assert "--check-weight-update-equal" not in train_args
    assert f"--rollout-only-publication-evidence {tmp_path / 'checkpoint-publication.json'}" in train_args
    assert "--debug-rollout-only" not in train_args
    assert "--colocate" in train_args
    assert "--start-rollout-id 0 --num-rollout 1" in train_args
    assert "--sglang-mem-fraction-static 0.15" in train_args
    assert "--sglang-max-total-tokens 393216" in train_args


def test_eval_checkpoint_preflight_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n")

    with pytest.raises(ValueError, match="iteration is incomplete"):
        baseline._validated_eval_checkpoint(checkpoint)

    iteration = checkpoint / "iter_0000000"
    iteration.mkdir()
    (iteration / ".metadata").write_bytes(b"metadata")
    (iteration / "__0_0.distcp").write_bytes(b"weights")
    assert baseline._validated_eval_checkpoint(checkpoint) == checkpoint.resolve()
