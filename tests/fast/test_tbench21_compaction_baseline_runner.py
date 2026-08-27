from __future__ import annotations

from pathlib import Path

from tools.probes import run_tbench21_compaction_baseline as baseline


def test_baseline_runner_enables_one_pass_no_replacement(
    monkeypatch, tmp_path: Path
) -> None:
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
