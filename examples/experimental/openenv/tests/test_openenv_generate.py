from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TEST_PATH = Path(__file__).resolve()
sys.path.insert(0, str(_TEST_PATH.parents[4]))
sys.path.insert(0, str(_TEST_PATH.parent.parent))
sys.path.insert(0, str(_TEST_PATH.parents[5] / "yeto-grpo-diloco-connector"))
import openenv_generate as generate  # noqa: E402

from miles.utils.types import Sample  # noqa: E402
from yeto.rl import tbench_outcome  # noqa: E402


def _sample(
    *,
    reward=1.0,
    exit_status="completed",
    aborted=False,
    segment_index=None,
    trajectory_id=None,
):
    verifier = (
        tbench_outcome.TIMEOUT_VERIFIER
        if exit_status == "timeout"
        else tbench_outcome.TEST_SH_VERIFIER
    )
    signed = tbench_outcome.build_signed_metadata(
        task_id="configure-git-webserver",
        sample_id="train:configure-git-webserver:r0",
        episode_id="openenv-0123456789abcdef",
        status=exit_status,
        reward=reward,
        verifier=verifier,
        testsh_rc=None if exit_status == "timeout" else 0,
    )
    metadata = {"reward": reward, "exit_status": exit_status, **signed}
    if segment_index is not None:
        metadata.update(
            {
                "compaction_schema_version": 1,
                "compaction_trajectory_id": trajectory_id,
                "compaction_segment_index": segment_index,
                "compaction_segment_type": (
                    "execution" if segment_index % 2 == 0 else "summary"
                ),
            }
        )
    sample = SimpleNamespace(
        metadata=metadata,
        status=Sample.Status.ABORTED if aborted else Sample.Status.COMPLETED,
    )
    return sample


def test_reward_requires_authenticated_binary_verdict(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    assert asyncio.run(generate.reward_func(None, _sample(reward=0.0))) == 0.0
    with pytest.raises(ValueError, match="explicit numeric"):
        sample = _sample()
        sample.metadata.pop("reward")
        asyncio.run(generate.reward_func(None, sample))
    sample = _sample()
    sample.metadata["reward"] = 0.0
    with pytest.raises(ValueError, match="differs from its signed"):
        asyncio.run(generate.reward_func(None, sample))


def test_filter_rejects_missing_verdict_and_aborted_episode(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    assert generate.check_terminal_bench_episode(None, [_sample()]).keep
    missing = _sample()
    missing.metadata.pop(tbench_outcome.MAC_KEY)
    assert not generate.check_terminal_bench_episode(None, [missing]).keep
    assert not generate.check_terminal_bench_episode(None, [_sample(aborted=True)]).keep


def test_filter_authenticates_nested_compaction_verdicts(monkeypatch):
    monkeypatch.setenv(tbench_outcome.HMAC_ENV, "t" * 48)
    segments = [
        _sample(segment_index=index, trajectory_id="trajectory-1")
        for index in range(3)
    ]
    assert generate.check_terminal_bench_episode(None, [segments]).keep

    segments[1].metadata[tbench_outcome.OUTCOME_KEY] = {
        **segments[1].metadata[tbench_outcome.OUTCOME_KEY],
        "sample_id": "train:configure-git-webserver:r1",
    }
    segments[1].metadata.update(
        tbench_outcome.build_signed_metadata(
            task_id="configure-git-webserver",
            sample_id="train:configure-git-webserver:r1",
            episode_id="openenv-0123456789abcdef",
            status="completed",
            reward=1.0,
            verifier=tbench_outcome.TEST_SH_VERIFIER,
            testsh_rc=0,
        )
    )
    result = generate.check_terminal_bench_episode(None, [segments])
    assert not result.keep
    assert result.reason == "terminal_bench_compaction_verdict_mismatch"
