from __future__ import annotations

import asyncio
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_OPENENV_DIR = Path(__file__).resolve().parent.parent
_FUZZLAND = Path(__file__).resolve().parents[5]
_YETO = _FUZZLAND / "yeto-grpo-diloco-connector"
sys.path[:0] = [str(_OPENENV_DIR), str(_YETO)]

import codex_openenv_agent_function as agent  # noqa: E402
from yeto.rl.tbench_outcome import verified_outcome  # noqa: E402


@pytest.fixture(autouse=True)
def _terminal_bench_hmac_key(monkeypatch):
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", "test-only-terminal-bench-key-32-bytes")


@dataclass
class _Action:
    action_type: str
    command: str | None = None
    wait_seconds: float | None = None


@dataclass
class _Observation:
    instruction: str = ""
    output: str = ""
    error: str = ""
    exit_code: int | None = 0
    info: dict[str, Any] | None = None


@dataclass
class _Result:
    observation: _Observation
    reward: float | None = None


class _Env:
    def __init__(self) -> None:
        self.actions: list[_Action] = []

    async def reset(self, *, task_id: str) -> _Result:
        assert task_id == "fix-git"
        return _Result(_Observation(instruction="Fix the repository and verify it."))

    async def step(self, action: _Action) -> _Result:
        self.actions.append(action)
        if action.action_type == "evaluate":
            return _Result(_Observation(), reward=1.0)
        return _Result(_Observation(output="ok", exit_code=0))


@pytest.mark.asyncio
async def test_openenv_client_maps_terminal_submit_and_canonical_evaluate():
    env = _Env()
    client = agent._OpenEnvEpisodeClient(env, _Action, native_evaluate=True)

    executed = await client.execute(
        "episode",
        "pwd",
        timeout_seconds=5,
        output_bytes=agent.stock.MAX_TOOL_OUTPUT_BYTES,
    )
    submitted = await client.submit("episode", {"evidence": "tests pass"})
    evaluated = await client.evaluate("episode")

    assert env.actions == [_Action("exec", "pwd", 5), _Action("evaluate")]
    assert executed == {
        "exit_code": 0,
        "output": "ok",
        "timed_out": False,
        "truncated": False,
    }
    assert submitted == {"accepted": True}
    assert evaluated == {"reward": 1.0, "testsh_rc": None}
    assert client.submission == {"evidence": "tests pass"}


@pytest.mark.asyncio
async def test_openenv_client_uses_managed_timeout_status_from_observation_info():
    env = _Env()

    async def timed_out(action: _Action) -> _Result:
        env.actions.append(action)
        return _Result(
            _Observation(
                output="command timed out",
                exit_code=None,
                info={"exit_code": 124, "timed_out": True},
            )
        )

    env.step = timed_out  # type: ignore[method-assign]
    client = agent._OpenEnvEpisodeClient(env, _Action, native_evaluate=True)

    result = await client.execute(
        "episode",
        "long-command",
        timeout_seconds=7,
        output_bytes=agent.stock.MAX_TOOL_OUTPUT_BYTES,
    )

    assert env.actions == [_Action("exec", "long-command", 7)]
    assert result["exit_code"] == 124
    assert result["timed_out"] is True


@pytest.mark.asyncio
async def test_openenv_client_bounds_utf8_output_exactly():
    env = _Env()

    async def oversized(_action: _Action) -> _Result:
        return _Result(_Observation(output="🙂" * agent.stock.MAX_TOOL_OUTPUT_BYTES))

    env.step = oversized  # type: ignore[method-assign]
    client = agent._OpenEnvEpisodeClient(env, _Action, native_evaluate=True)
    result = await client.execute(
        "episode",
        "generate-output",
        timeout_seconds=5,
        output_bytes=agent.stock.MAX_TOOL_OUTPUT_BYTES,
    )
    assert result["truncated"] is True
    assert len(result["output"].encode("utf-8")) <= agent.stock.MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_full_adapter_uses_reset_instruction_and_returns_verifier_reward(
    monkeypatch,
):
    env = _Env()
    seen: dict[str, Any] = {}

    monkeypatch.setattr(agent, "_attest_and_install_openenv_surface", lambda: Path("/codex"))
    monkeypatch.setattr(
        agent.openenv,
        "_load_tbench2",
        lambda: {"env": object, "action": _Action},
    )
    monkeypatch.setattr(agent.openenv, "_shared_native_evaluate_enabled", lambda: True)

    async def shared(_env_cls, _metadata, body):
        return await body(env)

    async def purge(_env, _action_cls):
        return None

    async def drive(
        binary,
        base_url,
        client,
        episode,
        request_kwargs,
        metrics,
        *,
        max_seq_len,
    ):
        seen.update(
            binary=binary,
            base_url=base_url,
            episode=copy.deepcopy(episode),
            request_kwargs=copy.deepcopy(request_kwargs),
            max_seq_len=max_seq_len,
        )
        await client.execute(
            episode["episode_id"],
            "pwd",
            timeout_seconds=5,
            output_bytes=agent.stock.MAX_TOOL_OUTPUT_BYTES,
        )
        await client.submit(episode["episode_id"], {"evidence": "ok"})
        metrics.turns = 2
        metrics.terminal_calls = 1
        metrics.submit_calls = 1
        return "completed"

    monkeypatch.setattr(agent.openenv, "_shared_run_body", shared)
    monkeypatch.setattr(agent.openenv, "_purge_trial_dirs", purge)
    monkeypatch.setattr(agent.stock, "_drive_codex", drive)

    result = await agent.run(
        "http://miles-session",
        [{"role": "user", "content": "dataset prompt is not authoritative"}],
        {"temperature": 0.7},
        {
            "task_id": "fix-git",
            "sample_id": "train:fix-git:r0",
            "max_seq_len": 8192,
        },
    )

    assert result is not None
    assert result["reward"] == 1.0
    assert result["exit_status"] == "completed"
    assert result["agent_metrics"]["turns"] == 2
    assert result["agent_metrics"]["terminal_submission"] == {"evidence": "ok"}
    outcome, signed_reward = verified_outcome(result)
    assert signed_reward == 1.0
    assert outcome["sample_id"] == "train:fix-git:r0"
    assert outcome["verifier"] == agent.NATIVE_VERIFIER
    assert seen["episode"]["prompt"] == "Fix the repository and verify it."
    assert seen["max_seq_len"] == 8192


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_phase",
    ["admission", "reset", "drive", "evaluate", "purge"],
)
async def test_one_deadline_covers_the_complete_environment_lifecycle(
    monkeypatch,
    blocked_phase,
):
    env = _Env()
    never = asyncio.Event()
    purge_started = asyncio.Event()

    monkeypatch.setenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "0.02")
    monkeypatch.setattr(
        agent, "_attest_and_install_openenv_surface", lambda: Path("/codex")
    )
    monkeypatch.setattr(
        agent.openenv,
        "_load_tbench2",
        lambda: {"env": object, "action": _Action},
    )
    monkeypatch.setattr(
        agent.openenv, "_shared_native_evaluate_enabled", lambda: True
    )

    original_reset = env.reset
    original_step = env.step

    async def reset(*, task_id):
        if blocked_phase == "reset":
            await never.wait()
        return await original_reset(task_id=task_id)

    async def step(action):
        if blocked_phase == "evaluate" and action.action_type == "evaluate":
            await never.wait()
        return await original_step(action)

    env.reset = reset  # type: ignore[method-assign]
    env.step = step  # type: ignore[method-assign]

    async def shared(_env_cls, _metadata, body):
        if blocked_phase == "admission":
            await never.wait()
        return await body(env)

    async def drive(*_args, **_kwargs):
        if blocked_phase == "drive":
            await never.wait()
        return "completed"

    async def purge(_env, _action_cls):
        purge_started.set()
        if blocked_phase == "purge":
            await never.wait()

    monkeypatch.setattr(agent.openenv, "_shared_run_body", shared)
    monkeypatch.setattr(agent.openenv, "_purge_trial_dirs", purge)
    monkeypatch.setattr(agent.stock, "_drive_codex", drive)

    result = await agent.run(
        "http://miles-session",
        [],
        {},
        {"task_id": "fix-git", "sample_id": "train:fix-git:r0", "max_seq_len": 8192},
    )

    assert result is not None
    assert result["reward"] == 0.0
    assert result["exit_status"] == "timeout"
    assert result["agent_metrics"]["timed_out"] == 1
    outcome, signed_reward = verified_outcome(result)
    assert signed_reward == 0.0
    assert outcome["status"] == "timeout"
    assert outcome["verifier"] == agent.TIMEOUT_VERIFIER
    assert purge_started.is_set() is (blocked_phase != "admission")


@pytest.mark.asyncio
async def test_external_cancellation_still_runs_environment_purge(monkeypatch):
    env = _Env()
    drive_started = asyncio.Event()
    purged = asyncio.Event()
    never = asyncio.Event()

    monkeypatch.setenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "60")
    monkeypatch.setattr(
        agent, "_attest_and_install_openenv_surface", lambda: Path("/codex")
    )
    monkeypatch.setattr(
        agent.openenv,
        "_load_tbench2",
        lambda: {"env": object, "action": _Action},
    )
    monkeypatch.setattr(
        agent.openenv, "_shared_native_evaluate_enabled", lambda: True
    )

    async def shared(_env_cls, _metadata, body):
        return await body(env)

    async def drive(*_args, **_kwargs):
        drive_started.set()
        await never.wait()

    async def purge(_env, _action_cls):
        purged.set()

    monkeypatch.setattr(agent.openenv, "_shared_run_body", shared)
    monkeypatch.setattr(agent.openenv, "_purge_trial_dirs", purge)
    monkeypatch.setattr(agent.stock, "_drive_codex", drive)

    task = asyncio.create_task(
        agent.run(
            "http://miles-session",
            [],
            {},
            {
                "task_id": "fix-git",
                "sample_id": "train:fix-git:r0",
                "max_seq_len": 8192,
            },
        )
    )
    await drive_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert purged.is_set()


def test_openenv_surface_attestation_fails_closed_before_rebinding(monkeypatch):
    monkeypatch.setattr(agent.stock, "_attest_runtime", lambda: Path("/codex"))
    for name in agent._OPENENV_IDENTITY_ENV:
        monkeypatch.delenv(name, raising=False)
    original = agent.stock.BASE_INSTRUCTIONS
    with pytest.raises(agent.stock.CodexHarnessError, match="does not match"):
        agent._attest_and_install_openenv_surface()
    assert agent.stock.BASE_INSTRUCTIONS == original


def test_openenv_surface_installs_only_on_exact_qwen08_identity(monkeypatch):
    monkeypatch.setattr(agent.stock, "_attest_runtime", lambda: Path("/codex"))
    monkeypatch.setattr(
        agent.stock,
        "_BACKEND_PROFILE",
        {
            "model_identifier": agent.QWEN35_08B_MODEL,
            "model_revision": agent.QWEN35_08B_REVISION,
            "tito_model": "qwen35",
        },
    )
    monkeypatch.setattr(agent.stock, "BACKEND_MODEL", "qwen35")
    for name, value in agent._OPENENV_IDENTITY_ENV.items():
        monkeypatch.setenv(name, value)
    names = (
        "BASE_INSTRUCTIONS",
        "TERMINAL_EXEC_TOOL",
        "SUBMIT_TOOL",
        "DYNAMIC_TOOLS",
        "_MILES_TOOLS",
        "_EXPECTED_RESPONSE_TOOLS",
    )
    originals = {name: getattr(agent.stock, name) for name in names}
    try:
        assert agent._attest_and_install_openenv_surface() == Path("/codex")
        assert agent.stock.BASE_INSTRUCTIONS == agent.BASE_INSTRUCTIONS
        assert agent.stock.DYNAMIC_TOOLS == agent.DYNAMIC_TOOLS
        assert [tool["function"]["name"] for tool in agent.stock._MILES_TOOLS] == [
            "terminal.exec",
            "submit",
        ]
    finally:
        for name, value in originals.items():
            setattr(agent.stock, name, value)


@pytest.mark.asyncio
async def test_agent_failure_propagates_instead_of_returning_null(monkeypatch):
    assert agent.run._miles_fail_closed_result is True

    def fail_attestation():
        raise RuntimeError("attestation failed")

    monkeypatch.setattr(agent, "_attest_and_install_openenv_surface", fail_attestation)

    with pytest.raises(RuntimeError, match="attestation failed"):
        await agent.run(
            "http://miles-session",
            [],
            {},
            {
                "task_id": "fix-git",
                "sample_id": "train:fix-git:r0",
                "max_seq_len": 8192,
            },
        )
