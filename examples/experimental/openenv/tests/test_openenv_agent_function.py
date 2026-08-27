"""Offline unit tests for the openenv tbench2 adapter (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers the shared-server leg of the agent loop (this module's run_episode):
its exec form, scoring path, and cleanup. The Daytona-sandbox leg's
dispatch and sandbox-create machinery live in
test_openenv_daytona_agent_function.py; the fakes below are shared with it.
"""

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openenv_agent_function as oaf  # noqa: E402


def run_async(coro):
    return asyncio.run(coro)


# --- fakes ---------------------------------------------------------------


class _FakeObs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResult:
    def __init__(self, output="", reward=None, instruction=""):
        self.observation = _FakeObs(output=output, instruction=instruction)
        if reward is not None:
            self.reward = reward


class _FakeEnv:
    """Records every step() action; answers both scoring protocols."""

    last_actions: list = []
    last_init: dict = {}

    def __init__(
        self,
        base_url="",
        message_timeout_s=0,
        websocket_ping_interval_s=None,
        websocket_ping_timeout_s=None,
    ):
        self.actions = []
        _FakeEnv.last_actions = self.actions
        _FakeEnv.last_init = {
            "base_url": base_url,
            "message_timeout_s": message_timeout_s,
            "websocket_ping_interval_s": websocket_ping_interval_s,
            "websocket_ping_timeout_s": websocket_ping_timeout_s,
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def reset(self, task_id=None):
        return _FakeResult(instruction="do the thing")

    async def state(self):
        return types.SimpleNamespace()

    async def step(self, action):
        self.actions.append(action)
        if action.action_type == "evaluate":
            return _FakeResult(reward=1.0)
        if "test.sh" in (action.command or ""):
            return _FakeResult(output=f"{oaf._REWARD_MARKER}1.0")
        return _FakeResult(output="ok")


class _FakeAction:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


class _FakePolicy:
    """Turn 1: emit a bash command. Turn 2: TASK_COMPLETE."""

    def __init__(self):
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.n += 1
        text = "```bash\necho hi\n```" if self.n == 1 else "TASK_COMPLETE"
        msg = types.SimpleNamespace(content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


_CLASSES = {"env": _FakeEnv, "action": _FakeAction}


class ConnectionClosedError(RuntimeError):
    """Named like the production websocket exception for retry classification."""


def test_with_env_retries_capacity_before_admission_only(monkeypatch):
    attempts = 0
    body_calls = 0

    class CapacityThenReady(_FakeEnv):
        async def state(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("CAPACITY_REACHED")
            return types.SimpleNamespace()

    async def no_sleep(_seconds):
        return None

    async def body(_env):
        nonlocal body_calls
        body_calls += 1
        return "ok"

    monkeypatch.setattr(oaf.asyncio, "sleep", no_sleep)
    assert run_async(oaf._with_env(CapacityThenReady, "http://env", body)) == "ok"
    assert attempts == 2
    assert body_calls == 1
    assert _FakeEnv.last_init == {
        "base_url": "http://env",
        "message_timeout_s": oaf._MESSAGE_TIMEOUT_S,
        "websocket_ping_interval_s": oaf._WEBSOCKET_PING_INTERVAL_S,
        "websocket_ping_timeout_s": oaf._WEBSOCKET_PING_TIMEOUT_S,
    }


def test_with_env_never_replays_after_admission():
    instances = 0
    body_calls = 0

    class Ready(_FakeEnv):
        def __init__(self, *args, **kwargs):
            nonlocal instances
            super().__init__(*args, **kwargs)
            instances += 1

    async def body(_env):
        nonlocal body_calls
        body_calls += 1
        raise ConnectionClosedError("connection dropped after reset")

    try:
        run_async(oaf._with_env(Ready, "http://env", body))
    except ConnectionClosedError as error:
        assert "after reset" in str(error)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("post-admission disconnect did not propagate")

    assert instances == 1
    assert body_calls == 1


# --- episode dispatch ------------------------------------------------------


def test_shared_leg_dispatch(monkeypatch):
    """The shared-server run_episode: exec prefixed with the task workdir,
    canonical-exec scoring, rm-hack present, standard `evaluate` never used."""
    monkeypatch.delenv("OPENENV_NATIVE_EVALUATE", raising=False)
    monkeypatch.setattr(oaf, "_load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"}))
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "cd /app && echo hi"
    assert any("bash /tests/test.sh" in a.command for a in execs)
    assert any("/tmp/tbench2_env_runs" in a.command for a in execs), "rm-hack missing"
    assert not any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1


def test_shared_leg_can_use_native_terminal_bench_verifier(monkeypatch):
    """TB2.1-capable servers own workdir selection and canonical scoring."""
    monkeypatch.setenv("OPENENV_NATIVE_EVALUATE", "1")
    monkeypatch.setattr(oaf, "_load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"}))
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert not any("bash /tests/test.sh" in a.command for a in execs)
    assert any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1
