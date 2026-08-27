"""Offline tests for the TB2.1 shared-server preflight (no network or GPU)."""

import asyncio
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import preflight_tbench21_shared_server as preflight  # noqa: E402


class _Action:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


class _Env:
    last_instance = None

    def __init__(self, base_url, message_timeout_s):
        self.base_url = base_url
        self.message_timeout_s = message_timeout_s
        self.task_id = None
        self.actions = []
        _Env.last_instance = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def reset(self, task_id):
        self.task_id = task_id
        return SimpleNamespace(observation=SimpleNamespace(error=""))

    async def step(self, action):
        self.actions.append(action)
        if action.action_type == "exec":
            if action.command == preflight.OUTPUT_BOUNDARY_COMMAND:
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        output="x" * preflight.EXEC_OUTPUT_BOUNDARY_BYTES,
                        error="",
                        info={"truncated": True},
                    )
                )
            return SimpleNamespace(
                observation=SimpleNamespace(output="/app/personal-site\n", error="")
            )
        return SimpleNamespace(reward=0.0, observation=SimpleNamespace(error=""))


def _managed_status(**overrides):
    payload = {
        "ok": True,
        "run_id": "baseline-20260826",
        "active_sessions": 0,
        "managed_containers": 0,
        "orphan_containers": 0,
        "cleanup_blocked": False,
        "failed_cleanup_count": 0,
    }
    payload.update(overrides)
    return payload


def test_probe_checks_pwd_server_side_output_boundary_and_native_evaluate():
    result = asyncio.run(
        preflight.probe(
            env_cls=_Env,
            action_cls=_Action,
            url="http://env:8003",
            task="fix-git",
            expected_workdir="/app/personal-site",
            timeout_s=12,
        )
    )

    env = _Env.last_instance
    assert result["ok"] is True
    assert result["reward"] == 0.0
    assert result["bounded_exec_output_bytes"] == preflight.EXEC_OUTPUT_BOUNDARY_BYTES
    assert env.task_id == "fix-git"
    assert [(action.action_type, action.command) for action in env.actions] == [
        ("exec", "pwd"),
        ("exec", preflight.OUTPUT_BOUNDARY_COMMAND),
        ("evaluate", None),
    ]


@pytest.mark.parametrize(
    ("output", "info"),
    [
        ("x" * (preflight.EXEC_OUTPUT_BOUNDARY_BYTES + 1), {"truncated": True}),
        ("x" * preflight.EXEC_OUTPUT_BOUNDARY_BYTES, {"truncated": False}),
        ("x" * preflight.EXEC_OUTPUT_BOUNDARY_BYTES, {"truncated": "yes"}),
    ],
)
def test_probe_rejects_an_unbounded_or_unsigned_exec_observation(output, info):
    class BadBoundaryEnv(_Env):
        async def step(self, action):
            if action.action_type == "exec" and action.command == preflight.OUTPUT_BOUNDARY_COMMAND:
                self.actions.append(action)
                return SimpleNamespace(
                    observation=SimpleNamespace(output=output, error="", info=info)
                )
            return await super().step(action)

    with pytest.raises(preflight.PreflightError, match="output boundary"):
        asyncio.run(
            preflight.probe(
                env_cls=BadBoundaryEnv,
                action_cls=_Action,
                url="http://env:8003",
                task="fix-git",
                expected_workdir="/app/personal-site",
                timeout_s=12,
            )
        )


def test_managed_status_requires_exact_run_and_zero_residue():
    payload = _managed_status()

    assert preflight.require_idle_managed_status(
        payload, expected_run_id="baseline-20260826"
    ) == payload

    for field, value in {
        "run_id": "wrong-run",
        "active_sessions": 1,
        "managed_containers": 1,
        "orphan_containers": 1,
        "cleanup_blocked": True,
        "failed_cleanup_count": 1,
        "ok": False,
    }.items():
        dirty = dict(payload)
        dirty[field] = value
        with pytest.raises(preflight.PreflightError, match="run scope"):
            preflight.require_idle_managed_status(
                dirty, expected_run_id="baseline-20260826"
            )


def test_fetch_managed_status_preserves_structured_http_503(monkeypatch):
    payload = _managed_status(
        ok=False,
        active_sessions=0,
        managed_containers=1,
        orphan_containers=1,
    )
    body = json.dumps(payload).encode()

    def raise_http_503(_request, timeout):
        assert timeout == 4.0
        raise urllib.error.HTTPError(
            "http://env:8003/miles/managed-status",
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr(preflight.urllib.request, "urlopen", raise_http_503)

    assert preflight._fetch_managed_status("http://env:8003", 4.0) == (503, payload)


def test_wait_for_idle_status_polls_transient_busy_503_then_clean():
    responses = iter(
        [
            (
                503,
                _managed_status(
                    ok=False,
                    managed_containers=1,
                    orphan_containers=1,
                ),
            ),
            (200, _managed_status()),
        ]
    )
    now = 10.0
    calls = []

    def fetch(url, timeout_s):
        calls.append((url, timeout_s))
        return next(responses)

    def monotonic():
        return now

    def sleep(delay_s):
        nonlocal now
        now += delay_s

    result = preflight.wait_for_idle_managed_status(
        url="http://env:8003",
        expected_run_id="baseline-20260826",
        timeout_s=2.0,
        poll_interval_s=0.25,
        fetch_status=fetch,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert result == _managed_status()
    assert calls == [
        ("http://env:8003", 2.0),
        ("http://env:8003", 1.75),
    ]


def test_wait_for_idle_status_times_out_without_weakening_final_gate():
    busy = _managed_status(
        ok=False,
        managed_containers=1,
        orphan_containers=1,
    )
    now = 4.0
    calls = 0

    def fetch(_url, _timeout_s):
        nonlocal calls
        calls += 1
        return 503, busy

    def monotonic():
        return now

    def sleep(delay_s):
        nonlocal now
        now += delay_s

    with pytest.raises(preflight.PreflightError, match="did not become clean"):
        preflight.wait_for_idle_managed_status(
            url="http://env:8003",
            expected_run_id="baseline-20260826",
            timeout_s=0.5,
            poll_interval_s=0.2,
            fetch_status=fetch,
            monotonic=monotonic,
            sleep=sleep,
        )

    assert calls == 3


def test_wait_for_idle_status_does_not_retry_transport_failure():
    calls = 0

    def unavailable(_url, _timeout_s):
        nonlocal calls
        calls += 1
        raise preflight.PreflightError(
            "managed server status endpoint is unavailable"
        )

    with pytest.raises(preflight.PreflightError, match="endpoint is unavailable"):
        preflight.wait_for_idle_managed_status(
            url="http://env:8003",
            expected_run_id="baseline-20260826",
            timeout_s=10.0,
            fetch_status=unavailable,
        )

    assert calls == 1


@pytest.mark.parametrize("reward", [None, "1", float("nan"), 0.5, 2.0, True])
def test_probe_rejects_non_numeric_non_finite_or_non_binary_reward(reward):
    class BadRewardEnv(_Env):
        async def step(self, action):
            if action.action_type == "exec":
                return await super().step(action)
            return SimpleNamespace(reward=reward, observation=SimpleNamespace(error=""))

    with pytest.raises(preflight.PreflightError):
        asyncio.run(
            preflight.probe(
                env_cls=BadRewardEnv,
                action_cls=_Action,
                url="http://env:8003",
                task="fix-git",
                expected_workdir="/app/personal-site",
                timeout_s=12,
            )
        )
