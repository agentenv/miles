"""Pure lifecycle tests for the Miles-managed TB2.1 server."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import managed_tbench21_server as managed  # noqa: E402


class NotFound(Exception):
    pass


class _Container:
    def __init__(self, container_id: str, labels=None, failures: int = 0):
        self.id = container_id
        self.labels = dict(labels or {})
        self.failures = failures
        self.remove_calls: list[tuple[bool, bool]] = []
        self.removed = False

    def remove(self, *, force: bool, v: bool):
        self.remove_calls.append((force, v))
        if self.removed:
            raise NotFound("already gone")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("daemon busy")
        self.removed = True


class _Containers:
    def __init__(self):
        self.values: list[_Container] = []
        self.run_calls: list[dict] = []
        self.list_calls: list[dict] = []
        self.after_create = None

    def run(self, *args, **kwargs):
        self.run_calls.append({"args": args, **kwargs})
        container = _Container(f"container-{len(self.values)}", kwargs.get("labels"))
        self.values.append(container)
        if self.after_create is not None:
            self.after_create(container)
        return container

    def list(self, *, all: bool, filters: dict, sparse: bool = False):
        assert all is True
        self.list_calls.append(
            {"all": all, "filters": filters, "sparse": sparse}
        )
        required = dict(item.split("=", 1) for item in filters["label"])
        return [
            value
            for value in self.values
            if not value.removed
            and builtins.all(
                value.labels.get(key) == expected
                for key, expected in required.items()
            )
        ]


class _Client:
    def __init__(self):
        self.containers = _Containers()
        self.images = SimpleNamespace()


def _registry(client: _Client, **kwargs) -> managed.ManagedContainerRegistry:
    return managed.ManagedContainerRegistry(
        "run-20260826",
        client_factory=lambda: client,
        retry_delay_s=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "value",
    ["", "UPPER", "space here", "slash/value", "-leading", "trailing-", "x" * 129],
)
def test_run_id_validation_rejects_ambiguous_label_values(value):
    with pytest.raises(ValueError):
        managed.validate_run_id(value)


def test_container_proxy_injects_unoverridable_run_and_owner_labels():
    client = _Client()
    registry = _registry(client)
    proxy = managed._ManagedDockerClientProxy(client, registry, "owner-a")

    container = proxy.containers.run("image", labels={"task": "fix-git"})

    assert container.labels == {
        "task": "fix-git",
        managed.MANAGED_LABEL: "true",
        managed.RUN_ID_LABEL: "run-20260826",
        managed.OWNER_LABEL: "owner-a",
    }
    with pytest.raises(managed.ManagedServerError, match="override"):
        proxy.containers.run(
            "image", labels={managed.RUN_ID_LABEL: "some-other-run"}
        )


def test_cleanup_scope_preserves_unmanaged_and_other_run_containers():
    client = _Client()
    registry = _registry(client)
    exact = _Container("exact", registry.labels)
    unmanaged = _Container("persistent", {managed.RUN_ID_LABEL: registry.run_id})
    other_run = _Container(
        "other-run",
        {managed.MANAGED_LABEL: "true", managed.RUN_ID_LABEL: "another-run"},
    )
    client.containers.values.extend([exact, unmanaged, other_run])

    assert registry.cleanup_all() == {"found": 1, "removed": 1}

    assert exact.removed is True
    assert unmanaged.removed is False
    assert other_run.removed is False


def test_inventory_uses_one_sparse_exact_label_snapshot():
    client = _Client()
    registry = _registry(client)
    exact = _Container("exact", registry.labels)
    client.containers.values.append(exact)

    assert registry.cleanup_all() == {"found": 1, "removed": 1}

    assert client.containers.list_calls == [
        {
            "all": True,
            "filters": {
                "label": [
                    f"{managed.MANAGED_LABEL}=true",
                    f"{managed.RUN_ID_LABEL}=run-20260826",
                ]
            },
            "sparse": True,
        }
    ]


def test_sparse_summary_labels_preserve_pending_owner_classification():
    class _SparseContainer:
        id = "sparse-container"

        def __init__(self, labels):
            self.attrs = {"Id": self.id, "Labels": labels}

        @property
        def labels(self):
            raise AssertionError("sparse Container.labels must not be accessed")

    client = _Client()
    registry = _registry(client)
    owner = "owner-pending"
    sparse = _SparseContainer({**registry.labels, managed.OWNER_LABEL: owner})

    class _SparseContainers:
        def list(self, *, all, filters, sparse):
            assert all is True
            assert sparse is True
            assert filters == registry.label_filters
            return [sparse_container]

    sparse_container = sparse
    client.containers = _SparseContainers()
    registry.reserve(owner)

    status = registry.status(active_sessions=1)

    assert status["managed_containers"] == 1
    assert status["orphan_containers"] == 0
    assert status["ok"] is True


def test_non_not_found_sparse_inventory_error_remains_fail_closed():
    class _BrokenContainers:
        calls = 0

        def list(self, *, all, filters, sparse):
            self.calls += 1
            raise RuntimeError("daemon unavailable")

    client = _Client()
    client.containers = _BrokenContainers()
    registry = _registry(client)

    with pytest.raises(managed.ManagedServerError, match="cannot inventory"):
        registry.status(active_sessions=0)

    assert client.containers.calls == 1
    with pytest.raises(managed.ManagedServerError, match="reset is blocked"):
        registry.assert_reset_allowed()


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {managed.MANAGED_LABEL: "true"},
        {
            managed.MANAGED_LABEL: "true",
            managed.RUN_ID_LABEL: "another-run",
        },
        {
            managed.MANAGED_LABEL: "false",
            managed.RUN_ID_LABEL: "run-20260826",
        },
    ],
)
def test_sparse_inventory_revalidates_exact_run_labels_fail_closed(labels):
    container = SimpleNamespace(id="unexpected", attrs={"Labels": labels})

    class _UnexpectedContainers:
        def list(self, *, all, filters, sparse):
            assert all is True
            assert sparse is True
            return [container]

    client = _Client()
    client.containers = _UnexpectedContainers()
    registry = _registry(client)

    with pytest.raises(managed.ManagedServerError, match="exact managed run scope"):
        registry.status(active_sessions=0)

    with pytest.raises(managed.ManagedServerError, match="reset is blocked"):
        registry.assert_reset_allowed()


def test_force_remove_retries_are_bounded_and_block_future_creation():
    client = _Client()
    registry = _registry(client, remove_attempts=3)
    broken = _Container("broken", registry.labels, failures=10)
    client.containers.values.append(broken)

    with pytest.raises(managed.ManagedServerError, match="after 3 attempts"):
        registry.force_remove(broken)
    assert broken.remove_calls == [(True, True), (True, True), (True, True)]
    with pytest.raises(managed.ManagedServerError, match="reset is blocked"):
        registry.assert_reset_allowed()

    proxy = managed._ManagedDockerClientProxy(client, registry, "owner")
    with pytest.raises(managed.ManagedServerError, match="reset is blocked"):
        proxy.containers.run("image")
    assert client.containers.run_calls == []


def test_not_found_is_successful_cleanup():
    client = _Client()
    registry = _registry(client)
    gone = _Container("gone", registry.labels)
    gone.removed = True

    registry.force_remove(gone)

    assert gone.remove_calls == [(True, True)]
    registry.assert_reset_allowed()


def test_janitor_removes_only_orphans_and_status_reports_counts():
    client = _Client()
    registry = _registry(client)
    active = _Container("active", registry.labels)
    orphan = _Container("orphan", registry.labels)
    client.containers.values.extend([active, orphan])
    registry.claim("owner", active)

    before = registry.status(active_sessions=3)
    result = registry.janitor_once()
    after = registry.status(active_sessions=3)

    assert before["managed_containers"] == 2
    assert before["orphan_containers"] == 1
    assert result == {
        "managed": 2,
        "orphans": 1,
        "attempted": 1,
        "removed": 1,
        "failed": 0,
    }
    assert active.removed is False
    assert orphan.removed is True
    assert after["active_sessions"] == 3
    assert after["managed_containers"] == 1
    assert after["orphan_containers"] == 0


def test_pending_owner_closes_run_return_to_registry_claim_janitor_race():
    client = _Client()
    registry = _registry(client)
    observed = []
    client.containers.after_create = lambda _container: observed.append(
        registry.janitor_once()
    )
    proxy = managed._ManagedDockerClientProxy(client, registry, "owner-pending")

    container = proxy.containers.run("image")

    assert observed == [
        {
            "managed": 1,
            "orphans": 0,
            "attempted": 0,
            "removed": 0,
            "failed": 0,
        }
    ]
    assert container.removed is False
    assert registry.status(active_sessions=1)["orphan_containers"] == 0


def test_environment_close_force_removes_without_stop_and_reset_latches_failure():
    client = _Client()
    registry = _registry(client, remove_attempts=1)

    class _Base:
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self):
            self._container = None
            self._task_dir = "task"
            self._instruction = "instruction"
            self._workdir = "/app"
            self.resets = 0

        def reset(self, *args, **kwargs):
            self.resets += 1
            return kwargs

    environment_cls = managed.managed_environment_class(_Base, registry)
    environment = environment_cls()
    failing = _Container("failing", registry.labels, failures=1)
    environment._container = failing
    registry.claim(environment._miles_owner_id, failing)

    with pytest.raises(managed.ManagedServerError):
        environment.close()
    with pytest.raises(managed.ManagedServerError, match="previously failed"):
        environment.reset(task_id="fix-git")
    assert environment.resets == 0
    assert failing.remove_calls == [(True, True)]

    # Session destruction may retry close; success clears the exact-session latch.
    environment.close()
    assert environment._container is None
    assert environment._task_dir is None


def test_managed_exec_enforces_requested_deadline_and_reports_timeout():
    client = _Client()
    registry = _registry(client)

    class _Base:
        def __init__(self):
            self._container = None
            self.state = SimpleNamespace(last_output="")
            self.commands = []

        def _exec_in_container(self, command, workdir=None):
            self.commands.append((command, workdir))
            return 124, "stopped"

        def step(self, action, *_args, **_kwargs):
            exit_code, output = self._exec_in_container(action.command, "/work")
            return SimpleNamespace(info={}, output=output, exit_code=exit_code)

    action = SimpleNamespace(
        action_type="exec",
        command="printf '%s' \"$value\"",
        wait_seconds=7.5,
    )
    environment = managed.managed_environment_class(_Base, registry)()

    observation = environment.step(action)

    assert observation.info == {"exit_code": 124, "timed_out": True}
    wrapped, workdir = environment.commands[0]
    assert workdir == "/work"
    assert "--signal=TERM --kill-after=5s 7.500s" in wrapped
    assert "bash -c 'printf '" in wrapped
    assert environment._miles_exec_timeout_s is None


@pytest.mark.parametrize("size", [0, managed.MAX_EXEC_OUTPUT_BYTES])
def test_managed_exec_preserves_output_at_or_below_cap(size):
    client = _Client()
    registry = _registry(client)
    raw_output = "x" * size
    original_info = {"source": "upstream", "truncated": True}

    class _Base:
        def __init__(self):
            self._container = None
            self.state = SimpleNamespace(last_output="stale")

        def _exec_in_container(self, _command, _workdir=None):
            return 0, raw_output

        def step(self, action):
            exit_code, output = self._exec_in_container(action.command)
            self.state.last_output = output
            return SimpleNamespace(output=output, info=original_info, exit_code=exit_code)

    environment = managed.managed_environment_class(_Base, registry)()
    observation = environment.step(
        SimpleNamespace(action_type="exec", command="emit", wait_seconds=1)
    )

    assert observation.output == raw_output
    assert environment.state.last_output == raw_output
    assert observation.info == {
        "source": "upstream",
        "truncated": True,
        "exit_code": 0,
        "timed_out": False,
    }
    assert original_info == {"source": "upstream", "truncated": True}


def test_managed_exec_caps_output_and_replaces_raw_state_copy():
    client = _Client()
    registry = _registry(client)
    raw_output = "x" * (managed.MAX_EXEC_OUTPUT_BYTES + 1)

    class _Base:
        def __init__(self):
            self._container = None
            self.state = SimpleNamespace(last_output="")

        def _exec_in_container(self, _command, _workdir=None):
            return 0, raw_output

        def step(self, action):
            exit_code, output = self._exec_in_container(action.command)
            self.state.last_output = output
            return SimpleNamespace(
                output=output,
                info={"source": "upstream", "truncated": False},
                exit_code=exit_code,
            )

    environment = managed.managed_environment_class(_Base, registry)()
    observation = environment.step(
        SimpleNamespace(action_type="exec", command="emit", wait_seconds=1)
    )

    assert observation.output.endswith(managed.EXEC_OUTPUT_TRUNCATION_MARKER)
    assert len(observation.output.encode("utf-8")) == managed.MAX_EXEC_OUTPUT_BYTES
    assert environment.state.last_output == observation.output
    assert environment.state.last_output != raw_output
    assert observation.info == {
        "source": "upstream",
        "truncated": True,
        "exit_code": 0,
        "timed_out": False,
    }


def test_managed_exec_cap_does_not_split_utf8_code_point():
    client = _Client()
    registry = _registry(client)
    prefix_budget = managed.MAX_EXEC_OUTPUT_BYTES - len(
        managed.EXEC_OUTPUT_TRUNCATION_MARKER.encode("utf-8")
    )
    raw_output = (
        "x" * (prefix_budget - 1)
        + "🙂"
        + "tail" * len(managed.EXEC_OUTPUT_TRUNCATION_MARKER)
    )

    class _Base:
        def __init__(self):
            self._container = None
            self.state = SimpleNamespace(last_output="")

        def _exec_in_container(self, _command, _workdir=None):
            return 0, raw_output

        def step(self, action):
            exit_code, output = self._exec_in_container(action.command)
            self.state.last_output = output
            return SimpleNamespace(output=output, info={}, exit_code=exit_code)

    environment = managed.managed_environment_class(_Base, registry)()
    observation = environment.step(
        SimpleNamespace(action_type="exec", command="emit", wait_seconds=1)
    )

    assert observation.output == (
        "x" * (prefix_budget - 1) + managed.EXEC_OUTPUT_TRUNCATION_MARKER
    )
    assert len(observation.output.encode("utf-8")) <= managed.MAX_EXEC_OUTPUT_BYTES
    assert environment.state.last_output == observation.output
    assert observation.info["truncated"] is True


def test_managed_exec_rejects_timeout_outside_signed_range():
    client = _Client()
    registry = _registry(client)

    class _Base:
        def __init__(self):
            self._container = None

        def step(self, _action):
            raise AssertionError("invalid timeout must fail before execution")

    environment = managed.managed_environment_class(_Base, registry)()
    action = SimpleNamespace(action_type="exec", command="sleep 1", wait_seconds=121)

    with pytest.raises(managed.ManagedServerError, match="1..120"):
        environment.step(action)


def test_janitor_success_clears_a_prior_cleanup_gate():
    client = _Client()
    registry = _registry(client, remove_attempts=1)
    flaky = _Container("flaky", registry.labels, failures=1)
    client.containers.values.append(flaky)
    with pytest.raises(managed.ManagedServerError):
        registry.force_remove(flaky)

    assert registry.janitor_once()["removed"] == 1
    registry.assert_reset_allowed()


def test_shutdown_residue_gate_runs_even_when_session_destroy_fails():
    class _Server:
        _sessions = {"bad": object(), "good": object()}

        async def _destroy_session(self, session_id):
            if session_id == "bad":
                raise RuntimeError("destroy failed")

    class _Registry:
        cleaned = False

        def cleanup_all(self):
            self.cleaned = True

    registry = _Registry()
    with pytest.raises(managed.ManagedServerError, match="residue gate still completed"):
        asyncio.run(managed._drain_sessions_and_cleanup(_Server(), registry))
    assert registry.cleaned is True


def test_create_app_uses_composed_lifespan_without_fastapi_event_method(
    monkeypatch,
):
    from fastapi import FastAPI

    events = []

    class _HTTPEnvServer:
        def __init__(self, *_args, max_concurrent_envs, **_kwargs):
            self._sessions = {}
            self.active_sessions = 0
            self.max_concurrent_envs = max_concurrent_envs

        async def _destroy_session(self, _session_id):
            raise AssertionError("there are no sessions in this test")

        def register_routes(self, app):
            async def upstream_start():
                events.append("upstream-start")

            async def upstream_stop():
                events.append("upstream-stop")

            app.router.on_startup.append(upstream_start)
            app.router.on_shutdown.append(upstream_stop)

    class _BaseEnvironment:
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self, *_args, **_kwargs):
            self._container = None

    class _Model:
        pass

    modules = {
        "openenv": types.ModuleType("openenv"),
        "openenv.core": types.ModuleType("openenv.core"),
        "openenv.core.env_server": types.ModuleType("openenv.core.env_server"),
        "openenv.core.env_server.http_server": types.ModuleType(
            "openenv.core.env_server.http_server"
        ),
        "tbench2_env": types.ModuleType("tbench2_env"),
        "tbench2_env.models": types.ModuleType("tbench2_env.models"),
        "tbench2_env.server": types.ModuleType("tbench2_env.server"),
        "tbench2_env.server.tbench2_env_environment": types.ModuleType(
            "tbench2_env.server.tbench2_env_environment"
        ),
    }
    modules["openenv.core.env_server.http_server"].HTTPEnvServer = _HTTPEnvServer
    modules["tbench2_env.models"].Tbench2Action = _Model
    modules["tbench2_env.models"].Tbench2Observation = _Model
    modules[
        "tbench2_env.server.tbench2_env_environment"
    ].Tbench2DockerEnvironment = _BaseEnvironment
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Reproduce the n3 FastAPI surface: app-level add_event_handler is absent.
    monkeypatch.delattr(FastAPI, "add_event_handler", raising=False)
    client = _Client()
    registry = _registry(client)
    original_cleanup = registry.cleanup_all

    def cleanup_all():
        events.append("cleanup")
        return original_cleanup()

    registry.cleanup_all = cleanup_all
    app = managed.create_managed_app(registry=registry, max_concurrent_envs=8)

    async def exercise_lifespan():
        async with app.router.lifespan_context(app):
            events.append("serving")

    asyncio.run(exercise_lifespan())

    assert events[0:3] == ["cleanup", "upstream-start", "serving"]
    assert events[-2:] == ["upstream-stop", "cleanup"]


def test_main_retains_bounded_websocket_keepalive(monkeypatch):
    calls = []
    fake_app = object()

    monkeypatch.setattr(managed, "ManagedContainerRegistry", lambda *_a, **_k: object())
    monkeypatch.setattr(managed, "create_managed_app", lambda **_kwargs: fake_app)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    assert (
        managed.main(
            ["--run-id", "ramp-v2", "--host", "127.0.0.2", "--port", "8123"]
        )
        == 0
    )
    assert calls == [
        (
            (fake_app,),
            {
                "host": "127.0.0.2",
                "port": 8123,
                "workers": 1,
                "ws_ping_interval": managed.WS_PING_INTERVAL_S,
                "ws_ping_timeout": managed.WS_PING_TIMEOUT_S,
            },
        )
    ]
