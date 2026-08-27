#!/usr/bin/env python3
"""Miles-owned, leak-resistant Terminal-Bench 2.1 shared Docker server.

This module wraps the pinned OpenEnv server without modifying OpenEnv.  Every
task container is labeled before creation, and cleanup is restricted to the
exact pair of Miles' managed label and one validated run ID.  Run exactly one
server process; OpenEnv provides session concurrency inside that process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import re
import shlex
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4


MANAGED_LABEL = "miles.tbench21.managed"
RUN_ID_LABEL = "miles.tbench21.run_id"
OWNER_LABEL = "miles.tbench21.owner"
STATUS_PATH = "/miles/managed-status"
WS_PING_INTERVAL_S = 60.0
WS_PING_TIMEOUT_S = 300.0
MAX_EXEC_TIMEOUT_S = 120.0
EXEC_WATCHDOG_GRACE_S = 15.0
MAX_EXEC_OUTPUT_BYTES = 32_768
EXEC_OUTPUT_TRUNCATION_MARKER = "\n[output truncated by managed Terminal-Bench server]"
_RUN_ID = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?")


class ManagedServerError(RuntimeError):
    """The managed server cannot preserve its container-lifecycle contract."""


def _cap_exec_output(output: str) -> tuple[str, bool]:
    """Bound one exec observation without splitting a UTF-8 code point."""
    raw = output.encode("utf-8")
    if len(raw) <= MAX_EXEC_OUTPUT_BYTES:
        return output, False
    marker = EXEC_OUTPUT_TRUNCATION_MARKER.encode("utf-8")
    prefix = raw[: MAX_EXEC_OUTPUT_BYTES - len(marker)]
    return prefix.decode("utf-8", errors="ignore") + EXEC_OUTPUT_TRUNCATION_MARKER, True


def validate_run_id(value: str) -> str:
    """Return one Docker-label-safe run ID or fail before touching Docker."""
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError(
            "run ID must be 1..128 lowercase ASCII letters/digits with internal ._-"
        )
    return value


def _container_id(container: Any) -> str:
    value = getattr(container, "id", None)
    if not isinstance(value, str) or not value:
        raise ManagedServerError("Docker returned a container without an ID")
    return value


def _container_labels(container: Any) -> dict[str, str]:
    # Docker's sparse Container objects intentionally contain only the
    # /containers/json summary.  Their ``labels`` property expects full inspect
    # data under Config and can raise KeyError, while the same exact labels are
    # available directly in the summary's top-level Labels mapping.
    attrs = getattr(container, "attrs", None)
    labels = attrs.get("Labels") if isinstance(attrs, dict) else None
    if not isinstance(labels, dict):
        config = attrs.get("Config") if isinstance(attrs, dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        labels = getattr(container, "labels", None)
    if not isinstance(labels, dict):
        return {}
    return {
        key: value
        for key, value in labels.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _not_found(error: BaseException) -> bool:
    if type(error).__name__ == "NotFound":
        return True
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    return status == 404 or getattr(response, "status_code", None) == 404


class ManagedContainerRegistry:
    """Own exact-run Docker inventory, cleanup gates, and orphan accounting."""

    def __init__(
        self,
        run_id: str,
        *,
        client_factory: Callable[[], Any] | None = None,
        remove_attempts: int = 3,
        retry_delay_s: float = 0.25,
        janitor_interval_s: float = 30.0,
        janitor_batch_size: int = 64,
        docker_timeout_s: float = 30.0,
    ) -> None:
        self.run_id = validate_run_id(run_id)
        if isinstance(remove_attempts, bool) or not 1 <= remove_attempts <= 10:
            raise ValueError("remove_attempts must be in [1, 10]")
        if not 0 <= retry_delay_s <= 10:
            raise ValueError("retry_delay_s must be in [0, 10]")
        if not 1 <= janitor_interval_s <= 3600:
            raise ValueError("janitor_interval_s must be in [1, 3600]")
        if isinstance(janitor_batch_size, bool) or not 1 <= janitor_batch_size <= 4096:
            raise ValueError("janitor_batch_size must be in [1, 4096]")
        if not 1 <= docker_timeout_s <= 300:
            raise ValueError("docker_timeout_s must be in [1, 300]")
        self.remove_attempts = remove_attempts
        self.retry_delay_s = retry_delay_s
        self.janitor_interval_s = janitor_interval_s
        self.janitor_batch_size = janitor_batch_size
        self.docker_timeout_s = docker_timeout_s
        self._client_factory = client_factory or self._default_client
        self._client: Any | None = None
        self._lock = threading.RLock()
        self._active_by_owner: dict[str, str] = {}
        self._pending_owners: set[str] = set()
        self._failed_cleanup_ids: set[str] = set()
        self._inventory_failed = False
        self._last_janitor_error: str | None = None
        self._last_janitor_at: float | None = None

    @property
    def labels(self) -> dict[str, str]:
        return {MANAGED_LABEL: "true", RUN_ID_LABEL: self.run_id}

    @property
    def label_filters(self) -> dict[str, list[str]]:
        return {
            "label": [f"{MANAGED_LABEL}=true", f"{RUN_ID_LABEL}={self.run_id}"]
        }

    def _default_client(self) -> Any:
        try:
            import docker
        except ImportError as error:
            raise ManagedServerError("the managed server requires docker-py") from error
        return docker.from_env(timeout=self.docker_timeout_s)

    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    def assert_reset_allowed(self) -> None:
        with self._lock:
            if self._inventory_failed or self._failed_cleanup_ids:
                raise ManagedServerError(
                    "a prior managed-container cleanup failed; reset is blocked until "
                    "the janitor proves the exact-run inventory clean"
                )

    def claim(self, owner_id: str, container: Any) -> None:
        container_id = _container_id(container)
        with self._lock:
            previous = self._active_by_owner.get(owner_id)
            if previous is not None and previous != container_id:
                raise ManagedServerError("one environment attempted to own two containers")
            self._active_by_owner[owner_id] = container_id
            self._pending_owners.discard(owner_id)

    def reserve(self, owner_id: str) -> None:
        """Reserve a non-serializing owner slot before Docker exposes a container."""
        with self._lock:
            if owner_id in self._pending_owners or owner_id in self._active_by_owner:
                raise ManagedServerError("one environment attempted concurrent container creation")
            self._pending_owners.add(owner_id)

    def cancel_reservation(self, owner_id: str) -> None:
        with self._lock:
            self._pending_owners.discard(owner_id)

    def release(self, owner_id: str, container: Any) -> None:
        container_id = _container_id(container)
        with self._lock:
            if self._active_by_owner.get(owner_id) == container_id:
                self._active_by_owner.pop(owner_id, None)
            self._pending_owners.discard(owner_id)

    def _inventory(self) -> list[Any]:
        try:
            containers = self.client().containers.list(
                all=True,
                filters=self.label_filters,
                # Without sparse mode docker-py first lists IDs and then
                # inspects them one by one.  At rollout-scale churn, a task can
                # disappear between those calls and the SDK raises NotFound for
                # the whole inventory.  Sparse mode returns the daemon's one
                # label-filtered snapshot and removes that race and N+1 load.
                sparse=True,
            )
        except Exception as error:
            with self._lock:
                self._inventory_failed = True
                self._last_janitor_error = f"inventory: {type(error).__name__}: {error}"
            raise ManagedServerError("cannot inventory exact-run managed containers") from error
        if not isinstance(containers, list):
            with self._lock:
                self._inventory_failed = True
                self._last_janitor_error = "inventory: malformed container list"
            raise ManagedServerError("Docker returned a malformed container inventory")
        try:
            ids: set[str] = set()
            for container in containers:
                container_id = _container_id(container)
                labels = _container_labels(container)
                if any(labels.get(key) != value for key, value in self.labels.items()):
                    raise ManagedServerError(
                        "Docker sparse inventory returned a container outside the "
                        "exact managed run scope"
                    )
                ids.add(container_id)
        except Exception as error:
            with self._lock:
                self._inventory_failed = True
                self._last_janitor_error = (
                    f"inventory validation: {type(error).__name__}: {error}"
                )
            if isinstance(error, ManagedServerError):
                raise
            raise ManagedServerError(
                "Docker returned a malformed container inventory"
            ) from error
        with self._lock:
            self._inventory_failed = False
            self._failed_cleanup_ids.intersection_update(ids)
        return containers

    def force_remove(self, container: Any) -> None:
        """Force-remove one container with bounded attempts and a cleanup latch."""
        container_id = _container_id(container)
        last_error: BaseException | None = None
        for attempt in range(self.remove_attempts):
            try:
                container.remove(force=True, v=True)
            except Exception as error:
                if _not_found(error):
                    last_error = None
                    break
                last_error = error
                if attempt + 1 < self.remove_attempts and self.retry_delay_s:
                    time.sleep(self.retry_delay_s)
            else:
                last_error = None
                break
        with self._lock:
            if last_error is None:
                self._failed_cleanup_ids.discard(container_id)
                stale_owners = [
                    owner
                    for owner, active_id in self._active_by_owner.items()
                    if active_id == container_id
                ]
                for owner in stale_owners:
                    self._active_by_owner.pop(owner, None)
            else:
                self._failed_cleanup_ids.add(container_id)
                self._last_janitor_error = (
                    f"remove {container_id}: {type(last_error).__name__}: {last_error}"
                )
        if last_error is not None:
            raise ManagedServerError(
                f"failed to force-remove managed container {container_id} after "
                f"{self.remove_attempts} attempts"
            ) from last_error

    def cleanup_all(self) -> dict[str, int]:
        """Remove every container in this exact run scope; never broaden filters."""
        containers = self._inventory()
        failures: list[str] = []
        for container in containers:
            try:
                self.force_remove(container)
            except ManagedServerError:
                failures.append(_container_id(container))
        if failures:
            raise ManagedServerError(
                f"exact-run cleanup failed for {len(failures)} container(s)"
            )
        return {"found": len(containers), "removed": len(containers)}

    def janitor_once(self) -> dict[str, int]:
        """Remove a bounded batch of labeled containers with no active owner."""
        containers = self._inventory()
        with self._lock:
            active_ids = set(self._active_by_owner.values())
            pending_owners = set(self._pending_owners)
        orphans = [
            container
            for container in containers
            if _container_id(container) not in active_ids
            and _container_labels(container).get(OWNER_LABEL) not in pending_owners
        ]
        selected = orphans[: self.janitor_batch_size]
        failures = 0
        for container in selected:
            try:
                self.force_remove(container)
            except ManagedServerError:
                failures += 1
        with self._lock:
            self._last_janitor_at = time.time()
            if failures == 0:
                self._last_janitor_error = None
        return {
            "managed": len(containers),
            "orphans": len(orphans),
            "attempted": len(selected),
            "removed": len(selected) - failures,
            "failed": failures,
        }

    def status(self, *, active_sessions: int) -> dict[str, Any]:
        containers = self._inventory()
        with self._lock:
            active_ids = set(self._active_by_owner.values())
            pending_owners = set(self._pending_owners)
            failed_ids = sorted(self._failed_cleanup_ids)
            last_error = self._last_janitor_error
            last_at = self._last_janitor_at
        managed_ids = {_container_id(container) for container in containers}
        pending_ids = {
            _container_id(container)
            for container in containers
            if _container_labels(container).get(OWNER_LABEL) in pending_owners
        }
        orphan_ids = managed_ids - active_ids - pending_ids
        return {
            "ok": not failed_ids and not orphan_ids,
            "run_id": self.run_id,
            "active_sessions": active_sessions,
            "managed_containers": len(managed_ids),
            "orphan_containers": len(orphan_ids),
            "cleanup_blocked": bool(failed_ids),
            "failed_cleanup_count": len(failed_ids),
            "last_janitor_at": last_at,
            "last_janitor_error": last_error,
        }


class _ManagedContainersProxy:
    def __init__(self, delegate: Any, registry: ManagedContainerRegistry, owner_id: str):
        self._delegate = delegate
        self._registry = registry
        self._owner_id = owner_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self._registry.assert_reset_allowed()
        supplied = kwargs.get("labels")
        if supplied is None:
            labels: dict[str, str] = {}
        elif isinstance(supplied, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in supplied.items()
        ):
            labels = dict(supplied)
        else:
            raise ManagedServerError("task-container labels must be a string mapping")
        required = {**self._registry.labels, OWNER_LABEL: self._owner_id}
        for key, value in required.items():
            if key in labels and labels[key] != value:
                raise ManagedServerError(f"task container attempted to override {key}")
            labels[key] = value
        kwargs["labels"] = labels
        self._registry.reserve(self._owner_id)
        try:
            container = self._delegate.run(*args, **kwargs)
        except BaseException:
            self._registry.cancel_reservation(self._owner_id)
            raise
        try:
            self._registry.claim(self._owner_id, container)
        except BaseException:
            self._registry.cancel_reservation(self._owner_id)
            try:
                self._registry.force_remove(container)
            finally:
                raise
        return container


class _ManagedDockerClientProxy:
    def __init__(self, delegate: Any, registry: ManagedContainerRegistry, owner_id: str):
        self._delegate = delegate
        self.containers = _ManagedContainersProxy(
            delegate.containers, registry, owner_id
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def managed_environment_class(
    base_environment: type[Any], registry: ManagedContainerRegistry
) -> type[Any]:
    """Build a Tbench2DockerEnvironment subclass bound to one registry."""

    class ManagedTbench2DockerEnvironment(base_environment):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._miles_owner_id = uuid4().hex
            self._miles_cleanup_failed = False
            self._miles_client_proxy: _ManagedDockerClientProxy | None = None
            self._miles_exec_timeout_s: float | None = None
            self._miles_last_exec_status: int | None = None
            self._miles_exec_fatal: BaseException | None = None

        def _get_docker_client(self) -> Any:
            if self._miles_client_proxy is None:
                self._miles_client_proxy = _ManagedDockerClientProxy(
                    registry.client(), registry, self._miles_owner_id
                )
            return self._miles_client_proxy

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            if self._miles_cleanup_failed:
                raise ManagedServerError(
                    "this session previously failed container cleanup; reconnect before reset"
                )
            registry.assert_reset_allowed()
            return super().reset(*args, **kwargs)

        def _exec_in_container(
            self, command: str, workdir: str | None = None
        ) -> tuple[int, str]:
            timeout_s = self._miles_exec_timeout_s
            if timeout_s is None:
                return super()._exec_in_container(command, workdir)

            # Preserve the model command as one argv element.  Coreutils timeout
            # normally ends only the command process group, leaving the task
            # container and all earlier work available to the next turn.
            duration = f"{timeout_s:.3f}s"
            wrapped = (
                "timeout_bin=$(command -v timeout) || exit 125; "
                '"$timeout_bin" --signal=TERM --kill-after=5s '
                f"{shlex.quote(duration)} bash -c {shlex.quote(command)}"
            )
            result: list[tuple[int, str]] = []
            failures: list[BaseException] = []
            finished = threading.Event()
            parent = super(ManagedTbench2DockerEnvironment, self)

            def execute() -> None:
                try:
                    result.append(parent._exec_in_container(wrapped, workdir))
                except BaseException as error:
                    failures.append(error)
                finally:
                    finished.set()

            threading.Thread(
                target=execute,
                name=f"miles-tbench-exec-{self._miles_owner_id[:8]}",
                daemon=True,
            ).start()
            if not finished.wait(timeout_s + EXEC_WATCHDOG_GRACE_S):
                fatal = ManagedServerError(
                    "managed task command exceeded its server-side kill grace"
                )
                try:
                    self.close()
                except BaseException as cleanup_error:
                    fatal.__cause__ = cleanup_error
                self._miles_exec_fatal = fatal
                self._miles_last_exec_status = 125
                return 125, str(fatal)
            if failures:
                raise failures[0]
            if len(result) != 1:
                raise ManagedServerError("managed task command returned no result")
            exit_code, output = result[0]
            self._miles_last_exec_status = int(exit_code)
            if exit_code == 125:
                fatal = ManagedServerError(
                    "task image could not enforce the managed command deadline"
                )
                try:
                    self.close()
                except BaseException as cleanup_error:
                    fatal.__cause__ = cleanup_error
                self._miles_exec_fatal = fatal
            return exit_code, output

        def step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
            is_exec = getattr(action, "action_type", None) == "exec"
            self._miles_last_exec_status = None
            self._miles_exec_fatal = None
            if is_exec:
                requested = getattr(action, "wait_seconds", None)
                timeout_s = MAX_EXEC_TIMEOUT_S if requested is None else requested
                if (
                    isinstance(timeout_s, bool)
                    or not isinstance(timeout_s, (int, float))
                    or not math.isfinite(float(timeout_s))
                    or not 1.0 <= float(timeout_s) <= MAX_EXEC_TIMEOUT_S
                ):
                    raise ManagedServerError(
                        "task command timeout is outside the managed 1..120 second range"
                    )
                self._miles_exec_timeout_s = float(timeout_s)
            try:
                observation = super().step(action, *args, **kwargs)
            finally:
                self._miles_exec_timeout_s = None
            if not is_exec:
                return observation
            output, truncated = _cap_exec_output(observation.output)
            observation.output = output
            self.state.last_output = output
            if self._miles_exec_fatal is not None:
                raise self._miles_exec_fatal
            if self._miles_last_exec_status is None:
                raise ManagedServerError("managed task command produced no exit status")
            info = getattr(observation, "info", None)
            info = dict(info) if isinstance(info, dict) else {}
            if truncated:
                info["truncated"] = True
            info.update(
                {
                    "exit_code": self._miles_last_exec_status,
                    "timed_out": self._miles_last_exec_status == 124,
                }
            )
            observation.info = info
            return observation

        def close(self) -> None:
            container = getattr(self, "_container", None)
            if container is not None:
                registry.release(self._miles_owner_id, container)
                try:
                    registry.force_remove(container)
                except BaseException:
                    self._miles_cleanup_failed = True
                    raise
                self._container = None
                self._miles_cleanup_failed = False
            self._task_dir = None
            self._instruction = ""
            self._workdir = ""

    ManagedTbench2DockerEnvironment.__name__ = "ManagedTbench2DockerEnvironment"
    return ManagedTbench2DockerEnvironment


async def _janitor_loop(
    registry: ManagedContainerRegistry, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            await asyncio.to_thread(registry.janitor_once)
        except Exception:
            logging.exception("managed Terminal-Bench janitor pass failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=registry.janitor_interval_s)
        except TimeoutError:
            pass


async def _drain_sessions_and_cleanup(server: Any, registry: ManagedContainerRegistry) -> None:
    """Drain pinned OpenEnv sessions, then always run the exact-scope residue gate."""
    session_ids = list(server._sessions)
    results = (
        await asyncio.gather(
            *(server._destroy_session(session_id) for session_id in session_ids),
            return_exceptions=True,
        )
        if session_ids
        else []
    )
    destroy_errors = [result for result in results if isinstance(result, BaseException)]
    try:
        await asyncio.to_thread(registry.cleanup_all)
    except BaseException as cleanup_error:
        if destroy_errors and hasattr(cleanup_error, "add_note"):
            cleanup_error.add_note(
                f"{len(destroy_errors)} OpenEnv session destroy operation(s) also failed"
            )
        raise
    if destroy_errors:
        raise ManagedServerError(
            f"{len(destroy_errors)} OpenEnv session destroy operation(s) failed; "
            "the exact-run Docker residue gate still completed"
        ) from destroy_errors[0]


def create_managed_app(
    *, registry: ManagedContainerRegistry, max_concurrent_envs: int
) -> Any:
    """Create the pinned OpenEnv-compatible FastAPI app plus managed status."""
    if isinstance(max_concurrent_envs, bool) or not 1 <= max_concurrent_envs <= 512:
        raise ValueError("max_concurrent_envs must be in [1, 512]")
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from openenv.core.env_server.http_server import HTTPEnvServer
        from tbench2_env.models import Tbench2Action, Tbench2Observation
        from tbench2_env.server.tbench2_env_environment import (
            Tbench2DockerEnvironment,
        )
    except ImportError as error:
        raise ManagedServerError(
            "install pinned OpenEnv/tbench2_env before starting the managed server"
        ) from error

    environment = managed_environment_class(Tbench2DockerEnvironment, registry)
    app = FastAPI(title="Miles Managed Terminal-Bench 2.1 Server")
    server = HTTPEnvServer(
        environment,
        Tbench2Action,
        Tbench2Observation,
        max_concurrent_envs=max_concurrent_envs,
        env_name="tbench2_env (Miles managed Docker mode)",
    )
    server.register_routes(app)
    upstream_lifespan = app.router.lifespan_context
    stop = asyncio.Event()
    janitor: asyncio.Task[None] | None = None

    async def startup() -> None:
        nonlocal janitor
        await asyncio.to_thread(registry.cleanup_all)
        janitor = asyncio.create_task(
            _janitor_loop(registry, stop), name="miles-tbench21-janitor"
        )

    async def shutdown() -> None:
        stop.set()
        if janitor is not None:
            await janitor
        # Pinned OpenEnv stops only its idle reaper on app shutdown. Explicitly
        # destroy every remaining session so its executor calls our close(). The
        # final label-scoped residue gate runs even when session destruction errs.
        await _drain_sessions_and_cleanup(server, registry)

    @asynccontextmanager
    async def managed_lifespan(lifespan_app: Any):
        # Compose rather than replace OpenEnv's router lifespan. Managed cleanup
        # completes before OpenEnv starts accepting sessions; on shutdown,
        # OpenEnv stops its own services before we drain sessions and enforce the
        # exact-label Docker residue gate.
        await startup()
        try:
            async with upstream_lifespan(lifespan_app):
                yield
        finally:
            await shutdown()

    app.router.lifespan_context = managed_lifespan

    @app.get(STATUS_PATH)
    async def managed_status() -> Any:
        try:
            payload = await asyncio.to_thread(
                registry.status, active_sessions=server.active_sessions
            )
        except ManagedServerError as error:
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "run_id": registry.run_id,
                    "error": str(error),
                },
            )
        return JSONResponse(status_code=200 if payload["ok"] else 503, content=payload)

    app.state.miles_managed_registry = registry
    app.state.miles_openenv_server = server
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Miles-managed Terminal-Bench 2.1 shared server."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--run-id", default=os.getenv("MILES_TBENCH_RUN_ID"))
    parser.add_argument(
        "--max-concurrent-envs",
        type=int,
        default=int(os.getenv("MAX_CONCURRENT_ENVS", "300")),
    )
    parser.add_argument(
        "--janitor-interval-s",
        type=float,
        default=float(os.getenv("MILES_TBENCH_JANITOR_INTERVAL_S", "30")),
    )
    parser.add_argument(
        "--janitor-batch-size",
        type=int,
        default=int(os.getenv("MILES_TBENCH_JANITOR_BATCH_SIZE", "64")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_id is None:
        raise SystemExit("--run-id or MILES_TBENCH_RUN_ID is required")
    registry = ManagedContainerRegistry(
        args.run_id,
        janitor_interval_s=args.janitor_interval_s,
        janitor_batch_size=args.janitor_batch_size,
    )
    app = create_managed_app(
        registry=registry, max_concurrent_envs=args.max_concurrent_envs
    )
    import uvicorn

    # Multiple workers would misclassify another worker's live containers as
    # orphans. OpenEnv's in-process session cap provides the required concurrency.
    # Stock Codex may briefly starve its OpenEnv client's event loop while a
    # model/tool turn is in flight.  Uvicorn's 20 s default closes that healthy
    # local connection with 1011 ("keepalive ping timeout").  Retain bounded
    # dead-peer detection, but make its timeout longer than the signed 120 s
    # terminal-tool ceiling.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        ws_ping_interval=WS_PING_INTERVAL_S,
        ws_ping_timeout=WS_PING_TIMEOUT_S,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
