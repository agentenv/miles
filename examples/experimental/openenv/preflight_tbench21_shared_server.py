"""CPU-only contract probe for the shared Terminal-Bench 2.1 Docker server.

Run this before reserving GPUs for a training job.  The probe uses only the
public OpenEnv client contract: it resets one task, executes ``pwd``, and asks
the server to run its native verifier.  It never reads or executes the task's
solution.

Both the server and this client must come from OpenEnv revision
``d2d4754b333ac285913d113e26c2126207434956``.  The script verifies the local
client checkout revision.  Because the wire protocol does not expose the
server's git SHA, the workdir and native-evaluate checks are also the runtime
gate for the remotely running server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import numbers
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


OPENENV_REVISION = "d2d4754b333ac285913d113e26c2126207434956"
DEFAULT_URL = "http://127.0.0.1:8003"
DEFAULT_TASK = "fix-git"
DEFAULT_WORKDIR = "/app/personal-site"
EXEC_OUTPUT_BOUNDARY_BYTES = 32_768
OUTPUT_BOUNDARY_PROBE_BYTES = 65_536
OUTPUT_BOUNDARY_COMMAND = f"yes x | head -c {OUTPUT_BOUNDARY_PROBE_BYTES}"
MANAGED_STATUS_PATH = "/miles/managed-status"
MANAGED_STATUS_MAX_BYTES = 64 * 1024
DEFAULT_MANAGED_STATUS_TIMEOUT_S = 30.0
MANAGED_STATUS_POLL_INTERVAL_S = 0.25


class PreflightError(RuntimeError):
    """The shared server does not satisfy the TB2.1 runtime contract."""


def _managed_status_url(url: str) -> str:
    return f"{url.rstrip('/')}{MANAGED_STATUS_PATH}"


def _fetch_managed_status(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        _managed_status_url(url), headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            status = int(status_value)
            raw = response.read(MANAGED_STATUS_MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise PreflightError(
                f"managed server status endpoint returned HTTP {error.code}"
            ) from error
        status = error.code
        try:
            raw = error.read(MANAGED_STATUS_MAX_BYTES + 1)
        except OSError as read_error:
            raise PreflightError(
                "managed server status endpoint is unavailable"
            ) from read_error
    except (OSError, urllib.error.URLError) as error:
        raise PreflightError("managed server status endpoint is unavailable") from error
    if status not in (200, 503):
        raise PreflightError(
            f"managed server status endpoint returned HTTP {status}"
        )
    if len(raw) > MANAGED_STATUS_MAX_BYTES:
        raise PreflightError("managed server status response is unbounded")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError("managed server status response is not JSON") from error
    if not isinstance(payload, dict):
        raise PreflightError("managed server status response is not an object")
    return status, payload


def _validated_managed_status(
    payload: Mapping[str, Any], *, expected_run_id: str
) -> tuple[dict[str, Any], dict[str, int]]:
    count_names = (
        "active_sessions",
        "managed_containers",
        "orphan_containers",
        "failed_cleanup_count",
    )
    counts = {name: payload.get(name) for name in count_names}
    if (
        payload.get("run_id") != expected_run_id
        or not isinstance(payload.get("ok"), bool)
        or not isinstance(payload.get("cleanup_blocked"), bool)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        )
    ):
        raise PreflightError(
            "managed server status response is not a valid exact run scope: "
            f"expected_run_id={expected_run_id!r}, "
            f"run_id={payload.get('run_id')!r}, counts={counts!r}, "
            f"ok={payload.get('ok')!r}, "
            f"cleanup_blocked={payload.get('cleanup_blocked')!r}"
        )
    return dict(payload), counts


def require_idle_managed_status(
    payload: Mapping[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    """Fail unless the exact managed server is clean before GPU launch."""
    validated, counts = _validated_managed_status(
        payload, expected_run_id=expected_run_id
    )
    if (
        validated["ok"] is not True
        or validated["cleanup_blocked"] is not False
        or any(value != 0 for value in counts.values())
    ):
        raise PreflightError(
            "managed server is not the requested clean, idle run scope: "
            f"run_id={validated['run_id']!r}, counts={counts!r}, "
            f"cleanup_blocked={validated['cleanup_blocked']!r}"
        )
    return validated


def wait_for_idle_managed_status(
    *,
    url: str,
    expected_run_id: str,
    timeout_s: float,
    poll_interval_s: float = MANAGED_STATUS_POLL_INTERVAL_S,
    fetch_status: Callable[[str, float], tuple[int, dict[str, Any]]] = (
        _fetch_managed_status
    ),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll exact-run status through asynchronous session teardown.

    A valid 200 or 503 response may temporarily be busy while OpenEnv closes a
    session and force-removes its task container. Transport errors, malformed
    responses, and wrong run IDs are never retried here. This helper polls only
    status; it does not retry the probe episode.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise PreflightError("managed status timeout must be finite and positive")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise PreflightError(
            "managed status poll interval must be finite and positive"
        )

    deadline = monotonic() + timeout_s
    last_status: int | None = None
    last_payload: dict[str, Any] | None = None
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise PreflightError(
                "managed server did not become clean and idle within "
                f"{timeout_s:g}s: HTTP {last_status}, payload={last_payload!r}"
            )

        status, payload = fetch_status(url, remaining)
        if status not in (200, 503):
            raise PreflightError(
                f"managed server status endpoint returned HTTP {status}"
            )
        validated, _counts = _validated_managed_status(
            payload, expected_run_id=expected_run_id
        )
        last_status = status
        last_payload = validated
        if monotonic() > deadline:
            continue
        try:
            idle = require_idle_managed_status(
                validated, expected_run_id=expected_run_id
            )
        except PreflightError:
            idle = None
        if idle is not None:
            if status != 200:
                raise PreflightError(
                    "managed server returned HTTP 503 for a clean, idle status"
                )
            return idle

        remaining = deadline - monotonic()
        if remaining <= 0:
            continue
        sleep(min(poll_interval_s, remaining))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _observation_field(result: Any, name: str, default: Any = None) -> Any:
    observation = _field(result, "observation", result)
    return _field(observation, name, default)


def _require_no_server_error(stage: str, result: Any) -> None:
    # OpenEnv versions have carried the error on either StepResult or its
    # observation, so reject a non-empty value in either location.
    errors = (_field(result, "error"), _observation_field(result, "error"))
    for error in errors:
        if error is not None and str(error).strip():
            raise PreflightError(f"{stage} returned a server error: {error}")


def _require_binary_reward(result: Any) -> float:
    reward = _field(result, "reward")
    if reward is None:
        reward = _observation_field(result, "reward")
    if isinstance(reward, bool) or not isinstance(reward, numbers.Real):
        raise PreflightError(f"evaluate reward is not numeric: {reward!r}")
    reward = float(reward)
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise PreflightError(f"evaluate reward is not finite and binary: {reward!r}")
    return reward


def _require_bounded_exec_output(result: Any) -> int:
    """Verify truncation happens on the server before WebSocket transport."""
    _require_no_server_error("output-boundary", result)
    raw_output = _observation_field(result, "output", "")
    if not isinstance(raw_output, str):
        raise PreflightError("output-boundary response is not text")
    output_bytes = len(raw_output.encode("utf-8"))
    info = _observation_field(result, "info", {})
    truncated = _field(info, "truncated", False)
    if (
        not isinstance(truncated, bool)
        or truncated is not True
        or not 0 < output_bytes <= EXEC_OUTPUT_BOUNDARY_BYTES
    ):
        raise PreflightError(
            "shared server did not enforce the terminal output boundary: "
            f"bytes={output_bytes}, truncated={truncated!r}, "
            f"limit={EXEC_OUTPUT_BOUNDARY_BYTES}"
        )
    return output_bytes


def _require_pinned_client(package_file: str) -> None:
    package_dir = Path(package_file).resolve().parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(
            "cannot verify the tbench2_env source revision; install it editable "
            f"from the OpenEnv checkout pinned at {OPENENV_REVISION}"
        ) from exc
    actual = completed.stdout.strip().lower()
    if actual != OPENENV_REVISION:
        raise PreflightError(
            "wrong OpenEnv client revision: "
            f"expected {OPENENV_REVISION}, found {actual or '<empty>'}"
        )


async def probe(
    *,
    env_cls: Any,
    action_cls: Any,
    url: str,
    task: str,
    expected_workdir: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Exercise reset, workdir, bounded output, and native verifier contracts."""
    async with env_cls(base_url=url, message_timeout_s=timeout_s) as env:
        reset_result = await env.reset(task_id=task)
        _require_no_server_error("reset", reset_result)

        pwd_result = await env.step(action_cls(action_type="exec", command="pwd"))
        _require_no_server_error("pwd", pwd_result)
        raw_output = _observation_field(pwd_result, "output", "")
        output = "" if raw_output is None else str(raw_output)
        # A shell normally appends exactly one newline.  Ignore that terminator;
        # whitespace, blank lines, or any other output still fails this check.
        if output.endswith("\r\n"):
            observed_workdir = output[:-2]
        elif output.endswith("\n"):
            observed_workdir = output[:-1]
        else:
            observed_workdir = output
        if observed_workdir != expected_workdir:
            raise PreflightError(
                f"pwd mismatch: expected {expected_workdir!r}, got {observed_workdir!r}"
            )

        boundary_result = await env.step(
            action_cls(action_type="exec", command=OUTPUT_BOUNDARY_COMMAND)
        )
        bounded_output_bytes = _require_bounded_exec_output(boundary_result)

        evaluate_result = await env.step(action_cls(action_type="evaluate"))
        _require_no_server_error("evaluate", evaluate_result)
        reward = _require_binary_reward(evaluate_result)

    return {
        "ok": True,
        "openenv_revision": OPENENV_REVISION,
        "server_url": url,
        "task_id": task,
        "workdir": observed_workdir,
        "bounded_exec_output_bytes": bounded_output_bytes,
        "reward": reward,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a pinned OpenEnv Terminal-Bench 2.1 shared Docker server without GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=os.getenv("OPENENV_ENV_URL", DEFAULT_URL))
    parser.add_argument(
        "--run-id",
        default=os.getenv("MILES_TBENCH_RUN_ID"),
        help="Exact managed-server run ID (or MILES_TBENCH_RUN_ID).",
    )
    parser.add_argument("--task", default=os.getenv("OPENENV_PREFLIGHT_TASK", DEFAULT_TASK))
    parser.add_argument(
        "--expected-workdir",
        default=os.getenv("OPENENV_EXPECTED_WORKDIR", DEFAULT_WORKDIR),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("OPENENV_MESSAGE_TIMEOUT_S", "900")),
    )
    parser.add_argument(
        "--managed-status-timeout-s",
        type=float,
        default=float(
            os.getenv(
                "MILES_TBENCH_MANAGED_STATUS_TIMEOUT_S",
                str(DEFAULT_MANAGED_STATUS_TIMEOUT_S),
            )
        ),
        help="Bounded wait for exact idle status after the probe closes.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import tbench2_env
        from tbench2_env import Tbench2Action, Tbench2Env
    except ImportError as exc:
        raise PreflightError(
            "tbench2_env is not installed; install it editable from OpenEnv "
            f"revision {OPENENV_REVISION}"
        ) from exc

    _require_pinned_client(tbench2_env.__file__)
    if not args.run_id:
        raise PreflightError("--run-id or MILES_TBENCH_RUN_ID is required")
    before_status, before_payload = await asyncio.to_thread(
        _fetch_managed_status,
        args.url,
        min(args.timeout_s, args.managed_status_timeout_s),
    )
    if before_status != 200:
        raise PreflightError(
            f"managed server was not idle before the probe: HTTP {before_status}"
        )
    before = require_idle_managed_status(
        before_payload, expected_run_id=args.run_id
    )
    result = await probe(
        env_cls=Tbench2Env,
        action_cls=Tbench2Action,
        url=args.url,
        task=args.task,
        expected_workdir=args.expected_workdir,
        timeout_s=args.timeout_s,
    )
    after = await asyncio.to_thread(
        wait_for_idle_managed_status,
        url=args.url,
        expected_run_id=args.run_id,
        timeout_s=args.managed_status_timeout_s,
    )
    result["managed_server"] = {
        "run_id": args.run_id,
        "status_path": MANAGED_STATUS_PATH,
        "clean_before": before["ok"],
        "clean_after": after["ok"],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(_async_main(args))
    except Exception as exc:  # keep launch-gate output concise and shell-friendly
        print(f"TB2.1 PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
