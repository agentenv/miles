"""Keep OpenEnv dependencies outside the Miles/Ray learner process."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_RESULT_SENTINEL = "__MILES_CODEX_OPENENV_AGENT_RESULT__="
_DEFAULT_PYTHON = "/root/openenv-venv/bin/python"
_MAX_FAILURE_DIAGNOSTIC_BYTES = 2048


def _bounded_diagnostic(value: bytes) -> str:
    marker = b"[diagnostic truncated]..."
    if len(value) > _MAX_FAILURE_DIAGNOSTIC_BYTES:
        stripped = value.rstrip()
        last_line = stripped.splitlines()[-1] if stripped else b""
        summary = b"last stderr line: " + last_line[-512:] + b"\n"
        remaining = _MAX_FAILURE_DIAGNOSTIC_BYTES - len(summary) - len(marker)
        value = summary + marker + value[-max(0, remaining) :]
    return value.decode("utf-8", errors="ignore")


def _decode_result(stdout: bytes) -> dict[str, Any]:
    text = stdout.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if not line.startswith(_RESULT_SENTINEL):
            continue
        try:
            result = json.loads(line[len(_RESULT_SENTINEL) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "isolated Codex OpenEnv agent returned malformed JSON"
            ) from exc
        if result is None:
            raise RuntimeError(
                "isolated Codex OpenEnv agent returned null without a trustworthy outcome"
            )
        if not isinstance(result, dict):
            raise RuntimeError("isolated Codex OpenEnv agent result must be an object")
        return result
    raise RuntimeError("isolated Codex OpenEnv agent returned no result envelope")


def _validate_trustworthy_result(result: dict[str, Any]) -> dict[str, Any]:
    from yeto.rl.tbench_outcome import verified_outcome

    outcome, signed_reward = verified_outcome(result)
    reported_reward = result.get("reward")
    if (
        isinstance(reported_reward, bool)
        or not isinstance(reported_reward, (int, float))
        or not math.isfinite(float(reported_reward))
        or float(reported_reward) != signed_reward
    ):
        raise RuntimeError(
            "isolated Codex OpenEnv result reward differs from its signed outcome"
        )
    if result.get("exit_status") != outcome["status"]:
        raise RuntimeError(
            "isolated Codex OpenEnv result status differs from its signed outcome"
        )
    return result


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    python = Path(os.getenv("OPENENV_AGENT_PYTHON", _DEFAULT_PYTHON))
    if not python.is_file():
        raise FileNotFoundError(f"isolated OpenEnv Python is missing: {python}")
    worker = Path(
        os.getenv(
            "CODEX_OPENENV_AGENT_WORKER",
            str(Path(__file__).with_name("codex_openenv_agent_worker.py")),
        )
    )
    if not worker.is_file():
        raise FileNotFoundError(f"isolated Codex OpenEnv worker is missing: {worker}")

    payload = json.dumps(
        {
            "base_url": base_url,
            "prompt": prompt,
            "request_kwargs": request_kwargs or {},
            "metadata": metadata or {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    child_env = os.environ.copy()
    child_pythonpath = [str(worker.parent)]
    if existing := child_env.get("PYTHONPATH"):
        child_pythonpath.extend(existing.split(os.pathsep))
    child_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(child_pythonpath))

    process = await asyncio.create_subprocess_exec(
        str(python),
        str(worker),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
    )
    try:
        stdout, stderr = await process.communicate(payload)
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    stderr_text = stderr.decode("utf-8", errors="replace")
    if process.returncode != 0:
        detail = _bounded_diagnostic(stderr or stdout)
        raise RuntimeError(
            f"isolated Codex OpenEnv agent exited with status "
            f"{process.returncode}: {detail}"
        )
    if stderr_text.strip():
        logger.debug("isolated Codex OpenEnv agent stderr:\n%s", stderr_text[-4000:])
    return _validate_trustworthy_result(_decode_result(stdout))


# ``agentic_tool_call`` uses this opt-in marker to treat a defensive null from
# this fail-closed adapter as an episode failure instead of optional metadata.
run._miles_fail_closed_result = True
