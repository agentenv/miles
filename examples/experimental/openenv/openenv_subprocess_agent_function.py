"""Run the Terminal-Bench client in an isolated Python environment.

OpenEnv's server package imports its MCP stack eagerly.  Loading that stack in
the Miles/Ray process can replace core runtime dependencies, so the online gate
keeps Miles pristine and launches only the environment client in a small child
process.  The child still calls the per-episode Miles session-server URL; the
parent tracer therefore records the original token IDs, behavior logprobs, and
action masks exactly as it does for an in-process agent function.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_RESULT_SENTINEL = "__MILES_OPENENV_AGENT_RESULT__="
_DEFAULT_PYTHON = "/root/openenv-venv/bin/python"


def _decode_result(stdout: bytes) -> dict[str, Any] | None:
    text = stdout.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if not line.startswith(_RESULT_SENTINEL):
            continue
        try:
            result = json.loads(line[len(_RESULT_SENTINEL) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError("isolated OpenEnv agent returned malformed JSON") from exc
        if result is not None and not isinstance(result, dict):
            raise RuntimeError("isolated OpenEnv agent result must be an object or null")
        return result
    raise RuntimeError("isolated OpenEnv agent returned no result envelope")


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
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Execute one agent episode without importing OpenEnv into Miles."""
    python = Path(os.getenv("OPENENV_AGENT_PYTHON", _DEFAULT_PYTHON))
    if not python.is_file():
        raise FileNotFoundError(f"isolated OpenEnv Python is missing: {python}")
    worker = Path(
        os.getenv(
            "OPENENV_AGENT_WORKER",
            str(Path(__file__).with_name("openenv_agent_worker.py")),
        )
    )
    if not worker.is_file():
        raise FileNotFoundError(f"isolated OpenEnv worker is missing: {worker}")

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
        detail = stderr_text[-4000:] or stdout.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"isolated OpenEnv agent exited with status {process.returncode}: {detail}"
        )
    if stderr_text.strip():
        logger.debug("isolated OpenEnv agent stderr:\n%s", stderr_text[-4000:])
    return _decode_result(stdout)

