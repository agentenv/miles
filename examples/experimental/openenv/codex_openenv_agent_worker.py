"""Isolated child entry point for the Codex/OpenEnv agent function."""

from __future__ import annotations

import asyncio
import json
import sys

import codex_openenv_agent_function
from codex_openenv_subprocess_agent_function import _RESULT_SENTINEL


def main() -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise TypeError("isolated Codex OpenEnv payload must be an object")
    result = asyncio.run(codex_openenv_agent_function.run(**payload))
    print(
        _RESULT_SENTINEL
        + json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

