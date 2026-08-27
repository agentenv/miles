"""Process boundary for ``openenv_subprocess_agent_function``."""

from __future__ import annotations

import asyncio
import json
import sys

import openenv_agent_function
from openenv_subprocess_agent_function import _RESULT_SENTINEL


def main() -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise TypeError("isolated OpenEnv agent payload must be an object")
    result = asyncio.run(openenv_agent_function.run(**payload))
    print(
        _RESULT_SENTINEL
        + json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
