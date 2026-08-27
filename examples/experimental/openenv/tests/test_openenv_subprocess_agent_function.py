from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openenv_subprocess_agent_function as subprocess_agent  # noqa: E402


@pytest.mark.asyncio
async def test_isolated_agent_round_trips_json(monkeypatch, tmp_path: Path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        f"print({subprocess_agent._RESULT_SENTINEL!r} + json.dumps({{'reward': 1.0, 'url': payload['base_url']}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENENV_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("OPENENV_AGENT_WORKER", str(worker))

    result = await subprocess_agent.run(
        "http://session/v1",
        [{"role": "user", "content": "task"}],
        {"temperature": 0.7},
        {"task_id": "fix-git"},
    )

    assert result == {"reward": 1.0, "url": "http://session/v1"}


def test_isolated_agent_rejects_missing_result_envelope():
    with pytest.raises(RuntimeError, match="no result envelope"):
        subprocess_agent._decode_result(b"ordinary child output\n")


def test_isolated_agent_rejects_non_object_result():
    payload = subprocess_agent._RESULT_SENTINEL + json.dumps([1, 2, 3])
    with pytest.raises(RuntimeError, match="object or null"):
        subprocess_agent._decode_result(payload.encode())
