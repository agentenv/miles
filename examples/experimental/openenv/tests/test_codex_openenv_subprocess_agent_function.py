from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import codex_openenv_subprocess_agent_function as subprocess_agent  # noqa: E402
from yeto.rl.tbench_outcome import (  # noqa: E402
    TEST_SH_VERIFIER,
    build_signed_metadata,
)


_HMAC_KEY = "test-only-terminal-bench-key-32-bytes"


@pytest.mark.asyncio
async def test_isolated_codex_openenv_agent_round_trips_json(monkeypatch, tmp_path: Path):
    trusted = {
        "reward": 1.0,
        "exit_status": "completed",
        "task": "fix-git",
        **build_signed_metadata(
            task_id="fix-git",
            sample_id="baseline:fix-git:r0",
            episode_id="episode-1",
            status="completed",
            reward=1.0,
            verifier=TEST_SH_VERIFIER,
            testsh_rc=0,
            key=_HMAC_KEY,
        ),
    }
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        f"print({subprocess_agent._RESULT_SENTINEL!r} + {json.dumps(json.dumps(trusted))})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENENV_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("CODEX_OPENENV_AGENT_WORKER", str(worker))
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", _HMAC_KEY)

    result = await subprocess_agent.run(
        "http://session",
        "unused",
        {"temperature": 0.7},
        {"task_id": "fix-git"},
    )
    assert result == trusted
    assert subprocess_agent.run._miles_fail_closed_result is True


def test_isolated_codex_openenv_agent_requires_object_result():
    raw = subprocess_agent._RESULT_SENTINEL + json.dumps([1, 2])
    with pytest.raises(RuntimeError, match="must be an object"):
        subprocess_agent._decode_result(raw.encode())


def test_isolated_codex_openenv_agent_rejects_null_result():
    raw = subprocess_agent._RESULT_SENTINEL + "null"
    with pytest.raises(RuntimeError, match="null without a trustworthy outcome"):
        subprocess_agent._decode_result(raw.encode())


@pytest.mark.asyncio
async def test_isolated_agent_failure_diagnostic_is_bounded(
    monkeypatch, tmp_path: Path
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\n"
        "sys.stderr.write('x' * 10000 + 'FINAL-AGENT-ERROR')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENENV_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("CODEX_OPENENV_AGENT_WORKER", str(worker))

    with pytest.raises(RuntimeError) as raised:
        await subprocess_agent.run("http://session", "unused", {}, {})

    diagnostic = str(raised.value)
    assert "FINAL-AGENT-ERROR" in diagnostic
    assert "last stderr line:" in diagnostic
    assert "diagnostic truncated" in diagnostic
    assert len(diagnostic.encode("utf-8")) < 2300
