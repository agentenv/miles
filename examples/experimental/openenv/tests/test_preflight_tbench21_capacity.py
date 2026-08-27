"""Offline tests for the TB2.1 host-capacity preflight."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import preflight_tbench21_capacity as preflight  # noqa: E402


def _write_cmdline(proc_root: Path, pid: int, *argv: str) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    (process_dir / "cmdline").write_bytes(b"\0".join(argument.encode() for argument in argv) + b"\0")


@pytest.mark.parametrize(
    "argv",
    [
        (
            "/usr/bin/python3",
            "/workspace/miles/examples/experimental/openenv/managed_tbench21_server.py",
            "--port",
            "8003",
        ),
        (
            "/workspace/miles/examples/experimental/openenv/managed_tbench21_server.py",
            "--port",
            "8003",
        ),
        (
            "/usr/bin/python3",
            "-m",
            "tbench2_env.server.app",
        ),
        (
            "/usr/bin/python3",
            "-m",
            "uvicorn",
            "tbench2_env.server.app:app",
        ),
    ],
)
def test_server_pid_recognizes_supported_server_invocations(tmp_path, argv):
    _write_cmdline(tmp_path, 41, *argv)

    assert preflight._server_pid(None, proc_root=tmp_path) == 41


@pytest.mark.parametrize(
    "argv",
    [
        ("rg", "managed_tbench21_server.py", "/workspace"),
        ("bash", "-c", "python managed_tbench21_server.py"),
        ("python3", "worker.py", "managed_tbench21_server.py"),
        ("python3", "managed_tbench21_server.py.backup"),
        ("uvicorn", "unrelated.app:app", "tbench2_env.server.app"),
    ],
)
def test_server_pid_rejects_commands_that_only_mention_server_names(tmp_path, argv):
    _write_cmdline(tmp_path, 52, *argv)

    with pytest.raises(preflight.CapacityError, match=r"found \[\]"):
        preflight._server_pid(None, proc_root=tmp_path)


def test_server_pid_requires_one_unambiguous_live_server(tmp_path):
    _write_cmdline(tmp_path, 7, "python3", "managed_tbench21_server.py")
    _write_cmdline(tmp_path, 8, "python3", "-m", "tbench2_env.server.app")

    with pytest.raises(preflight.CapacityError, match=r"found \[7, 8\]"):
        preflight._server_pid(None, proc_root=tmp_path)


def test_cli_defaults_to_full_wave_capacity(monkeypatch, capsys):
    observed = {}

    def fake_probe(*, server_pid, required_concurrency):
        observed.update(
            server_pid=server_pid,
            required_concurrency=required_concurrency,
        )
        return {"ok": True}

    monkeypatch.setattr(preflight, "probe", fake_probe)

    assert preflight.main([]) == 0
    assert observed == {
        "server_pid": None,
        "required_concurrency": 304,
    }
    assert json.loads(capsys.readouterr().out) == {"ok": True}
