"""Host-side capacity gate for the full Terminal-Bench 2.1 rollout wave.

Run this inside the trusted shared-server container after starting its pinned
OpenEnv daemon.  It validates the live daemon process, resource limits, task
inventory, and all Docker images without creating an episode or reserving a
GPU.  The behavioral reset/exec/evaluate probe remains a separate gate.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote


class CapacityError(RuntimeError):
    pass


DEFAULT_REQUIRED_CONCURRENCY = 304
_MANAGED_SERVER_SCRIPT = b"managed_tbench21_server.py"
_UPSTREAM_SERVER_MODULE = b"tbench2_env.server.app"


def _basename(argument: bytes) -> bytes:
    return argument.rsplit(b"/", 1)[-1]


def _is_python_executable(argument: bytes) -> bool:
    name = _basename(argument)
    if name in {b"python", b"python3"}:
        return True
    if not name.startswith(b"python"):
        return False
    suffix = name.removeprefix(b"python")
    return bool(suffix) and all(part.isdigit() for part in suffix.split(b"."))


def _python_script_argument(argv: list[bytes]) -> bytes | None:
    """Return Python's script operand, excluding ``-c`` and ``-m`` commands."""
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == b"--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if argument in {b"-c", b"-m"}:
            return None
        if argument in {b"-W", b"-X"}:
            index += 2
            continue
        if argument.startswith(b"-"):
            index += 1
            continue
        return argument
    return None


def _is_server_argv(argv: list[bytes]) -> bool:
    if not argv:
        return False

    executable = _basename(argv[0])
    if executable == _MANAGED_SERVER_SCRIPT:
        return True
    if _is_python_executable(argv[0]):
        script = _python_script_argument(argv)
        if script is not None and _basename(script) == _MANAGED_SERVER_SCRIPT:
            return True
        for index, argument in enumerate(argv[1:-1], start=1):
            if argument != b"-m":
                continue
            module = argv[index + 1]
            if module == _UPSTREAM_SERVER_MODULE:
                return True
            if module == b"uvicorn":
                return index + 2 < len(argv) and argv[index + 2].partition(b":")[0] == _UPSTREAM_SERVER_MODULE
        return False
    if executable == b"uvicorn":
        return len(argv) > 1 and argv[1].partition(b":")[0] == _UPSTREAM_SERVER_MODULE
    return False


def _server_pid(explicit: int | None, *, proc_root: Path = Path("/proc")) -> int:
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = sorted(int(path.name) for path in proc_root.iterdir() if path.name.isdecimal())
    matches: list[int] = []
    for pid in candidates:
        try:
            raw_command = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [argument for argument in raw_command.split(b"\0") if argument]
        if _is_server_argv(argv):
            matches.append(pid)
    if len(matches) != 1:
        raise CapacityError(f"expected exactly one live Terminal-Bench server process, found {matches}")
    return matches[0]


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError as error:
        raise CapacityError("cannot read the live server environment") from error
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8")] = value.decode("utf-8")
    return values


def _open_file_limit(pid: int) -> int:
    try:
        lines = (Path("/proc") / str(pid) / "limits").read_text().splitlines()
    except OSError as error:
        raise CapacityError("cannot read the live server resource limits") from error
    for line in lines:
        if line.startswith("Max open files"):
            fields = line.split()
            if len(fields) < 4 or not fields[3].isdecimal():
                break
            return int(fields[3])
    raise CapacityError("cannot resolve the server's soft open-file limit")


def _task_images(tasks_dir: Path) -> dict[str, str]:
    if not tasks_dir.is_dir() or tasks_dir.is_symlink():
        raise CapacityError("Terminal-Bench tasks directory is not a real directory")
    values: dict[str, str] = {}
    for task_dir in sorted(tasks_dir.iterdir(), key=lambda path: path.name):
        if not task_dir.is_dir() or task_dir.is_symlink():
            continue
        source = task_dir / "task.toml"
        try:
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
            image = raw["environment"]["docker_image"]
        except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
            raise CapacityError(f"task {task_dir.name!r} has invalid image metadata") from error
        if not isinstance(image, str) or not image:
            raise CapacityError(f"task {task_dir.name!r} has no Docker image")
        values[task_dir.name] = image
    if len(values) != 89:
        raise CapacityError(f"expected 89 Terminal-Bench tasks, found {len(values)}")
    return values


def _inspect_images(images: dict[str, str]) -> None:
    references = sorted(set(images.values()))
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", *references],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _inspect_images_via_socket(references)
        return
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise CapacityError(f"one or more Terminal-Bench images are absent: {detail}")


class _DockerSocketConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def _inspect_images_via_socket(references: list[str]) -> None:
    missing: list[str] = []
    for reference in references:
        connection = _DockerSocketConnection("localhost", timeout=30)
        try:
            connection.request("GET", f"/images/{quote(reference, safe='')}/json")
            response = connection.getresponse()
            response.read()
            if response.status != 200:
                missing.append(reference)
        except OSError as error:
            raise CapacityError("cannot query Docker image inventory") from error
        finally:
            connection.close()
    if missing:
        raise CapacityError("one or more Terminal-Bench images are absent: " + ", ".join(missing))


def probe(*, server_pid: int | None, required_concurrency: int) -> dict[str, object]:
    if required_concurrency < 1:
        raise CapacityError("required concurrency must be positive")
    pid = _server_pid(server_pid)
    environment = _process_environment(pid)
    try:
        configured = int(environment["MAX_CONCURRENT_ENVS"])
    except (KeyError, ValueError) as error:
        raise CapacityError("live server has no valid MAX_CONCURRENT_ENVS") from error
    if configured < required_concurrency:
        raise CapacityError(f"server capacity {configured} is below required {required_concurrency}")
    if environment.get("TB2_MODE") != "docker":
        raise CapacityError("full Terminal-Bench requires TB2_MODE=docker")
    if environment.get("TB2_WITHHOLD_TESTS") != "1":
        raise CapacityError("Terminal-Bench verifier assets are not withheld")
    tasks_value = environment.get("TB2_TASKS_DIR")
    if not tasks_value:
        raise CapacityError("live server has no TB2_TASKS_DIR")
    tasks_dir = Path(tasks_value).resolve()
    images = _task_images(tasks_dir)
    _inspect_images(images)
    nofile = _open_file_limit(pid)
    if nofile < 65536:
        raise CapacityError(f"server open-file limit {nofile} is below 65536")
    docker_socket = Path("/var/run/docker.sock")
    if not docker_socket.exists() or not docker_socket.is_socket():
        raise CapacityError("shared server cannot reach the Docker daemon socket")
    cpu_count = os.cpu_count() or 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return {
        "ok": True,
        "server_pid": pid,
        "configured_concurrency": configured,
        "required_concurrency": required_concurrency,
        "open_file_limit": nofile,
        "task_count": len(images),
        "unique_image_count": len(set(images.values())),
        "cpu_count": cpu_count,
        "available_memory_bytes": page_size * available_pages,
        "tasks_dir": str(tasks_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int)
    parser.add_argument(
        "--required-concurrency",
        type=int,
        default=DEFAULT_REQUIRED_CONCURRENCY,
    )
    args = parser.parse_args(argv)
    try:
        result = probe(
            server_pid=args.server_pid,
            required_concurrency=args.required_concurrency,
        )
    except Exception as error:
        print(
            f"TB2.1 CAPACITY PREFLIGHT FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
