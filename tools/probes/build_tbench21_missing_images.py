#!/usr/bin/env python3
"""Build missing Terminal-Bench 2.1 task images from their pinned Dockerfiles."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


_IMAGE_RE = re.compile(r'^\s*docker_image\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def _image_for_task(task_dir: Path) -> str:
    source = task_dir / "task.toml"
    match = _IMAGE_RE.search(source.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{source}: missing environment.docker_image")
    return match.group(1)


def _inspect(image: str) -> dict[str, object] | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"docker returned malformed inspection data for {image}")
    record = payload[0]
    return {
        "id": record.get("Id"),
        "repo_digests": sorted(record.get("RepoDigests") or []),
        "size": record.get("Size"),
    }


def _build(task: str, task_dir: Path, image: str, log_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    log_path = log_dir / f"{task}.log"
    with log_path.open("wb") as log:
        result = subprocess.run(
            [
                "docker",
                "build",
                "--progress=plain",
                "--tag",
                image,
                str(task_dir / "environment"),
            ],
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    inspection = _inspect(image) if result.returncode == 0 else None
    return {
        "task": task,
        "image": image,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "inspection": inspection,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    tasks_dir = args.tasks_dir.resolve()
    if not tasks_dir.is_dir():
        raise FileNotFoundError(tasks_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    tasks: list[tuple[str, Path, str]] = []
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        if not (task_dir / "task.toml").is_file():
            continue
        image = _image_for_task(task_dir)
        if not (task_dir / "environment" / "Dockerfile").is_file():
            raise FileNotFoundError(f"{task_dir}: missing environment/Dockerfile")
        tasks.append((task_dir.name, task_dir, image))
    if not tasks:
        raise ValueError("no Terminal-Bench tasks found")
    images = [image for _, _, image in tasks]
    if len(images) != len(set(images)):
        raise ValueError("Terminal-Bench tasks do not have unique image references")

    source_digest = hashlib.sha256()
    for task, task_dir, image in tasks:
        source_digest.update(task.encode())
        source_digest.update(b"\0")
        source_digest.update(image.encode())
        source_digest.update(b"\0")
        source_digest.update((task_dir / "task.toml").read_bytes())
        source_digest.update((task_dir / "environment" / "Dockerfile").read_bytes())

    present: dict[str, dict[str, object]] = {}
    missing: list[tuple[str, Path, str]] = []
    for task, task_dir, image in tasks:
        inspection = _inspect(image)
        if inspection is None:
            missing.append((task, task_dir, image))
        else:
            present[task] = {"image": image, "inspection": inspection}

    started_at = time.time()
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_build, task, task_dir, image, log_dir): task
            for task, task_dir, image in missing
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            state = "ok" if result["exit_code"] == 0 else "failed"
            print(
                f"[{len(results)}/{len(missing)}] {result['task']} {state} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
            _atomic_json(
                args.output_dir / "progress.json",
                {
                    "schema": "miles.tbench21-image-build-progress.v1",
                    "tasks": len(tasks),
                    "initially_present": len(present),
                    "initially_missing": len(missing),
                    "completed": sorted(results, key=lambda item: str(item["task"])),
                },
            )

    results.sort(key=lambda item: str(item["task"]))
    failures = [result for result in results if result["exit_code"] != 0]
    manifest = {
        "schema": "miles.tbench21-image-build.v1",
        "tasks": len(tasks),
        "source_contract_sha256": source_digest.hexdigest(),
        "workers": args.workers,
        "started_at_unix": started_at,
        "elapsed_seconds": time.time() - started_at,
        "initially_present": present,
        "builds": results,
        "failures": [result["task"] for result in failures],
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    if failures:
        raise SystemExit(f"{len(failures)} Terminal-Bench image builds failed")
    print(f"TBENCH21_IMAGES_READY tasks={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
