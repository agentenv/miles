from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "probes" / "run_sao_qwen35_08_value_pretrain.sh"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_launcher_requires_converter_manifest_digest() -> None:
    env = os.environ.copy()
    env.pop("VALUE_MANIFEST_SHA256", None)

    completed = _run(env)

    assert completed.returncode != 0
    assert "VALUE_MANIFEST_SHA256 is required" in completed.stderr


def test_launcher_rejects_malformed_manifest_digest() -> None:
    env = os.environ.copy()
    env["VALUE_MANIFEST_SHA256"] = "A" * 64

    completed = _run(env)

    assert completed.returncode != 0
    assert "must be a lowercase SHA-256 digest" in completed.stderr


def test_launcher_rejects_digest_that_does_not_match_manifest(
    tmp_path: Path,
) -> None:
    miles = tmp_path / "miles"
    megatron = tmp_path / "Megatron-LM"
    model = tmp_path / "model"
    checkpoint = tmp_path / "actor"
    for directory in (miles, megatron, model, checkpoint):
        directory.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema":"miles.value-pretrain.v1"}\n', encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "MILES_ROOT": str(miles),
            "MEGATRON_ROOT": str(megatron),
            "MODEL_ROOT": str(model),
            "BASE_CHECKPOINT": str(checkpoint),
            "VALUE_CHECKPOINT": str(tmp_path / "critic"),
            "VALUE_PLAN": str(tmp_path / "critic-plan.json"),
            "VALUE_MANIFEST": str(manifest),
            "VALUE_MANIFEST_SHA256": "0" * 64,
        }
    )
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() != "0" * 64

    completed = _run(env)

    assert completed.returncode != 0
    assert "does not match VALUE_MANIFEST" in completed.stderr
