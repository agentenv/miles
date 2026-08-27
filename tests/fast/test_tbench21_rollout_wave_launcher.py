from __future__ import annotations

from pathlib import Path

from tools.probes import launch_tbench21_rollout_wave as launcher


def _argv(tmp_path: Path, *, phase: str) -> list[str]:
    return launcher._container_argv(
        phase=phase,
        island_id=3,
        name=f"tbench21-{phase}-island-3",
        miles_root=tmp_path / "miles",
        yeto_root=tmp_path / "yeto",
        model=tmp_path / "base-hf",
        checkpoint=tmp_path / "actor-checkpoint",
        plan_dir=tmp_path / "plan",
        output=tmp_path / "output",
        codex_dir=tmp_path / "codex",
        hmac_key=tmp_path / "reward.key",
        bridge_gateway="172.17.0.1",
    )


def test_eval_container_explicitly_requests_checkpoint_publication(
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path, phase="eval")

    assert "--serve-ref-checkpoint" in argv
    assert argv[argv.index("--hf-checkpoint") + 1] == "/root/model"
    assert argv[argv.index("--ref-load") + 1] == "/root/input-checkpoint"
    assert any(value.endswith("actor-checkpoint,dst=/root/input-checkpoint,readonly") for value in argv)


def test_baseline_container_keeps_direct_hf_rollout_mode(tmp_path: Path) -> None:
    argv = _argv(tmp_path, phase="baseline")

    assert "--serve-ref-checkpoint" not in argv
