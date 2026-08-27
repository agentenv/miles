from __future__ import annotations

import shlex
import sys
from pathlib import Path


_OPENENV_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_OPENENV_ROOT))

import openenv_launch_common as launch_common  # noqa: E402


def _option_value(args: str, option: str) -> str:
    tokens = shlex.split(args)
    return tokens[tokens.index(option) + 1]


def test_agent_args_uses_safe_default_filter() -> None:
    args = launch_common.agent_args("qwen35")
    assert _option_value(args, "--dynamic-sampling-filter-path") == (
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"
    )


def test_agent_args_accepts_terminal_bench_verdict_filter() -> None:
    args = launch_common.agent_args(
        "qwen35",
        dynamic_sampling_filter_path="openenv_generate.check_terminal_bench_episode",
    )
    assert _option_value(args, "--dynamic-sampling-filter-path") == (
        "openenv_generate.check_terminal_bench_episode"
    )
