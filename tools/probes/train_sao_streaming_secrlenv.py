"""Full-parameter SAO/SecRLEnv entrypoint for dual streaming DiLoCo.

This is intentionally separate from ``train_sao_secrlenv.py``.  The latter
remains the proven centralized path; this entry requires a second hash-bound
runtime contract and installs Yeto's actor/critic streaming synchronizer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tools.probes.train_sao_secrlenv import bind_context, load_context


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sao-secrlenv-context", required=True)
    parser.add_argument("--sao-secrlenv-context-sha256", required=True)
    parser.add_argument("--sao-streaming-context", required=True)
    parser.add_argument("--sao-streaming-context-sha256", required=True)
    context_args, miles_argv = parser.parse_known_args(raw_argv)

    secrlenv_context = load_context(
        context_args.sao_secrlenv_context,
        context_args.sao_secrlenv_context_sha256,
    )
    from yeto.rl.sao_streaming_runtime import (
        load_sao_streaming_runtime,
        sao_streaming_sync_factory_path,
    )

    streaming_runtime = load_sao_streaming_runtime(
        context_args.sao_streaming_context,
        expected_sha256=context_args.sao_streaming_context_sha256,
        expected_sao_context_sha256=context_args.sao_secrlenv_context_sha256,
    )
    sync_flag = "--external-policy-sync-path"
    if any(
        token == sync_flag or token.startswith(f"{sync_flag}=")
        for token in miles_argv
    ):
        raise ValueError(
            "the SAO streaming entrypoint owns the external sync callback"
        )
    miles_argv.extend([sync_flag, sao_streaming_sync_factory_path()])
    if (
        getattr(streaming_runtime, "trajectory_evidence_kind", None)
        == "terminal-bench-2.1"
    ):
        from yeto.rl.tbench_direct_preflight import (
            preflight_tbench_codex_streaming,
        )

        preflight_tbench_codex_streaming(
            miles_argv,
            sao_context=secrlenv_context,
            streaming_runtime=streaming_runtime,
            miles_root=Path(__file__).resolve().parents[2],
        )

    previous = sys.argv
    try:
        sys.argv = [previous[0], *miles_argv]
        from miles.utils.arguments import parse_args

        args = parse_args()
    finally:
        sys.argv = previous

    bind_context(args, secrlenv_context)
    from yeto.rl.sao_streaming_runtime import bind_sao_streaming_runtime

    bind_sao_streaming_runtime(args, streaming_runtime)

    from miles.utils.tracking_utils.tracking import finish_tracking
    from train import train

    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()


if __name__ == "__main__":
    main()
