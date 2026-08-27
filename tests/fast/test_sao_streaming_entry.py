from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.probes import train_sao_streaming_secrlenv as entry


SYNC_FACTORY = "yeto.rl.miles_sao_streaming.create_miles_sao_streaming_sync"


def test_streaming_entry_loads_both_contracts_and_binds_before_train(monkeypatch):
    calls = []
    args = SimpleNamespace()
    central_context = object()
    streaming_runtime = SimpleNamespace(
        trajectory_evidence_kind="terminal-bench-2.1"
    )

    def load_context(path, digest):
        calls.append(("load-central", path, digest))
        return central_context

    def bind_context(bound_args, context):
        assert bound_args is args
        assert context is central_context
        calls.append(("bind-central",))

    runtime_module = ModuleType("yeto.rl.sao_streaming_runtime")

    def load_runtime(path, *, expected_sha256, expected_sao_context_sha256):
        calls.append(
            (
                "load-streaming",
                path,
                expected_sha256,
                expected_sao_context_sha256,
            )
        )
        return streaming_runtime

    def bind_runtime(bound_args, runtime):
        assert bound_args is args
        assert runtime is streaming_runtime
        calls.append(("bind-streaming",))

    runtime_module.load_sao_streaming_runtime = load_runtime
    runtime_module.bind_sao_streaming_runtime = bind_runtime
    runtime_module.sao_streaming_sync_factory_path = lambda: SYNC_FACTORY
    preflight_module = ModuleType("yeto.rl.tbench_direct_preflight")

    def preflight(argv, *, sao_context, streaming_runtime, miles_root):
        assert sao_context is central_context
        assert streaming_runtime is not None
        assert miles_root == Path(entry.__file__).resolve().parents[2]
        calls.append(("preflight", tuple(argv)))

    preflight_module.preflight_tbench_codex_streaming = preflight
    yeto_module = ModuleType("yeto")
    yeto_module.__path__ = []
    yeto_rl_module = ModuleType("yeto.rl")
    yeto_rl_module.__path__ = []

    arguments_module = ModuleType("miles.utils.arguments")

    def parse_args():
        calls.append(("parse-miles", tuple(sys.argv[1:])))
        return args

    arguments_module.parse_args = parse_args
    tracking_module = ModuleType("miles.utils.tracking_utils.tracking")
    tracking_module.finish_tracking = lambda: calls.append(("finish",))
    train_module = ModuleType("train")

    async def train(bound_args):
        assert bound_args is args
        calls.append(("train",))

    train_module.train = train

    monkeypatch.setattr(entry, "load_context", load_context)
    monkeypatch.setattr(entry, "bind_context", bind_context)
    monkeypatch.setitem(sys.modules, "yeto", yeto_module)
    monkeypatch.setitem(sys.modules, "yeto.rl", yeto_rl_module)
    monkeypatch.setitem(sys.modules, "yeto.rl.sao_streaming_runtime", runtime_module)
    monkeypatch.setitem(
        sys.modules, "yeto.rl.tbench_direct_preflight", preflight_module
    )
    monkeypatch.setitem(sys.modules, "miles.utils.arguments", arguments_module)
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.tracking_utils.tracking",
        tracking_module,
    )
    monkeypatch.setitem(sys.modules, "train", train_module)

    entry.main(
        [
            "--sao-secrlenv-context",
            "/run/central.json",
            "--sao-secrlenv-context-sha256",
            "a" * 64,
            "--sao-streaming-context",
            "/run/streaming.json",
            "--sao-streaming-context-sha256",
            "b" * 64,
            "--num-rollout",
            "1",
        ]
    )

    assert calls == [
        ("load-central", "/run/central.json", "a" * 64),
        ("load-streaming", "/run/streaming.json", "b" * 64, "a" * 64),
        (
            "preflight",
            (
                "--num-rollout",
                "1",
                "--external-policy-sync-path",
                SYNC_FACTORY,
            ),
        ),
        (
            "parse-miles",
            (
                "--num-rollout",
                "1",
                "--external-policy-sync-path",
                SYNC_FACTORY,
            ),
        ),
        ("bind-central",),
        ("bind-streaming",),
        ("train",),
        ("finish",),
    ]


def test_streaming_entry_rejects_caller_owned_sync_callback(monkeypatch):
    runtime_module = ModuleType("yeto.rl.sao_streaming_runtime")
    runtime_module.load_sao_streaming_runtime = lambda *_args, **_kwargs: object()
    runtime_module.sao_streaming_sync_factory_path = lambda: SYNC_FACTORY
    monkeypatch.setitem(
        sys.modules, "yeto.rl.sao_streaming_runtime", runtime_module
    )
    monkeypatch.setattr(entry, "load_context", lambda *_args: object())

    with pytest.raises(ValueError, match="entrypoint owns"):
        entry.main(
            [
                "--sao-secrlenv-context",
                "/run/central.json",
                "--sao-secrlenv-context-sha256",
                "a" * 64,
                "--sao-streaming-context",
                "/run/streaming.json",
                "--sao-streaming-context-sha256",
                "b" * 64,
                "--external-policy-sync-path=untrusted.factory",
            ]
        )
