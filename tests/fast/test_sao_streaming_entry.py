from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from tools.probes import train_sao_streaming_secrlenv as entry


def test_streaming_entry_loads_both_contracts_and_binds_before_train(monkeypatch):
    calls = []
    args = SimpleNamespace()
    central_context = object()
    streaming_runtime = object()

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
        ("parse-miles", ("--num-rollout", "1")),
        ("bind-central",),
        ("bind-streaming",),
        ("train",),
        ("finish",),
    ]
