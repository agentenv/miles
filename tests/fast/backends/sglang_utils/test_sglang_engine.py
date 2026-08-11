import time
from types import SimpleNamespace

import pytest
import requests


def test_lora_disk_mode_enables_sglang_weight_disk_backup(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine

    args = SimpleNamespace(
        rollout_num_gpus_per_engine=8,
        num_gpus_per_node=8,
        hf_checkpoint="/model",
        seed=1,
        offload_rollout=True,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=8,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=64,
        target_modules=["q_proj"],
        lora_adapter_path=None,
        lora_base_disk_reload=True,
    )
    monkeypatch.setattr(sglang_engine, "get_base_gpu_id", lambda *_: 0)
    monkeypatch.setattr(sglang_engine, "is_multi_lora_enabled", lambda *_: False)
    monkeypatch.setattr(sglang_engine, "is_lora_enabled", lambda *_: True)
    monkeypatch.setattr(
        sglang_engine, "lora_base_cpu_backup_enabled", lambda *_: False
    )
    monkeypatch.setattr(
        sglang_engine, "convert_target_modules_to_hf", lambda value: value
    )

    server_args, _ = sglang_engine._compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=1236,
    )

    assert server_args["enable_weights_disk_backup"] is True
    assert "enable_weights_cpu_backup" not in server_args


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )
