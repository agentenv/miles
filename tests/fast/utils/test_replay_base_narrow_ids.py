"""R3 ids shipped narrow (int16) are widened to int32 on the device: same routes as int32 ids (CPU)."""

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.utils import replay_base, sync_free
from miles.utils.replay_base import BaseReplayManager, Replay

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


class _FakeReplay:
    def __init__(self, top_indices):
        self.top_indices_list = [top_indices]
        self.forward_index = self.backward_index = 0
        self.staged_forward_ids = None

    pop_forward = Replay.pop_forward
    pop_backward = Replay.pop_backward


def _routes(num_tokens, num_experts, topk, *, padding_rows, seed):
    gen = torch.Generator().manual_seed(seed)
    top = torch.rand(num_tokens, num_experts, generator=gen).argsort(dim=1)[:, :topk].to(torch.int32)
    if padding_rows:
        top[torch.randperm(num_tokens, generator=gen)[:padding_rows]] = -1
    return top, torch.rand(num_tokens, num_experts, generator=gen)


def _replayed(top_indices, scores, *, stage, return_probs, monkeypatch):
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: "cpu")
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = stage
    manager.set_current(_FakeReplay(top_indices))
    return manager.get_topk_fn(lambda s, k: torch.topk(s, k, dim=1).indices, return_probs=return_probs)(
        scores, top_indices.shape[1]
    )


@pytest.mark.parametrize("stage", ["replay_forward", "replay_backward"])
@pytest.mark.parametrize("return_probs", [False, True])
@pytest.mark.parametrize("padding_rows", [0, 1, 9])
@pytest.mark.parametrize("switch", ["1", "0"])
def test_int16_ids_replay_like_int32_ids(stage, return_probs, padding_rows, switch, monkeypatch):
    monkeypatch.setattr(sync_free, "_ENABLED", switch == "1")
    top, scores = _routes(64, 384, 16, padding_rows=padding_rows, seed=padding_rows)
    if return_probs and not _gather_takes_int32():
        pytest.skip("this torch's CPU gather takes int64 indices only")
    want = _replayed(top.clone(), scores, stage=stage, return_probs=return_probs, monkeypatch=monkeypatch)
    got = _replayed(top.to(torch.int16), scores, stage=stage, return_probs=return_probs, monkeypatch=monkeypatch)
    want, got = (want, got) if return_probs else ((want,), (got,))
    for a, b in zip(got, want, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b)


def _gather_takes_int32() -> bool:
    try:
        torch.zeros(1, 2).gather(1, torch.zeros(1, 1, dtype=torch.int32))
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
def test_narrow_ids_widen_to_int32(dtype, monkeypatch):
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: "cpu")
    ids = torch.tensor([[0, 5, 7], [1, 2, 3]], dtype=dtype)
    if dtype is not torch.uint8:
        ids[1] = -1
    out = replay_base._to_current_device(ids)
    assert out.dtype == torch.int32
    assert torch.equal(out, ids.to(torch.int32))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_wide_ids_are_returned_as_copied(dtype, monkeypatch):
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: "cpu")
    ids = torch.tensor([[0, 5, -1]], dtype=dtype)
    out = replay_base._to_current_device(ids)
    assert out is ids  # .to() on the same device returns the tensor itself: no extra copy or cast
    assert replay_base.REPLAY_IDS_WIDENED_ON_DEVICE is True
