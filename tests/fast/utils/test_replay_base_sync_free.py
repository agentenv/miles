"""R3 replay without host syncs: same routes as the guarded fill and blocking copies (CPU)."""

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.utils import replay_base, sync_free
from miles.utils.replay_base import BaseReplayManager, Replay

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


class _FakeReplay:
    def __init__(self, top_indices):
        self.top_indices = top_indices
        self.staged_forward_ids = None

    def pop_forward(self):
        return self.top_indices

    pop_backward = pop_forward


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices


def _replayed(top_indices, scores, *, stage, return_probs, switch, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = stage
    manager.set_current(_FakeReplay(top_indices.clone()))
    return manager.get_topk_fn(_topk, return_probs=return_probs)(scores, top_indices.shape[1])


def _routes(num_tokens, num_experts, topk, *, padding_rows, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    top = torch.rand(num_tokens, num_experts, generator=gen).argsort(dim=1)[:, :topk].to(dtype)
    if padding_rows:
        top[torch.randperm(num_tokens, generator=gen)[:padding_rows]] = -1
    return top, torch.rand(num_tokens, num_experts, generator=gen)


def _guarded_fill(top_indices, num_experts):
    # the fill before upstream made it branchless (#4218): only when a padding row exists
    all_invalid = (top_indices == -1).all(dim=-1)
    if all_invalid.any():
        ar = torch.arange(top_indices.shape[1], dtype=top_indices.dtype) % num_experts
        top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)
    return top_indices


@pytest.mark.parametrize("stage", ["replay_forward", "replay_backward"])
@pytest.mark.parametrize("return_probs", [False, True])
@pytest.mark.parametrize("padding_rows", [0, 1, 9])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("switch", ["1", "0"])
def test_replayed_routes_match_the_guarded_fill(stage, return_probs, padding_rows, dtype, switch, monkeypatch):
    top, scores = _routes(64, 16, 6, padding_rows=padding_rows, dtype=dtype, seed=padding_rows)
    got = _replayed(top, scores, stage=stage, return_probs=return_probs, switch=switch, monkeypatch=monkeypatch)
    expected_top = _guarded_fill(top.clone(), scores.shape[1])
    expected = (scores.gather(1, expected_top), expected_top) if return_probs else (expected_top,)
    got = got if return_probs else (got,)
    for a, b in zip(got, expected, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b)


def test_fill_keeps_partial_padding_and_fills_padding_rows(monkeypatch):
    top = torch.tensor([[2, -1, -1], [-1, -1, -1], [4, 0, 1]], dtype=torch.int32)
    filled = _replayed(
        top, torch.rand(3, 2), stage="replay_forward", return_probs=False, switch="1", monkeypatch=monkeypatch
    )
    assert filled.tolist() == [[2, -1, -1], [0, 1, 0], [4, 0, 1]]
    assert filled.dtype == torch.int32


class _CopySpy:
    def __init__(self):
        self.calls = []

    def to(self, device, non_blocking=False):
        self.calls.append((device, non_blocking))
        return self


@pytest.mark.parametrize("switch,non_blocking", [("1", True), ("0", False)])
def test_pop_copies_without_blocking_when_on(switch, non_blocking, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: 3)
    replay = Replay()
    replay.top_indices_list = [_CopySpy(), _CopySpy()]
    assert replay.pop_forward() is replay.top_indices_list[0]
    assert replay.pop_backward() is replay.top_indices_list[0]
    assert replay.top_indices_list[0].calls == [(3, non_blocking), (3, non_blocking)]
