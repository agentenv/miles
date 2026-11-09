"""THD cu_seqlens / max_seqlen from the host boundaries equal the device-read values (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.utils import sync_free


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))


def _rollout(lengths):
    return {
        "tokens": [torch.arange(1, n + 1, dtype=torch.long) for n in lengths],
        "loss_masks": [torch.ones(max(1, n // 2), dtype=torch.int) for n in lengths],
        "total_lengths": list(lengths),
        "response_lengths": [max(1, n // 2) for n in lengths],
    }


def _get_batch(lengths, *, cp_size, cp_rank, allgather_cp, switch, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    rollout = _rollout(lengths)
    return data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
        list(rollout),
        pad_multiplier=128,
        qkv_format="thd",
        allgather_cp=allgather_cp,
    )


def _device_read_max_seqlen(cu_list, cp_size):
    # the expression get_batch used before (there on the device, with .item())
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int) * cp_size
    return cu_seqlens, (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()


LENGTHS = [[7, 13, 5, 9, 4], [4096], [1, 1, 1], [8191, 1, 300]]


@pytest.mark.parametrize("lengths", LENGTHS, ids=range(len(LENGTHS)))
@pytest.mark.parametrize("cp_size,cp_rank,allgather_cp", [(1, 0, False), (2, 1, False), (2, 0, True)])
def test_get_batch_same_with_and_without_host_syncs(lengths, cp_size, cp_rank, allgather_cp, monkeypatch):
    kwargs = dict(cp_size=cp_size, cp_rank=cp_rank, allgather_cp=allgather_cp, monkeypatch=monkeypatch)
    new = _get_batch(lengths, switch="1", **kwargs)
    old = _get_batch(lengths, switch="0", **kwargs)
    assert torch.equal(new["tokens"], old["tokens"])
    assert new["cu_seqlens"].dtype == old["cu_seqlens"].dtype == torch.int32
    assert torch.equal(new["cu_seqlens"], old["cu_seqlens"])
    assert type(new["max_seqlen"]) is type(old["max_seqlen"]) is int
    assert new["max_seqlen"] == old["max_seqlen"]
    assert new["cu_seqlens_host"] == old["cu_seqlens_host"] == tuple(new["cu_seqlens"].tolist())


@pytest.mark.parametrize("cu_list", [[0, 7, 20, 25, 34, 38, 128], [0, 4096], [0, 1, 2, 3, 128]])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("switch", ["1", "0"])
def test_thd_cu_seqlens_on_device_matches_the_device_read(cu_list, cp_size, switch, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    expected_cu, expected_max = _device_read_max_seqlen(cu_list, cp_size)
    host = tuple(boundary * cp_size for boundary in cu_list)
    cu_seqlens, max_seqlen = data_utils.thd_cu_seqlens_on_device(host, "cpu")
    assert torch.equal(cu_seqlens, expected_cu) and cu_seqlens.dtype == torch.int32
    assert max_seqlen == expected_max and type(max_seqlen) is int
