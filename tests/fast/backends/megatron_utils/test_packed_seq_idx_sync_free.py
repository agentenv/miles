"""seq_idx built from the host copy of cu_seqlens equals PackedSeqParams.__post_init__'s (CPU)."""

import dataclasses
import logging
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
packed_seq_params_mod = pytest.importorskip("megatron.core.packed_seq_params")

from miles.backends.megatron_utils import parallel  # noqa: E402
from miles.utils import sync_free  # noqa: E402

ARGS = Namespace(qkv_format="thd")


@pytest.fixture(autouse=True)
def _fresh_source_check():
    parallel._host_seq_idx_matches_megatron.cache_clear()
    yield
    parallel._host_seq_idx_matches_megatron.cache_clear()


def _batch(lengths, pad):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if pad:
        cu.append(cu[-1] + pad)
    return {
        "tokens": torch.zeros(1, cu[-1], dtype=torch.long),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int),
        "cu_seqlens_host": tuple(cu),
        "max_seqlen": max(lengths),
    }


def _fields(params):
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}


def _assert_same_params(a, b):
    fa, fb = _fields(a), _fields(b)
    assert fa.keys() == fb.keys()
    for name in fa:
        x, y = fa[name], fb[name]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), name
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), name
        else:
            assert x == y, name


def test_megatron_source_is_the_known_one():
    assert parallel._host_seq_idx_matches_megatron()


PACKS = [([3, 2, 4], 3), ([5, 1], 0), ([8192], 0), ([1], 127), ([700, 1, 1, 2000, 333], 64)]


@pytest.mark.parametrize("lengths,pad", PACKS)
def test_host_seq_idx_gives_the_same_params(lengths, pad, monkeypatch):
    monkeypatch.setattr(sync_free, "_ENABLED", True)
    new = parallel.get_packed_seq_params(_batch(lengths, pad), ARGS)
    monkeypatch.setattr(sync_free, "_ENABLED", False)
    old = parallel.get_packed_seq_params(_batch(lengths, pad), ARGS)
    assert new.total_tokens == sum(lengths) + pad and new.seq_idx is not None
    _assert_same_params(new, old)


@pytest.mark.parametrize("lengths,pad", PACKS)
def test_thd_seq_idx_equals_post_init(lengths, pad):
    batch = _batch(lengths, pad)
    expected = packed_seq_params_mod.PackedSeqParams(
        cu_seqlens_q=batch["cu_seqlens"], qkv_format="thd", total_tokens=sum(lengths) + pad
    ).seq_idx
    got = parallel.thd_seq_idx(torch.tensor(batch["cu_seqlens_host"], dtype=torch.int), sum(lengths) + pad)
    assert got.dtype == expected.dtype == torch.int32 and torch.equal(got, expected)


def test_thd_seq_idx_clamps_like_post_init():
    # total_tokens below the last boundary: the trailing length clamps to 0
    cu = torch.tensor([0, 4, 10], dtype=torch.int)
    expected = packed_seq_params_mod.PackedSeqParams(cu_seqlens_q=cu, qkv_format="thd", total_tokens=8).seq_idx
    assert torch.equal(parallel.thd_seq_idx(cu, 8), expected)


def test_unknown_post_init_falls_back_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(sync_free, "_ENABLED", True)
    monkeypatch.setattr(parallel.sync_free, "source_sha256", lambda _fn: "0" * 64)
    calls = []
    monkeypatch.setattr(parallel, "thd_seq_idx", lambda *a: calls.append(a))
    with caplog.at_level(logging.WARNING, logger=parallel.logger.name):
        params = parallel.get_packed_seq_params(_batch([3, 2], 1), ARGS)
    assert calls == []
    assert params.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2]]
    assert "does not match the Megatron source" in caplog.text


@pytest.mark.parametrize("cp_size", [2, 4])
def test_context_parallel_gives_the_same_params(cp_size, monkeypatch):
    """With CP the boundaries describe the global stream; both paths still derive seq_idx from them."""
    batch = _batch([3 * cp_size, 2 * cp_size], cp_size)
    monkeypatch.setattr(sync_free, "_ENABLED", True)
    new = parallel.get_packed_seq_params(dict(batch), ARGS)
    monkeypatch.setattr(sync_free, "_ENABLED", False)
    old = parallel.get_packed_seq_params(dict(batch), ARGS)
    _assert_same_params(new, old)
