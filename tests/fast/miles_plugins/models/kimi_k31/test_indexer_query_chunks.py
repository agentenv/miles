"""The frozen K3.1 indexer scored in query blocks selects exactly the keys of one full call."""

import pytest
import torch

import miles.kernels.attention.dsa.indexer as indexer_mod
from miles.kernels.attention.dsa import causal_ranges
from miles.kernels.attention.dsa.topk import torch_dsa_topk

model = pytest.importorskip("miles_plugins.models.kimi_k31.model")


def _reference_write_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, out, clean_logits, config):
    """CPU stand-in for the indexer kernel: relu(q . k) weighted over heads, -inf outside [ks, ke)."""
    del clean_logits, config
    scores = torch.einsum("thd,sd->ths", q.float(), k.float()).relu()
    scores = (scores * weights[:, :, None]).sum(dim=1)
    cols = torch.arange(k.shape[0])
    inside = (cols[None, :] >= cu_seqlen_ks[:, None].long()) & (cols[None, :] < cu_seqlen_ke[:, None].long())
    out[:, : k.shape[0]] = torch.where(inside, scores, torch.tensor(float("-inf")))


@pytest.fixture(autouse=True)
def _cpu_indexer(monkeypatch):
    monkeypatch.setattr(indexer_mod, "_write_logits", _reference_write_logits)
    monkeypatch.setattr(indexer_mod, "_default_config", lambda q: None)
    # the backward-only autograd wrapper needs TileLang; the selection path never differentiates
    monkeypatch.setattr(indexer_mod, "indexer_topk_scores", lambda q, k, w, logits, idx: indexer_mod.gather_topk_scores(logits, idx))


def _inputs(cu, positions=None, heads=4, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = int(cu[-1])
    rows = total if positions is None else positions.numel()
    q = torch.randn(rows, heads, dim, generator=g)
    k = torch.randn(total, dim, generator=g)
    w = torch.rand(rows, heads, generator=g)
    starts, ends = causal_ranges(cu, positions)
    return q, k, w, starts.to(torch.int32), ends.to(torch.int32)


@pytest.mark.parametrize("chunk", [1, 7, 64, 100, 1000])
@pytest.mark.parametrize("topk", [8, 64])
def test_blocks_match_one_call_packed_sequences(chunk, topk):
    cu = torch.tensor([0, 37, 50, 211, 256], dtype=torch.int32)  # segment edges fall inside blocks
    q, k, w, ks, ke = _inputs(cu)
    whole = model.chunked_indexer_topk(q, k, w, ks, ke, topk, torch_dsa_topk, chunk=0)
    blocks = model.chunked_indexer_topk(q, k, w, ks, ke, topk, torch_dsa_topk, chunk=chunk)
    assert torch.equal(whole, blocks)
    # causal: no query selects a key at or past itself, nor before its segment
    valid = blocks >= 0
    assert (blocks[valid] < ke.repeat_interleave(valid.sum(1)).to(blocks.dtype)).all()
    assert (blocks[valid] >= ks.repeat_interleave(valid.sum(1)).to(blocks.dtype)).all()


@pytest.mark.parametrize("chunk", [5, 33])
def test_blocks_match_one_call_on_a_zigzag_cp_shard(chunk):
    # CP 2 zigzag rank 0 of one 128-token sequence holds chunks 0 and 3 of 4: rows are not contiguous
    cu = torch.tensor([0, 128], dtype=torch.int32)
    positions = torch.cat([torch.arange(0, 32), torch.arange(96, 128)])
    q, k, w, ks, ke = _inputs(cu, positions)
    whole = model.chunked_indexer_topk(q, k, w, ks, ke, 16, torch_dsa_topk, chunk=0)
    blocks = model.chunked_indexer_topk(q, k, w, ks, ke, 16, torch_dsa_topk, chunk=chunk)
    assert torch.equal(whole, blocks)
    assert int(blocks.max()) < 128 and (ke == positions + 1).all()


def test_default_chunk_is_8192():
    assert model.INDEXER_QUERY_CHUNK == 8192
