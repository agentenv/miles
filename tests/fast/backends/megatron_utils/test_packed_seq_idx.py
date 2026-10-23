"""THD packing gives Mamba layers a per-sample ``seq_idx`` (state restarts at each sample)."""

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core.packed_seq_params")

from miles.backends.megatron_utils import parallel  # noqa: E402


def _batch(lengths, pad):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if pad:
        cu.append(cu[-1] + pad)
    return {
        "tokens": torch.zeros(1, cu[-1], dtype=torch.long),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int),
        "max_seqlen": max(lengths),
    }


def test_thd_pack_without_cp_has_one_seq_idx_per_sample_and_pad():
    batch = _batch([3, 2, 4], pad=3)
    params = parallel.get_packed_seq_params(batch, Namespace(qkv_format="thd", context_parallel_size=1))
    assert params.total_tokens == 12
    assert params.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]]
    assert batch["packed_seq_params"] is params


def test_thd_pack_without_pad():
    params = parallel.get_packed_seq_params(_batch([5, 1], pad=0), Namespace(qkv_format="thd"))
    assert params.seq_idx.tolist() == [[0, 0, 0, 0, 0, 1]]


@pytest.mark.parametrize("args", [Namespace(qkv_format="thd", context_parallel_size=2),
                                  Namespace(qkv_format="thd", context_parallel_size=1, allgather_cp=True)])
def test_context_parallel_keeps_the_previous_params(args):
    params = parallel.get_packed_seq_params(_batch([3, 2], pad=1), args)
    assert params.total_tokens is None and params.seq_idx is None


def test_bshd_has_no_packed_params():
    assert parallel.get_packed_seq_params(_batch([3], pad=0), Namespace(qkv_format="bshd")) is None
