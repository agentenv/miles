"""The attention residual aggregate scored in checkpointed token blocks matches one full call."""

import pytest
import torch

from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, attn_res_aggregate

HIDDEN = 64


def _modules(seed):
    torch.manual_seed(seed)
    score_proj = torch.nn.Linear(HIDDEN, 1, bias=False)
    score_norm = KimiRMSNorm(HIDDEN, eps=1e-6)
    output_norm = KimiRMSNorm(HIDDEN, eps=1e-6)
    with torch.no_grad():
        score_norm.weight.uniform_(0.5, 1.5)
        output_norm.weight.uniform_(0.5, 1.5)
    return score_proj, score_norm, output_norm


def _max_err(a, b):
    """(max abs error, max abs reference) over two same-shape tensors; zeros when empty."""
    if a.numel() == 0:
        return 0.0, 0.0
    return (a.float() - b.float()).abs().max().item(), a.float().abs().max().item()


def _run(tokens, blocks, chunk, dtype, seed=0):
    score_proj, score_norm, output_norm = _modules(seed)
    g = torch.Generator().manual_seed(seed + 1)
    prefix_sum = torch.randn(tokens, 1, HIDDEN, generator=g).to(dtype).requires_grad_()
    block_residual = torch.randn(tokens, 1, blocks, HIDDEN, generator=g).to(dtype).requires_grad_()
    out = attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, output_norm, chunk=chunk)
    grad_out = torch.randn(out.shape, generator=g).to(dtype)
    out.backward(grad_out)
    params = (score_proj.weight, score_norm.weight, output_norm.weight)
    return [out, prefix_sum.grad, block_residual.grad, *(p.grad for p in params)]


NAMES = ("out", "d_prefix_sum", "d_block_residual", "d_score_proj", "d_score_norm", "d_output_norm")


@pytest.mark.parametrize("blocks", [0, 1, 8])
@pytest.mark.parametrize("tokens,chunk", [(64, 16), (100, 32), (37, 5), (8, 1), (20, 64)])
def test_fp32_blocks_match_one_call(tokens, blocks, chunk):
    whole = _run(tokens, blocks, 0, torch.float32)
    parts = _run(tokens, blocks, chunk, torch.float32)
    for name, a, b in zip(NAMES, whole, parts):
        assert a.shape == b.shape, name
        err, scale = _max_err(a, b)
        print(f"fp32 T={tokens} blocks={blocks} chunk={chunk} {name} max_abs_err={err:.3e} max_ref={scale:.3e}")
        # the weight grads sum over tokens in another order; 1e-5 of the largest entry bounds that
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5 * scale, msg=name)


@pytest.mark.parametrize("blocks", [0, 8])
def test_bf16_blocks_match_one_call(blocks):
    whole = _run(100, blocks, 0, torch.bfloat16)
    parts = _run(100, blocks, 32, torch.bfloat16)
    for name, a, b in zip(NAMES, whole, parts):
        err, scale = _max_err(a, b)
        print(f"bf16 blocks={blocks} {name} max_abs_err={err:.3e} max_ref={scale:.3e}")
        # per-token math is unchanged in blocks; only the fp32 weight-grad sum over tokens reorders
        torch.testing.assert_close(b.float(), a.float(), rtol=1.6e-2, atol=1e-5 * scale, msg=name)
    assert torch.equal(whole[0], parts[0])


def _saved_bytes(chunk, tokens=256, blocks=8):
    score_proj, score_norm, output_norm = _modules(0)
    prefix_sum = torch.randn(tokens, 1, HIDDEN, dtype=torch.bfloat16, requires_grad=True)
    block_residual = torch.randn(tokens, 1, blocks, HIDDEN, dtype=torch.bfloat16, requires_grad=True)
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        key = (t.untyped_storage().data_ptr(), t.dtype)
        if key not in seen:
            seen.add(key)
            total += t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        attn_res_aggregate(prefix_sum, block_residual, score_proj, score_norm, output_norm, chunk=chunk)
    return total


def test_blocks_keep_fp32_rows_out_of_the_saved_activations():
    rows_fp32 = 256 * 9 * HIDDEN * 4
    whole, parts = _saved_bytes(0), _saved_bytes(32)
    print(f"saved for backward: one call {whole} B, 32-token blocks {parts} B, fp32 rows {rows_fp32} B")
    assert whole > 2 * rows_fp32
    assert parts < rows_fp32
