"""Fused Triton SiTU-and-mul for Kimi K3 (``MILES_K3_SITU_TRITON=1``), optionally with the router-prob scale.

Reference (miles_plugins/models/kimi_k3/ops.py:situ_and_mul, eager, fp32 chain):

    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)

followed in Megatron TEGroupedMLP.bias_act_func by ``(out * permuted_probs).to(bf16)``.

The kernels below replay that chain op for op so forward and backward are bit-identical to eager:
* every transcendental the chain needs is a function of one BF16 input, so it is read from 65536-entry tables
  that ATen itself fills once per device (``_tables``): ``tanh(g / 4)``, ``sigmoid(g)``, ``tanh(l / 25)`` and
  ATen's ``tanh_backward(1, t)`` (its ``1 - t*t``, however nvcc contracted it). Recomputing them with
  libdevice did not reproduce ATen's fp32 values in every process: inside a Megatron/TE process d(gate)
  differed by one BF16 ulp wherever its two terms cancel (g = -1.21875), although the BF16 forward matched,
  so the kernels compute no transcendental themselves;
* ``t / scalar`` on CUDA is ``t * (1 / scalar)`` (ATen div_true with a CPU scalar), so /4 -> *0.25 and
  /25 -> *fp32(1/25) in backward;
* the rest is IEEE fp32 multiply / add in eager's order, launched with ``enable_fp_fusion=False``.
The backward reproduces autograd's per-op gradients (MulBackward, TanhBackward, SigmoidBackward,
DivBackward with a scalar, ToCopyBackward) and their dtype casts; d(probs) is reduced by the same torch
``sum_to_size`` autograd uses, so it can be bit-identical too.

Saves only the BF16 input (and probs) for backward instead of six fp32 [rows, N] intermediates.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

BETA = 4.0
LINEAR_BETA = 25.0
INV_BETA = float(np.float32(1.0) / np.float32(BETA))  # 0.25
INV_LINEAR_BETA = float(np.float32(1.0) / np.float32(LINEAR_BETA))  # fp32(1/25), as ATen's host reciprocal
_GATE_COLS = 4  # tanh(g/4), sigmoid(g), tanh_backward(1, tanh(g/4)), pad (one 16-byte row per input)
_LIN_COLS = 2  # tanh(l/25), tanh_backward(1, tanh(l/25))
_TABLES: dict = {}


def _tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [65536, 4] gate and [65536, 2] linear tables indexed by the input's BF16 bit pattern, filled by
    the same ATen ops (and Python-float scalars) the eager chain runs, on this device."""
    key = (device.type, device.index)
    if key not in _TABLES:
        bits = torch.arange(65536, dtype=torch.int32, device=device)
        v = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16).float()
        t1 = torch.tanh(v / BETA)
        t2 = torch.tanh(v / LINEAR_BETA)
        one = torch.ones_like(v)
        gate = torch.stack(
            [t1, torch.sigmoid(v), torch.ops.aten.tanh_backward(one, t1), torch.zeros_like(v)], dim=1
        ).contiguous()
        lin = torch.stack([t2, torch.ops.aten.tanh_backward(one, t2)], dim=1).contiguous()
        _TABLES[key] = (gate, lin)
    return _TABLES[key]


@triton.jit
def _index(v):
    """BF16 values -> int64 table row (their bit pattern as unsigned 16 bit)."""
    return v.to(tl.int16, bitcast=True).to(tl.int32).to(tl.int64) & 0xFFFF


@triton.jit
def _situ_fwd_kernel(
    x_ptr,
    p_ptr,
    gate_ptr,
    lin_ptr,
    out_ptr,
    total,
    N: tl.constexpr,
    BETA: tl.constexpr,
    LBETA: tl.constexpr,
    HAS_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    row = offs // N
    col = offs - row * N
    base = row * (2 * N) + col
    ig = _index(tl.load(x_ptr + base, mask=mask, other=0.0))
    il = _index(tl.load(x_ptr + base + N, mask=mask, other=0.0))
    t1 = tl.load(gate_ptr + ig * 4, mask=mask, other=0.0)
    s = tl.load(gate_ptr + ig * 4 + 1, mask=mask, other=0.0)
    t2 = tl.load(lin_ptr + il * 2, mask=mask, other=0.0)
    a = t1 * BETA
    go = a * s
    lo = t2 * LBETA
    y = (go * lo).to(tl.bfloat16)
    if HAS_P:
        p = tl.load(p_ptr + row, mask=mask, other=0.0).to(tl.float32)
        y = (y.to(tl.float32) * p).to(tl.bfloat16)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _situ_bwd_kernel(
    x_ptr,
    p_ptr,
    gate_ptr,
    lin_ptr,
    dy_ptr,
    dx_ptr,
    y_ptr,
    total,
    N: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    LBETA: tl.constexpr,
    INV_LBETA: tl.constexpr,
    HAS_P: tl.constexpr,
    WRITE_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    row = offs // N
    col = offs - row * N
    base = row * (2 * N) + col
    ig = _index(tl.load(x_ptr + base, mask=mask, other=0.0))
    il = _index(tl.load(x_ptr + base + N, mask=mask, other=0.0))
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_P:
        p = tl.load(p_ptr + row, mask=mask, other=0.0).to(tl.float32)
        # MulBackward0 grad_self = grad * probs (fp32), cast to the bf16 input dtype; ToCopyBackward -> fp32
        gy = (dy * p).to(tl.bfloat16).to(tl.float32)
    else:
        gy = dy
    t1 = tl.load(gate_ptr + ig * 4, mask=mask, other=0.0)
    s = tl.load(gate_ptr + ig * 4 + 1, mask=mask, other=0.0)
    u1 = tl.load(gate_ptr + ig * 4 + 2, mask=mask, other=0.0)
    t2 = tl.load(lin_ptr + il * 2, mask=mask, other=0.0)
    u2 = tl.load(lin_ptr + il * 2 + 1, mask=mask, other=0.0)
    a = t1 * BETA
    go = a * s
    lo = t2 * LBETA
    d_go = gy * lo
    d_lo = gy * go
    d_t2 = d_lo * LBETA
    d_l1 = d_t2 * u2  # tanh_backward(d_t2, t2) = d_t2 * (1 - t2*t2)
    d_lin = d_l1 * INV_LBETA
    d_a = d_go * s
    d_s = d_go * a
    d_g_sig = d_s * (1.0 - s) * s  # sigmoid_backward a * (1 - b) * b
    d_t1 = d_a * BETA
    d_g1 = d_t1 * u1
    d_g = d_g1 * INV_BETA + d_g_sig
    tl.store(dx_ptr + base, d_g.to(tl.bfloat16), mask=mask)
    tl.store(dx_ptr + base + N, d_lin.to(tl.bfloat16), mask=mask)
    if WRITE_Y:
        tl.store(y_ptr + offs, (go * lo).to(tl.bfloat16), mask=mask)


_BLOCK = 1024


def _launch_fwd(x, probs):
    rows, two_n = x.shape
    n = two_n // 2
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=x.device)
    total = rows * n
    if total == 0:
        return out
    gate, lin = _tables(x.device)
    grid = (triton.cdiv(total, _BLOCK),)
    _situ_fwd_kernel[grid](
        x,
        probs if probs is not None else x,
        gate,
        lin,
        out,
        total,
        n,
        BETA,
        LINEAR_BETA,
        probs is not None,
        _BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


class _FusedSitu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.dim() == 2
        ctx.probs_shape = None if probs is None else probs.shape
        if probs is not None:
            probs = probs.reshape(-1).contiguous()
        ctx.save_for_backward(x, probs)
        return _launch_fwd(x, probs)

    @staticmethod
    def backward(ctx, dy):
        x, probs = ctx.saved_tensors
        dy = dy.contiguous()
        rows, two_n = x.shape
        n = two_n // 2
        dx = torch.empty_like(x)
        need_dp = probs is not None and ctx.needs_input_grad[1]
        y = torch.empty(rows, n, dtype=torch.bfloat16, device=x.device) if need_dp else dx
        total = rows * n
        if total:
            gate, lin = _tables(x.device)
            grid = (triton.cdiv(total, _BLOCK),)
            _situ_bwd_kernel[grid](
                x,
                probs if probs is not None else x,
                gate,
                lin,
                dy,
                dx,
                y,
                total,
                n,
                BETA,
                INV_BETA,
                LINEAR_BETA,
                INV_LINEAR_BETA,
                probs is not None,
                need_dp,
                _BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
        dp = None
        if need_dp:
            promoted = torch.result_type(y, probs)
            dp = torch.mul(dy.to(promoted), y).sum_to_size(rows, 1).reshape(ctx.probs_shape)
        return dx, dp


def fused_situ_and_mul(x: torch.Tensor, probs: torch.Tensor | None = None) -> torch.Tensor:
    """x [rows, 2N] bf16 -> [rows, N] bf16; with probs [rows] or [rows, 1], also scales by them."""
    return _FusedSitu.apply(x, probs)
