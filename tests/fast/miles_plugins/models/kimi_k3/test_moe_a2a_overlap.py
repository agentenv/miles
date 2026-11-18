"""``MILES_K3_MOE_A2A_OVERLAP``: Megatron's real ``MoELayer`` (alltoall dispatcher, shared experts, router) on gloo
EP ranks gives bit-identical outputs and gradients with the all-to-alls moved off the critical path. The test
covers the plain forward, the no-grad forward and a checkpointed (recomputed) forward. It also checks that the
backward runs the shared experts inside the combine all-to-all's window, and that the dispatch prefetches the
frozen MXFP4 experts' fc1 dequantization, which the grouped linear then uses (once).

CPU stand-ins: the experts are a TE-free grouped MLP whose fc1 is the production ``RedequantFrozenExperts``
mixin over packed MXFP4 rows; the dispatcher's DtoH step is its no-stream equivalent.
"""

import functools
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci

overlap = pytest.importorskip("miles_plugins.models.kimi_k3.moe_a2a_overlap")
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, mappings  # noqa: E402
from megatron.core.transformer.mlp import MLPSubmodules  # noqa: E402
from megatron.core.transformer.moe import token_dispatcher  # noqa: E402
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules  # noqa: E402
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP  # noqa: E402
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, pack_mxfp4_rows, quantize_mxfp4  # noqa: E402
from miles_plugins.models.kimi_k3.mxfp4_redequant import RedequantFrozenExperts  # noqa: E402

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["miles-plugin"])

HIDDEN, FFN, SHARED_FFN, EXPERTS, TOPK, TOKENS = 64, 32, 48, 8, 2, 24


class _GroupedGemm(torch.autograd.Function):
    """TE's grouped linear without wgrad on CPU: saves the fetched weights for the dgrad."""

    @staticmethod
    def forward(ctx, inp, m_splits, *weights):
        ctx.save_for_backward(*weights)
        ctx.m_splits = m_splits
        return torch.cat([x @ w.t() for x, w in zip(torch.split(inp, m_splits), weights, strict=True)])

    @staticmethod
    def backward(ctx, grad):
        weights = ctx.saved_tensors
        dgrad = torch.cat([g @ w for g, w in zip(torch.split(grad, ctx.m_splits), weights, strict=True)])
        return (dgrad, None, *([None] * len(weights)))


class _FrozenFc1(RedequantFrozenExperts, torch.nn.Module):
    def __init__(self, rows):
        torch.nn.Module.__init__(self)
        self.num_gemms = len(rows)
        for i, packed in enumerate(rows):
            self.register_parameter(f"weight{i}", torch.nn.Parameter(packed, requires_grad=False))
        self.dequantized = 0
        self.fetches = 0

    def _dequantize(self, rows, lut):
        self.dequantized += len(rows)
        return RedequantFrozenExperts._dequantize(rows, lut)

    def forward(self, inp, m_splits):
        self.fetches += 1
        return _GroupedGemm.apply(inp, m_splits, *self._get_weight_tensors())


class _Experts(torch.nn.Module):
    """Frozen MXFP4 fc1 (``experts.frozen.linear_fc1``, as in PartiallyTrainableGroupedMLP), trainable fc2."""

    def __init__(self, num_local_experts, config, pg_collection=None, name=None):
        super().__init__()
        rows = []
        for _ in range(num_local_experts):
            packed, scale = quantize_mxfp4(torch.randn(2 * FFN, HIDDEN), MXFP4_GROUP_SIZE)
            rows.append(pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE))
        self.frozen = torch.nn.Module()
        self.frozen.linear_fc1 = _FrozenFc1(rows)
        self.fc2 = torch.nn.Parameter(torch.randn(num_local_experts, HIDDEN, FFN) / FFN**0.5)

    def forward(self, x, tokens_per_expert, probs):
        splits = [int(c) for c in torch.as_tensor(tokens_per_expert).tolist()]
        h = self.frozen.linear_fc1(x.to(torch.bfloat16), splits).float()
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up * probs.unsqueeze(-1)
        out = torch.cat([r @ self.fc2[e].t() for e, r in enumerate(torch.split(h, splits))])
        return out, None


def _cpu_dtoh(self, point, tokens_per_expert=None):
    """``_maybe_dtoh_and_synchronize`` without CUDA streams: the split sizes become host lists."""
    if point == self.cuda_dtoh_point:
        for name in ("input_splits", "output_splits", "output_splits_tp"):
            value = getattr(self, name)
            if torch.is_tensor(value):
                setattr(self, name, value.tolist())
    return tokens_per_expert


def _build(world, seed):
    group = dist.group.WORLD
    single = dist.new_group([dist.get_rank()])
    pgs = ProcessGroupCollection(
        tp=single, cp=single, tp_cp=single, ep=group, tp_ep=group, expt_tp=single, expt_dp=single,
        tp_ep_pp=group, dp=single, dp_cp=single, tp_dp_cp=single, pp=single,
    )  # fmt: skip
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_permute_fusion=False,
        moe_router_dtype="fp32",
        moe_router_use_torch_mm=True,  # TE's router GEMM needs CUDA
        expert_model_parallel_size=world,
        add_bias_linear=False,
        gated_linear_unit=True,
        use_cpu_initialization=True,
        perform_initialization=True,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
    )
    torch.manual_seed(seed)  # the same router / shared expert weights on every rank (replicated)
    shared = functools.partial(
        SharedExpertMLP, submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
    )
    torch.manual_seed(seed)
    layer = overlap.A2AOverlapMoELayer(
        config, MoESubmodules(experts=_Experts, shared_experts=shared), layer_number=1, pg_collection=pgs
    )
    torch.manual_seed(seed + 1 + dist.get_rank())  # rank-local experts
    layer.experts = _Experts(EXPERTS // world, config)
    return layer


def _run(layer, x, use_overlap, mode, events):
    forward = layer.forward if use_overlap else functools.partial(MoELayer.forward, layer)
    xin = x.clone().requires_grad_(mode != "no_grad")
    if mode == "no_grad":
        with torch.no_grad():
            out, _ = forward(xin)
        return out, None, None
    if mode == "checkpoint":
        out = torch.utils.checkpoint.checkpoint(lambda h: forward(h)[0], xin, use_reentrant=True)
    else:
        out, _ = forward(xin)
    events.clear()
    (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    layer.zero_grad(set_to_none=True)
    return out.detach(), xin.grad, grads


def _worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.cuda.current_device = lambda: "cpu"  # Megatron's all-to-all allocates its output there
        MoEAlltoAllTokenDispatcher.cuda_dtoh_stream = object()  # no CUDA stream on CPU
        MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = _cpu_dtoh
        token_dispatcher._record_current_stream = lambda *tensors: None
        events = []
        a2a_backward = mappings._AllToAll.backward

        def recorded_backward(ctx, *grad):
            events.append("a2a")
            return a2a_backward(ctx, *grad)

        mappings._AllToAll.backward = staticmethod(recorded_backward)
        layer = _build(world, seed=7)
        assert layer._a2a_overlap_applies(None)
        layer.shared_experts.register_full_backward_hook(lambda *_: events.append("shared"))
        fc1 = layer.experts.frozen.linear_fc1
        torch.manual_seed(100 + rank)
        x = torch.randn(TOKENS, 1, HIDDEN)

        checks, orders = {}, {}
        for mode in ("plain", "no_grad", "checkpoint"):
            ref = _run(layer, x, False, mode, events)
            orders[(mode, False)] = list(events)
            fc1.dequantized = fc1.fetches = 0
            got = _run(layer, x, True, mode, events)
            orders[(mode, True)] = list(events)
            fetches = {"plain": 1, "no_grad": 1, "checkpoint": 2}[mode]
            # each fetch dequantizes once, the prefetched copy is the one taken, nothing is left behind
            checks[f"{mode}_prefetch_used"] = (
                fc1.fetches == fetches and fc1.dequantized == fetches * fc1.num_gemms and fc1._prefetched is None
            )
            checks[f"{mode}_out"] = torch.equal(ref[0], got[0])
            if mode != "no_grad":
                checks[f"{mode}_dgrad"] = torch.equal(ref[1], got[1])
                checks[f"{mode}_wgrad"] = ref[2].keys() == got[2].keys() and all(
                    torch.equal(ref[2][k], got[2][k]) for k in ref[2]
                )
        # Megatron: combine-bwd a2a, probs a2a, token a2a, then the shared experts (created before the router).
        # Overlap: the shared experts' backward is queued right after the unpermute's, before the combine-bwd a2a
        # node runs, so its kernels sit inside that a2a's window.
        checks["order_ref"] = orders[("plain", False)] == ["a2a", "a2a", "a2a", "shared"]
        checks["order_overlap"] = orders[("plain", True)] == ["shared", "a2a", "a2a", "a2a"]
        checks["order_checkpoint"] = orders[("checkpoint", True)] == ["shared", "a2a", "a2a", "a2a"]
        results[rank] = (checks, {str(k): v for k, v in orders.items()})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_moe_a2a_overlap_is_bit_identical(world):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(world, port, results), nprocs=world, start_method="spawn")
    assert len(results) == world, dict(results)
    for rank, (checks, orders) in results.items():
        assert all(checks.values()), (rank, checks, orders)


def test_spec_swap_and_switch(monkeypatch):
    spec = functools.partial(MoELayer, submodules="s")
    swapped = overlap.a2a_overlap_moe_spec(spec)
    assert swapped.func is overlap.A2AOverlapMoELayer and swapped.keywords == {"submodules": "s"}
    monkeypatch.delenv(overlap.A2A_OVERLAP_ENV, raising=False)
    assert not overlap.a2a_overlap_enabled()
    monkeypatch.setenv(overlap.A2A_OVERLAP_ENV, "1")
    assert overlap.a2a_overlap_enabled()


def test_prefetch_is_dropped_when_the_bytes_change():
    packed, scale = quantize_mxfp4(torch.randn(8, 64), MXFP4_GROUP_SIZE)
    fc1 = _FrozenFc1([pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE)])
    fc1.prefetch_weights()
    stale = fc1._prefetched[0][0]
    packed2, scale2 = quantize_mxfp4(torch.randn(8, 64), MXFP4_GROUP_SIZE)
    with torch.no_grad():
        fc1.weight0.copy_(pack_mxfp4_rows(packed2, scale2, MXFP4_GROUP_SIZE))  # a reload bumps the version
    (weight,) = fc1._get_weight_tensors()
    assert weight is not stale and fc1._prefetched is None
    assert torch.equal(weight, RedequantFrozenExperts._dequantize([fc1.weight0], False)[0])


@pytest.mark.parametrize("switch", ["0", "1"])
def test_k3_spec_builds_the_overlap_layer_only_with_the_switch(monkeypatch, switch):
    from megatron.core import parallel_state

    from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec

    # one pipeline stage, without initializing Megatron's process groups
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda *a, **k: 0)
    monkeypatch.setenv(overlap.A2A_OVERLAP_ENV, switch)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_layer_freq=[0, 1],
        moe_ffn_hidden_size=FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_grouped_gemm=True,
        add_bias_linear=False,
    )
    config.kimi_kda_layers = ()
    dense, moe = (spec.submodules.mlp for spec in build_kimi_k3_spec(config).layer_specs)
    assert moe.func is (overlap.A2AOverlapMoELayer if switch == "1" else MoELayer)
    assert getattr(dense, "func", None) is not overlap.A2AOverlapMoELayer
