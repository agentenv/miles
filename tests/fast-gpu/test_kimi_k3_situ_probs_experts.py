"""SituProbsTEGroupedMLP (MILES_K3_SITU_PROBS_FUSED) equals Megatron's TEGroupedMLP bit for bit on TE grouped
linears: output, input gradient, router-prob gradient and both expert weight gradients, including empty
experts and an all-empty microbatch."""

import os

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles_plugins.models.kimi_k3.ops import situ_and_mul  # noqa: E402
from miles_plugins.models.kimi_k3.situ_probs_experts import SituProbsTEGroupedMLP  # noqa: E402

HIDDEN, FFN, EXPERTS = 256, 3072, 4


@pytest.fixture(scope="module")
def setup():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        gated_linear_unit=True,
        gated_activation_func=situ_and_mul,
        bias_activation_fusion=False,
    )
    yield config, ProcessGroupCollection.use_mpu_process_groups()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.int32)


@pytest.mark.parametrize("counts", [[700, 0, 1301, 47], [0, 0, 0, 0]])
def test_matches_te_grouped_mlp_bitwise(setup, counts):
    config, pg_collection = setup
    submodules = GroupedMLPSubmodules(linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear)
    ref = TEGroupedMLP(EXPERTS, config, submodules, pg_collection).cuda()
    fused = SituProbsTEGroupedMLP(EXPERTS, config, submodules, pg_collection).cuda()
    torch.manual_seed(2)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    fused.load_state_dict(ref.state_dict())

    rows = sum(counts)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 2).to(torch.bfloat16)
    probs = torch.rand(rows, device="cuda")
    dy = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
    tokens = torch.tensor(counts)

    results = []
    for mod in (ref, fused):
        xi, pi = x.clone().requires_grad_(), probs.clone().requires_grad_()
        out, bias = mod(xi, tokens, pi)
        assert bias is None
        out.backward(dy)
        results.append((out, xi.grad, pi.grad, *[p.grad for p in mod.parameters()]))
    assert rows == 0 or all(g is not None for g in results[0])
    for got_ref, got in zip(*results, strict=True):
        if got_ref is None:
            assert got is None
            continue
        assert got.dtype == got_ref.dtype and got.shape == got_ref.shape
        assert torch.equal(_bits(got), _bits(got_ref))
