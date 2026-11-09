"""Routed-expert MXFP4 QAT: the trainer's rounded weights equal the release round trip bit for bit.

The kernel tests run the Triton ``fake_quant_fp4_forward`` and need CUDA (the Triton interpreter has no
``libdevice.rint``).
"""

import importlib.util

import pytest
import torch

from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, dequantize_mxfp4, quantize_mxfp4
from miles.utils.mxfp4_fake_qat import (
    fake_quantize_mxfp4,
    install_routed_expert_mxfp4_qat,
    mxfp4_round_trip,
    round_in_place_if_off_grid,
)

_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIES_TO_EVEN = (0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """Bit patterns, so -0.0 and +0.0 differ."""
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


def _assert_bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(_bits(a), _bits(b))


def _assert_equal_up_to_signed_zero(a: torch.Tensor, b: torch.Tensor) -> None:
    """Equal values; bit patterns differ at most where a zero carries a different sign."""
    assert torch.equal(a, b)
    differs = _bits(a) != _bits(b)
    assert (a[differs] == 0).all()


def _random_weight(dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    magnitudes = torch.logspace(-30, 30, steps=61, base=2.0).reshape(-1, 1)
    return (torch.randn((61, 8 * MXFP4_GROUP_SIZE), generator=gen) * magnitudes).to(dtype)


def _tie_blocks() -> torch.Tensor:
    """Blocks whose amax/6 sits on or next to a power of two, with every rounding tie and its neighbours."""
    blocks = []
    for exponent in (-140, -130, -126, -24, -9, -6, 0, 3, 8):
        scale = 2.0**exponent
        for amax_factor in (1.0, 1.0 + 2.0**-23, 1.0 - 2.0**-24):
            for tie in _TIES:
                block = torch.full((MXFP4_GROUP_SIZE,), tie * scale, dtype=torch.float32)
                block[0] = 6.0 * scale * amax_factor
                block[1::2] *= -1
                block[2] = torch.nextafter(block[2], torch.tensor(float("inf")))
                block[3] = torch.nextafter(block[3], torch.tensor(0.0))
                block[4] = -(2.0**-30) * scale  # rounds to -0
                block[5] = -0.0
                blocks.append(block)
    return torch.stack(blocks)


def _bf16_tie_weight() -> torch.Tensor:
    """BF16 expert-like weight in which every element of half the blocks is an exact rounding tie."""
    gen = torch.Generator().manual_seed(3)
    rows = []
    for e in range(-12, -2):
        row = torch.tensor(_TIES).repeat(5)[:MXFP4_GROUP_SIZE] * 2.0**e
        row[0] = 6.0 * 2.0**e
        signs = torch.where(torch.rand(MXFP4_GROUP_SIZE, generator=gen) < 0.5, -1.0, 1.0)
        rows.append(torch.cat([row * signs, torch.randn(MXFP4_GROUP_SIZE, generator=gen) * 2.0**e]))
    return torch.stack(rows).to(torch.bfloat16)


# --- the release codec ------------------------------------------------------------------------------


def test_release_quantizer_rounds_ties_to_even():
    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 6.0
    weight[0, 1:8] = torch.tensor(_TIES)
    weight[0, 8:15] = -torch.tensor(_TIES)

    decoded = dequantize_mxfp4(*quantize_mxfp4(weight, MXFP4_GROUP_SIZE), MXFP4_GROUP_SIZE).float()

    assert decoded[0, 1:8].tolist() == list(_TIES_TO_EVEN)
    assert decoded[0, 8:15].tolist() == [-v for v in _TIES_TO_EVEN]


def test_release_quantizer_scale_is_ceil_pow2_of_amax_over_6():
    weight = torch.zeros(3, MXFP4_GROUP_SIZE)
    weight[0, 0] = 48.0  # 48/6 = 8 exactly
    weight[1, 0] = 49.0  # just above: 16
    weight[2, 0] = 2.0**-140  # below the floor: 2**-126
    _, scale = quantize_mxfp4(weight, MXFP4_GROUP_SIZE)
    assert scale.flatten().tolist() == [127 + 3, 127 + 4, 1]


def test_release_quantizer_keeps_values_already_on_the_grid():
    """A checkpoint that is already MXFP4 (the K3.1 base) decodes to the same values after re-quantization."""
    gen = torch.Generator().manual_seed(4)
    packed = torch.randint(0, 256, (64, 4 * MXFP4_GROUP_SIZE // 2), dtype=torch.uint8, generator=gen)
    scale = torch.randint(110, 135, (64, 4), dtype=torch.uint8, generator=gen)
    base = dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE)

    # -0 codes (sign bit, magnitude 0) re-encode as +0, in the codec and in the Triton kernel alike
    _assert_equal_up_to_signed_zero(mxfp4_round_trip(base), base)


# --- the trainer side -------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cpu_fake_quant_is_the_release_round_trip(dtype):
    for weight in (_random_weight(dtype), _tie_blocks().to(dtype), _bf16_tie_weight().to(dtype)):
        packed, scale = quantize_mxfp4(weight, MXFP4_GROUP_SIZE)
        release = dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE).to(dtype)
        _assert_bitwise_equal(fake_quantize_mxfp4(weight), release)


def test_rounding_is_idempotent_and_release_of_rounded_weight_decodes_unchanged():
    rounded = mxfp4_round_trip(_random_weight(torch.bfloat16, seed=5))
    _assert_equal_up_to_signed_zero(mxfp4_round_trip(rounded), rounded)


def test_round_in_place_writes_once_then_leaves_the_parameter_alone():
    param = torch.nn.Parameter(_random_weight(torch.bfloat16, seed=6))
    expected = mxfp4_round_trip(param.detach())
    ptr, version = param.data_ptr(), param._version

    assert round_in_place_if_off_grid(param) is True
    _assert_bitwise_equal(param.detach(), expected)
    assert param.data_ptr() == ptr
    version_after_write = param._version
    assert version_after_write > version

    assert round_in_place_if_off_grid(param) is False
    assert param._version == version_after_write  # no write: saved-for-backward checks stay valid


class _FakeGroupedLinear(torch.nn.Module):
    """The TE GroupedLinear surface the hook relies on: num_gemms, weight<i>, _get_weight_tensors, config."""

    def __init__(self, num_gemms: int, *, trainable: bool, seed: int):
        super().__init__()
        self.num_gemms = num_gemms
        self.config = type("Cfg", (), {"moe_grouped_gemm": True})()
        for i in range(num_gemms):
            w = _random_weight(torch.bfloat16, seed=seed + i)[:8, : 2 * MXFP4_GROUP_SIZE].clone()
            self.register_parameter(f"weight{i}", torch.nn.Parameter(w, requires_grad=trainable))

    def _get_weight_tensors(self):
        return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

    def forward(self, x):
        return torch.cat([x @ w.t() for w in self._get_weight_tensors()], dim=-1)


def _model():
    experts = torch.nn.Module()
    experts.trainable = torch.nn.Module()
    experts.trainable.linear_fc1 = _FakeGroupedLinear(2, trainable=True, seed=10)
    experts.frozen = torch.nn.Module()
    experts.frozen.linear_fc1 = _FakeGroupedLinear(2, trainable=False, seed=20)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = experts
    layer.mlp.shared_experts = _FakeGroupedLinear(1, trainable=True, seed=30)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([layer])
    return model


def test_install_hooks_only_the_trainable_routed_experts():
    model = _model()
    layer = model.decoder.layers[0]
    frozen_before = [w.detach().clone() for w in layer.mlp.experts.frozen.linear_fc1._get_weight_tensors()]
    shared_before = layer.mlp.shared_experts.weight0.detach().clone()
    hot = layer.mlp.experts.trainable.linear_fc1
    masters = [w.detach().clone() for w in hot._get_weight_tensors()]

    assert install_routed_expert_mxfp4_qat(model) == 1
    assert install_routed_expert_mxfp4_qat(model) == 0  # idempotent

    x = torch.randn(3, 2 * MXFP4_GROUP_SIZE).to(torch.bfloat16)
    hot(x).float().sum().backward()

    for i, master in enumerate(masters):
        _assert_bitwise_equal(getattr(hot, f"weight{i}").detach(), mxfp4_round_trip(master))
        # straight-through: the gradient reaches the parameter itself (d/dW of x @ W.T summed = column sums of x)
        grad = getattr(hot, f"weight{i}").grad
        torch.testing.assert_close(grad.float(), x.float().sum(0).expand(8, -1), rtol=1e-2, atol=1e-2)
    for w, before in zip(layer.mlp.experts.frozen.linear_fc1._get_weight_tensors(), frozen_before, strict=True):
        _assert_bitwise_equal(w.detach(), before)
    _assert_bitwise_equal(layer.mlp.shared_experts.weight0.detach(), shared_before)


# --- the Triton kernel the CUDA path uses -----------------------------------------------------------

_HAS_TRITON = importlib.util.find_spec("triton") is not None
_KERNEL_DEVICE = "cuda"
kernel = pytest.mark.skipif(not _HAS_TRITON or not torch.cuda.is_available(), reason="needs triton on CUDA")


@kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernel_matches_the_release_round_trip_bit_for_bit(dtype):
    from miles.kernels.quant.fake_quant import fake_quant_fp4_forward

    for weight in (_random_weight(dtype), _tie_blocks().to(dtype), _bf16_tie_weight().to(dtype)):
        weight = weight.to(_KERNEL_DEVICE)
        _assert_bitwise_equal(fake_quant_fp4_forward(weight).cpu(), mxfp4_round_trip(weight.cpu()))


@kernel
def test_kernel_ties_round_to_even():
    from miles.kernels.quant.fake_quant import fake_quant_fp4_forward

    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 6.0
    weight[0, 1:8] = torch.tensor(_TIES)
    out = fake_quant_fp4_forward(weight.to(_KERNEL_DEVICE)).cpu()
    assert out[0, 1:8].tolist() == list(_TIES_TO_EVEN)
    assert set(E2M1_VALUES) >= set(out.abs().flatten().tolist())
