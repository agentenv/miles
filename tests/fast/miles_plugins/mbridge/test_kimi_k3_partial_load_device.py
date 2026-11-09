"""The trainable experts' BF16 masters dequantize on the experts' device, bit-identical to CPU."""

import pytest
import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows, mxfp4_row_width

kimi_k3 = pytest.importorskip("miles_plugins.mbridge.kimi_k3")


class _Half(torch.nn.Module):
    def __init__(self, n, out1, out2, k1, k2):
        super().__init__()
        self.linear_fc1 = torch.nn.Module()
        self.linear_fc2 = torch.nn.Module()
        for i in range(n):
            self.linear_fc1.register_parameter(f"weight{i}", torch.nn.Parameter(torch.zeros(out1, k1, dtype=torch.bfloat16)))
            self.linear_fc2.register_parameter(f"weight{i}", torch.nn.Parameter(torch.zeros(out2, k2, dtype=torch.bfloat16)))


def test_expert_device_is_the_halfs_parameter_device():
    half = _Half(1, 4, 4, 64, 64)
    assert kimi_k3._expert_device(half) == torch.device("cpu")


def test_dequantized_copy_matches_the_cpu_dequantization():
    torch.manual_seed(0)
    k = 64
    rows = torch.randint(0, 256, (4, mxfp4_row_width(k, MXFP4_GROUP_SIZE)), dtype=torch.uint8)
    rows[:, k // 2 :] = torch.randint(120, 130, (4, k // MXFP4_GROUP_SIZE), dtype=torch.uint8)
    half = _Half(1, 4, 4, k, k)
    device = kimi_k3._expert_device(half)
    w = dequantize_mxfp4_rows(rows.to(device), MXFP4_GROUP_SIZE)
    with torch.no_grad():
        kimi_k3._copy_expert(half, 0, w, w)
    assert torch.equal(half.linear_fc1.weight0, dequantize_mxfp4_rows(rows, MXFP4_GROUP_SIZE))
