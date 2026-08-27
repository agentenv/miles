from types import SimpleNamespace

import torch

from miles_plugins.megatron_bridge.gated_canonical_lora import (
    interleave_gated_qkv,
    transform_gated_canonical_lora,
)


def test_gated_canonical_lora_builds_query_and_gate_rows_then_restores_config():
    config = SimpleNamespace(attention_output_gate=True, num_attention_heads=4)
    module = SimpleNamespace(config=config)
    observed_heads = []

    def transform(_peft, transformed, name=None, prefix=None):
        observed_heads.append(transformed.config.num_attention_heads)
        return transformed

    result = transform_gated_canonical_lora(transform, object(), module, name="linear_qkv")

    assert result is module
    assert observed_heads == [8]
    assert config.num_attention_heads == 4


def test_gated_canonical_lora_interleaves_query_gate_key_value_by_group():
    config = SimpleNamespace(
        attention_output_gate=True,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=1,
    )
    query_and_gate = torch.tensor([[10, 20, 11, 21, 12, 22, 13, 23]])
    key = torch.tensor([[30, 31]])
    value = torch.tensor([[40, 41]])

    result = interleave_gated_qkv(query_and_gate, key, value, config)

    assert result.tolist() == [[10, 11, 20, 21, 30, 40, 12, 13, 22, 23, 31, 41]]
