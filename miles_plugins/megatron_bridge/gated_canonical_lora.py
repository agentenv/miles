"""Canonical LoRA compatibility for Megatron gated attention."""

from collections.abc import Callable
from typing import Any

import torch


def transform_gated_canonical_lora(
    transform: Callable[..., Any],
    peft: Any,
    module: Any,
    name: str | None = None,
    prefix: str | None = None,
) -> Any:
    """Make CanonicalLoRA allocate both query and gate projection rows."""
    config = getattr(module, "config", None)
    if name != "linear_qkv" or not getattr(config, "attention_output_gate", False):
        return transform(peft, module, name=name, prefix=prefix)

    num_heads = config.num_attention_heads
    config.num_attention_heads = num_heads * 2
    try:
        return transform(peft, module, name=name, prefix=prefix)
    finally:
        config.num_attention_heads = num_heads


def interleave_gated_qkv(
    query_and_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Pack canonical Q/gate/K/V outputs in Megatron's grouped order."""
    num_heads = config.num_attention_heads
    num_groups = config.num_query_groups
    head_size = config.kv_channels or config.hidden_size // num_heads
    heads_per_group = num_heads // num_groups
    leading_shape = query_and_gate.shape[:-1]

    query_and_gate = query_and_gate.reshape(-1, num_heads, 2, head_size)
    query, gate = query_and_gate.unbind(dim=2)
    key = key.reshape(-1, num_groups, head_size)
    value = value.reshape(-1, num_groups, head_size)

    chunks = []
    for group in range(num_groups):
        start = group * heads_per_group
        stop = start + heads_per_group
        chunks.extend(
            [
                query[:, start:stop],
                gate[:, start:stop],
                key[:, group : group + 1],
                value[:, group : group + 1],
            ]
        )
    return torch.cat(chunks, dim=1).reshape(*leading_shape, -1)


def install() -> None:
    """Patch the two CanonicalLoRA operations missing gated-attention support."""
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitQKV

    if getattr(CanonicalLoRA, "_miles_gated_attention_installed", False):
        return

    original_transform = CanonicalLoRA.transform
    original_interleave = LoRALinearSplitQKV._interleave_qkv

    def transform(self, module, name=None, prefix=None):
        return transform_gated_canonical_lora(original_transform, self, module, name, prefix)

    def interleave(self, query, key, value):
        config = self.to_wrap.config
        if not getattr(config, "attention_output_gate", False):
            return original_interleave(self, query, key, value)
        return interleave_gated_qkv(query, key, value, config)

    CanonicalLoRA.transform = transform
    LoRALinearSplitQKV._interleave_qkv = interleave
    CanonicalLoRA._miles_gated_attention_installed = True
