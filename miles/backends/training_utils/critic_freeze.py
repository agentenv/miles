"""Critic attention freeze (ported from agentenv/miles SAO 16a9bea4); megatron-free for CPU tests."""

import torch

_ATTENTION_COMPONENTS = frozenset(
    {"attention", "attn", "cross_attention", "cross_attn", "self_attention", "self_attn"}
)


def freeze_attention_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze parameters owned by attention modules, once per shared tensor (SAO 16a9bea4).

    Returns (tensors, elements) frozen.
    """
    parameters: dict[int, torch.nn.Parameter] = {}
    for module_name, module in model.named_modules():
        if not set(module_name.lower().split(".")) & _ATTENTION_COMPONENTS:
            continue
        for parameter in module.parameters(recurse=True):
            parameters[id(parameter)] = parameter
    if not parameters:
        raise RuntimeError("critic attention freeze matched no parameters")
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    return len(parameters), sum(parameter.numel() for parameter in parameters.values())
