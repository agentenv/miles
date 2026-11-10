"""--partial-experts-selection freezes the embedding; under full recompute the first pipeline stage's reentrant
checkpoint then sees no input requiring grad and its trainable experts get no gradient. The freeze installs
the LoRA path's first-activation hook so they do.

model_provider imports megatron, so ``_maybe_freeze_for_partial_experts`` is read as source (ast) and run on CPU.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / "miles/backends/megatron_utils/model_provider.py"


def _freeze_fn():
    tree = ast.parse(SOURCE.read_text())
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_freeze_for_partial_experts"]
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"TrainerConfig": object, "GPTModel": object, "logger": logging.getLogger(__name__)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_maybe_freeze_for_partial_experts"]


class _Checkpoint(torch.autograd.Function):
    """Megatron's tensor_parallel.checkpoint (CheckpointFunction), trimmed: reentrant, recompute in backward."""

    @staticmethod
    def forward(ctx, run_function, hidden):
        ctx.run_function = run_function
        ctx.save_for_backward(hidden)
        with torch.no_grad():
            return run_function(hidden)

    @staticmethod
    def backward(ctx, grad):
        (hidden,) = ctx.saved_tensors
        hidden = hidden.detach().requires_grad_(hidden.requires_grad)
        with torch.enable_grad():
            out = ctx.run_function(hidden)
        torch.autograd.backward(out, grad)
        return None, hidden.grad


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight0 = nn.Parameter(torch.randn(8, 8))


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Module()
        self.trainable.linear_fc1 = _Linear()


class _FirstStage(nn.Module):
    """The first PP stage: embedding, then one MoE layer whose trainable experts run under full recompute."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _Experts()
        layer.mlp.router = nn.Linear(8, 4)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([layer])

    def forward(self, tokens):
        weight = self.decoder.layers[0].mlp.experts.trainable.linear_fc1.weight0
        return _Checkpoint.apply(lambda h: h @ weight, self.embedding(tokens))


def _expert_grad(model):
    model.train()
    out = model(torch.tensor([1, 2, 3]))
    if out.requires_grad:
        out.sum().backward()
    return model.decoder.layers[0].mlp.experts.trainable.linear_fc1.weight0.grad


def _args():
    return SimpleNamespace(partial_experts_selection="selection.json", backend=SimpleNamespace(moe_router_bias_update_rate=0))


def test_reentrant_recompute_loses_first_stage_expert_grad_without_the_hook():
    from miles_plugins.models.kimi_k3.partial_experts import freeze_all_but_trainable_experts

    model = _FirstStage()
    freeze_all_but_trainable_experts(model)
    assert not model.embedding.weight.requires_grad
    assert _expert_grad(model) is None


def test_partial_experts_freeze_keeps_first_stage_expert_grad_under_full_recompute():
    model = _FirstStage()
    _freeze_fn()(_args(), model)
    assert not model.embedding.weight.requires_grad
    assert model.decoder.layers[0].mlp.experts.trainable.linear_fc1.weight0.requires_grad
    grad = _expert_grad(model)
    assert grad is not None and grad.abs().sum() > 0
    assert model.embedding.weight.grad is None


def test_no_selection_installs_nothing():
    model = _FirstStage()
    _freeze_fn()(SimpleNamespace(partial_experts_selection=None), model)
    assert not model.embedding._forward_hooks
