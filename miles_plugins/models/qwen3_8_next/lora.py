"""Native LoRA for Qwen3.8-Next (HF ``qwen4_exp``), raw ``--megatron-to-hf-mode`` only.

The Megatron-Bridge LoRA path cannot build this model (no qwen4_exp bridge in the
pinned Bridge), so adapters are injected into the spec-built model the way the
Kimi K3 and Inkling plugins do. Layout:

- GDN layers: the five ``linear_attn`` projections (TP-replicated ``nn.Linear``).
- QSA layers: the fused ``linear_qkv`` (one A, TP-split B exported as q/k/v) and
  ``linear_proj``. The QSA indexer is deliberately not adapted (SGLang has no LoRA
  wrapper for it) so training and serving stay in sync.
- MoE: routed experts in the **per-expert** layout (every expert has its own A and B;
  rank ``--lora-expert-rank``), plus the shared expert.
- Hyper-connections, PLE, router, norms and conv are frozen.

All exported tensors are padded on the rank axis to ``--lora-rank`` because SGLang
keeps a single ``r`` (and ``alpha / r`` scaling) per adapter; the trainer scales every
module by ``alpha / lora_rank`` as well, so the padded columns are exact zeros.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

HF_LAYER_PREFIX = "model.language_model.layers."
HF_LAYER_PATTERN = f"{HF_LAYER_PREFIX}*."

_GDN_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")

_REQUIRED_TARGET_SUFFIXES = frozenset(
    {
        *(f"linear_attn.{name}" for name in _GDN_PROJECTIONS),
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    }
)
# Supported on the trainer side but not in SGLang's LoRA pool; adapting it would be
# silently ignored at rollout, so it is rejected until the serving side lands.
_DEFERRED_TARGET_SUFFIXES = frozenset({"self_attn.indexer.index_qk_proj"})


class Qwen38NextLoRAAdapter(nn.Module):
    def __init__(self, kind: str, hf_prefix: str) -> None:
        super().__init__()
        self.kind = kind
        self.hf_prefix = hf_prefix
        # qsa_attention only: geometry needed to un-interleave the fused qkv B.
        self.num_query_groups = 0
        self.heads_per_group = 0
        self.head_dim = 0

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


# Indirections so CPU tests can stub Megatron's process-group state.
def _parallel_state():
    from megatron.core import parallel_state

    return parallel_state


def _tp_mappings():
    from megatron.core.tensor_parallel import mappings

    return mappings


def expert_lora_rank(args) -> int:
    rank = int(args.lora_rank)
    expert_rank = int(getattr(args, "lora_expert_rank", 0) or 0) or rank
    if not 0 < expert_rank <= rank:
        raise ValueError(
            f"--lora-expert-rank must satisfy 0 < expert_rank <= --lora-rank, got {expert_rank} and {rank}; "
            "SGLang keeps one r per adapter, so expert tensors are zero-padded up to --lora-rank at export"
        )
    return expert_rank


def _new_param(
    ref_weight: torch.Tensor,
    shape: tuple[int, ...],
    *,
    init: str,
    grad_sum_group: str | None = None,
    expert: bool = False,
) -> nn.Parameter:
    tensor = torch.empty(*shape, dtype=ref_weight.dtype, device=ref_weight.device)
    if init == "zero":
        tensor.zero_()
    elif init == "xavier":
        if tensor.ndim == 2:
            nn.init.xavier_uniform_(tensor)
        else:
            for expert_tensor in tensor:
                nn.init.xavier_uniform_(expert_tensor)
    else:
        raise ValueError(f"Unsupported Qwen3.8-Next LoRA init method: {init}")

    param = nn.Parameter(tensor)
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
    if expert:
        # Routed into Megatron's expert-parallel DDP buffer (reduced over the expert DP group).
        param.allreduce = False
    if grad_sum_group == "tp":
        # Megatron sums TP-replicated partial grads itself in finalize_model_grads
        param.sum_gradients_across_tp_domain = True
    elif grad_sum_group is not None:
        raise ValueError(f"unsupported LoRA gradient sum group {grad_sum_group!r}")
    return param


def _register_param(adapter, name, ref_weight, shape, *, init, grad_sum_group=None, expert=False) -> None:
    adapter.register_parameter(
        name, _new_param(ref_weight, shape, init=init, grad_sum_group=grad_sum_group, expert=expert)
    )


def _dropout(inputs: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability and training:
        return F.dropout(inputs, p=probability, training=True)
    return inputs


def _grouped_linear(inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert) -> torch.Tensor:
    """``inputs`` [tokens, in] grouped by expert, ``weights`` [experts, out, in]."""
    if inputs.is_cuda:
        offsets = torch.as_tensor(tokens_per_expert, device=inputs.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return F.grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)
    counts = [int(count) for count in tokens_per_expert]
    segments = torch.split(inputs, counts, dim=0)
    return torch.cat([F.linear(segment, weights[idx]) for idx, segment in enumerate(segments)], dim=0)


def resolve_qwen3_8_next_adapter_targets(targets, *, canonical, experts_shared_outer_loras):
    suffixes = {target.removeprefix(HF_LAYER_PATTERN) for target in targets}
    deferred = suffixes & _DEFERRED_TARGET_SUFFIXES
    if deferred:
        raise NotImplementedError(
            f"Qwen3.8-Next LoRA targets {sorted(deferred)} are deferred: SGLang has no LoRA wrapper for the QSA "
            "indexer, so adapting it on the trainer would never be applied at rollout"
        )
    unsupported = suffixes - _REQUIRED_TARGET_SUFFIXES
    missing = _REQUIRED_TARGET_SUFFIXES - suffixes
    if unsupported or missing:
        raise NotImplementedError(
            "Qwen3.8-Next native LoRA currently requires the verified target set; "
            f"unsupported={sorted(unsupported)}, missing={sorted(missing)}"
        )
    assert not canonical, "Qwen3.8-Next native LoRA does not implement canonical_lora"
    if experts_shared_outer_loras:
        raise NotImplementedError(
            "Qwen3.8-Next native LoRA uses the per-expert layout; drop --experts-shared-outer-loras"
        )
    return list(targets)


def _enable_full_recompute_input_grads(model) -> None:
    if model.config.recompute_granularity != "full" or not model.pre_process:
        return

    def enable_grad(module, _inputs, output):
        if module.training:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Qwen3.8-Next embedding returned {type(output)}, expected torch.Tensor")
            output.requires_grad_(True)
        return output

    model.embedding.register_forward_hook(enable_grad)


def _apply_gdn_lora(attention, args, layer_idx: int, scale: float, dropout: float) -> None:
    """The five ``nn.Linear`` projections of the gated delta net.

    They are TP-replicated (the input is gathered with ``tensor_parallel_output_grad=False``),
    so every TP rank holds identical A/B and computes identical grads; no TP reduction.
    """
    rank = int(args.lora_rank)
    gdn = attention.linear_attn
    adapter = Qwen38NextLoRAAdapter("gdn_attention", f"{HF_LAYER_PREFIX}{layer_idx}.linear_attn.")

    for name in _GDN_PROJECTIONS:
        module = getattr(gdn, name)
        weight = module.weight
        _register_param(adapter, f"{name}_lora_A", weight, (rank, weight.shape[1]), init="xavier")
        _register_param(adapter, f"{name}_lora_B", weight, (weight.shape[0], rank), init="zero")
        lora_a = getattr(adapter, f"{name}_lora_A")
        lora_b = getattr(adapter, f"{name}_lora_B")
        original_forward = module.forward

        def replicated_forward(inputs, *forward_args, _module=module, _original=original_forward, _a=lora_a, _b=lora_b, **forward_kwargs):
            output = _original(inputs, *forward_args, **forward_kwargs)
            delta = F.linear(F.linear(_dropout(inputs, dropout, _module.training), _a), _b)
            return torch.add(output, delta, alpha=scale)

        module.forward = replicated_forward

    attention.lora_adapter = adapter


def _apply_qsa_lora(attention, args, layer_idx: int, scale: float, dropout: float) -> None:
    """Fused ``linear_qkv`` (TE column-parallel) and ``linear_proj`` (TE row-parallel)."""
    mappings = _tp_mappings()
    rank = int(args.lora_rank)
    config = attention.config
    hidden_size = config.hidden_size
    sequence_parallel = bool(config.sequence_parallel)
    tp_group = getattr(attention, "tp_group", None)
    qkv = attention.linear_qkv
    proj = attention.linear_proj

    adapter = Qwen38NextLoRAAdapter("qsa_attention", f"{HF_LAYER_PREFIX}{layer_idx}.self_attn.")
    adapter.num_query_groups = int(config.num_query_groups)
    adapter.heads_per_group = int(config.num_attention_heads) // int(config.num_query_groups)
    adapter.head_dim = int(config.kv_channels)
    if not getattr(config, "attention_output_gate", False):
        raise NotImplementedError("Qwen3.8-Next QSA LoRA export assumes attention_output_gate=True")

    _register_param(adapter, "qkv_lora_A", qkv.weight, (rank, hidden_size), init="xavier", grad_sum_group="tp")
    _register_param(adapter, "qkv_lora_B", qkv.weight, (qkv.weight.shape[0], rank), init="zero")
    _register_param(adapter, "o_lora_A", proj.weight, (rank, proj.weight.shape[1]), init="xavier")
    _register_param(
        adapter,
        "o_lora_B",
        proj.weight,
        (hidden_size, rank),
        init="zero",
        grad_sum_group="tp" if sequence_parallel else None,
    )

    original_qkv = qkv.forward

    def qkv_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_qkv(inputs, *forward_args, **forward_kwargs)
        adapter_inputs = (
            mappings.gather_from_sequence_parallel_region(inputs, group=tp_group) if sequence_parallel else inputs
        )
        delta = F.linear(F.linear(_dropout(adapter_inputs, dropout, qkv.training), adapter.qkv_lora_A), adapter.qkv_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    qkv.forward = qkv_forward
    original_proj = proj.forward

    def proj_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_proj(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, proj.training), adapter.o_lora_A)
        reduced = (
            mappings.reduce_scatter_to_sequence_parallel_region(local, group=tp_group)
            if sequence_parallel
            else mappings.reduce_from_tensor_model_parallel_region(local, group=tp_group)
        )
        delta = F.linear(reduced, adapter.o_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    proj.forward = proj_forward
    attention.lora_adapter = adapter


def _apply_shared_expert_lora(mlp, args, layer_idx: int, scale: float, dropout: float) -> None:
    """Shared expert: ``linear_fc1`` (gate||up, TP column) and ``linear_fc2`` (TP row)."""
    mappings = _tp_mappings()
    rank = int(args.lora_rank)
    sequence_parallel = bool(mlp.config.sequence_parallel)
    tp_group = getattr(mlp, "tp_group", None)
    fc1 = mlp.linear_fc1
    fc2 = mlp.linear_fc2
    adapter = Qwen38NextLoRAAdapter("shared_experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.shared_expert.")
    _register_param(adapter, "fc1_lora_A", fc1.weight, (rank, mlp.config.hidden_size), init="xavier", grad_sum_group="tp")
    _register_param(adapter, "fc1_lora_B", fc1.weight, (fc1.weight.shape[0], rank), init="zero")
    _register_param(adapter, "fc2_lora_A", fc2.weight, (rank, fc2.weight.shape[1]), init="xavier")
    _register_param(
        adapter,
        "fc2_lora_B",
        fc2.weight,
        (mlp.config.hidden_size, rank),
        init="zero",
        grad_sum_group="tp" if sequence_parallel else None,
    )

    original_fc1 = fc1.forward

    def fc1_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, *forward_args, **forward_kwargs)
        adapter_inputs = (
            mappings.gather_from_sequence_parallel_region(inputs, group=tp_group) if sequence_parallel else inputs
        )
        delta = F.linear(F.linear(_dropout(adapter_inputs, dropout, fc1.training), adapter.fc1_lora_A), adapter.fc1_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = fc1_forward
    original_fc2 = fc2.forward

    def fc2_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, fc2.training), adapter.fc2_lora_A)
        reduced = (
            mappings.reduce_scatter_to_sequence_parallel_region(local, group=tp_group)
            if sequence_parallel
            else mappings.reduce_from_tensor_model_parallel_region(local, group=tp_group)
        )
        delta = F.linear(reduced, adapter.fc2_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = fc2_forward
    mlp.lora_adapter = adapter


def _apply_expert_lora(moe, args, layer_idx: int, scale: float, dropout: float) -> None:
    """Routed experts, per-expert layout over the TE grouped GEMMs (ETP=1).

    Every local expert owns its A and B, so there is nothing to reduce over EP: the
    parameters live in the expert-parallel DDP buffer like the base expert weights.
    Each projection costs two extra grouped GEMMs (x@A_e then @B_e); no per-expert loop.
    """
    experts = moe.experts
    if (moe.config.expert_tensor_parallel_size or 1) != 1:
        raise NotImplementedError("Qwen3.8-Next native expert LoRA currently requires ETP=1")

    rank = expert_lora_rank(args)
    num_local_experts = int(experts.num_local_experts)
    hidden_size = int(moe.config.hidden_size)
    ref_fc1 = experts.linear_fc1.weight0  # [2 * I, H]
    ref_fc2 = experts.linear_fc2.weight0  # [H, I]
    intermediate_size = int(ref_fc2.shape[1])
    adapter = Qwen38NextLoRAAdapter("experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.experts.")
    _register_param(adapter, "fc1_lora_A", ref_fc1, (num_local_experts, rank, hidden_size), init="xavier", expert=True)
    _register_param(adapter, "fc1_lora_B", ref_fc1, (num_local_experts, ref_fc1.shape[0], rank), init="zero", expert=True)
    _register_param(adapter, "fc2_lora_A", ref_fc2, (num_local_experts, rank, intermediate_size), init="xavier", expert=True)
    _register_param(adapter, "fc2_lora_B", ref_fc2, (num_local_experts, hidden_size, rank), init="zero", expert=True)

    fc1 = experts.linear_fc1
    original_fc1 = fc1.forward

    def expert_fc1_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        inner = _grouped_linear(_dropout(inputs, dropout, fc1.training), adapter.fc1_lora_A, tokens_per_expert)
        delta = _grouped_linear(inner, adapter.fc1_lora_B, tokens_per_expert)
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = expert_fc1_forward
    fc2 = experts.linear_fc2
    original_fc2 = fc2.forward

    def expert_fc2_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        inner = _grouped_linear(_dropout(inputs, dropout, fc2.training), adapter.fc2_lora_A, tokens_per_expert)
        delta = _grouped_linear(inner, adapter.fc2_lora_B, tokens_per_expert)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = expert_fc2_forward
    experts.lora_adapter = adapter


def apply_qwen3_8_next_lora(model, args):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention
    from miles_plugins.models.qwen3_8_next.qwen3_8_next import Qwen38NextLinearAttention

    rank = int(args.lora_rank)
    if rank <= 0:
        raise ValueError("apply_qwen3_8_next_lora requires --lora-rank > 0")
    expert_rank = expert_lora_rank(args)
    # One scale for every module: SGLang applies alpha / r with the adapter-wide r.
    scale = float(args.lora_alpha) / rank
    dropout = float(args.lora_dropout or 0.0)

    for parameter in model.parameters():
        parameter.requires_grad = False
    _enable_full_recompute_input_grads(model)

    for layer in model.decoder.layers:
        layer_idx = layer.layer_number - 1
        attention = layer.self_attention
        if isinstance(attention, Qwen38NextLinearAttention):
            _apply_gdn_lora(attention, args, layer_idx, scale, dropout)
        elif isinstance(attention, Qwen38NextAttention):
            _apply_qsa_lora(attention, args, layer_idx, scale, dropout)
        else:
            raise TypeError(f"Qwen3.8-Next layer {layer_idx} has unexpected attention type {type(attention)}")

        if not isinstance(layer.mlp, MoELayer):
            raise TypeError(f"Qwen3.8-Next layer {layer_idx} has unexpected MLP type {type(layer.mlp)}")
        _apply_expert_lora(layer.mlp, args, layer_idx, scale, dropout)
        if layer.mlp.shared_experts is None:
            raise RuntimeError(f"Qwen3.8-Next MoE layer {layer_idx} is missing its shared expert")
        _apply_shared_expert_lora(layer.mlp.shared_experts, args, layer_idx, scale, dropout)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Qwen3.8-Next native LoRA applied: rank=%d expert_rank=%d alpha=%s trainable=%d total=%d ratio=%.6f%%",
        rank,
        expert_rank,
        args.lora_alpha,
        trainable,
        total,
        100.0 * trainable / total,
    )
    return model


def wrap_model_provider_with_qwen3_8_next_lora(provider_func, args):
    def wrapped(*provider_args, **provider_kwargs):
        return apply_qwen3_8_next_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


class _GatherBatch:
    """Batch TP/EP all-gathers of adapter shards into one collective per group."""

    class _Token:
        def __init__(self, batch: _GatherBatch, kind: str, index: int) -> None:
            self.batch = batch
            self.kind = kind
            self.index = index

        def get(self) -> torch.Tensor:
            return self.batch.resolved[self.kind][self.index]

    def __init__(self) -> None:
        self.requests: dict[str, list[tuple[torch.Tensor, int]]] = {"tp": [], "ep": []}
        self.resolved: dict[str, list[torch.Tensor]] = {"tp": [], "ep": []}

    def add(self, kind: str, local: torch.Tensor, dim: int) -> _Token:
        self.requests[kind].append((local, dim))
        return self._Token(self, kind, len(self.requests[kind]) - 1)

    def flush(self) -> int:
        parallel_state = _parallel_state()
        groups = {
            "tp": (
                parallel_state.get_tensor_model_parallel_group,
                parallel_state.get_tensor_model_parallel_world_size(),
            ),
            "ep": (
                parallel_state.get_expert_model_parallel_group,
                parallel_state.get_expert_model_parallel_world_size(),
            ),
        }
        calls = 0
        for kind, requests in self.requests.items():
            if not requests:
                continue
            get_group, world_size = groups[kind]
            if world_size == 1:
                self.resolved[kind] = [local for local, _dim in requests]
                continue
            group = get_group()
            dtypes = {local.dtype for local, _dim in requests}
            if len(dtypes) != 1:
                raise TypeError(f"Qwen3.8-Next LoRA {kind} gather has mixed dtypes: {dtypes}")
            flat_parts = [local.detach().contiguous().view(-1) for local, _dim in requests]
            sizes = [part.numel() for part in flat_parts]
            flat_local = torch.cat(flat_parts)
            gathered = flat_local.new_empty(world_size * flat_local.numel())
            torch.distributed.all_gather_into_tensor(gathered, flat_local, group=group)
            per_rank = gathered.view(world_size, flat_local.numel())
            offset = 0
            resolved = []
            for (local, dim), size in zip(requests, sizes, strict=True):
                partitions = [per_rank[rank, offset : offset + size].view(local.shape) for rank in range(world_size)]
                resolved.append(torch.cat(partitions, dim=dim))
                offset += size
            self.resolved[kind] = resolved
            calls += 1
        return calls


def _unwrap_model_chunks(model_chunks):
    for chunk in model_chunks:
        while hasattr(chunk, "module"):
            chunk = chunk.module
        yield chunk


def _validate_adapter_layout(models, adapters: list[Qwen38NextLoRAAdapter]) -> None:
    expected = []
    for model in models:
        for layer in model.decoder.layers:
            layer_idx = layer.layer_number - 1
            attention_kind = "gdn_attention" if hasattr(layer.self_attention, "linear_attn") else "qsa_attention"
            attention_prefix = "linear_attn." if attention_kind == "gdn_attention" else "self_attn."
            expected.append((attention_kind, f"{HF_LAYER_PREFIX}{layer_idx}.{attention_prefix}"))
            expected.append(("experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.experts."))
            expected.append(("shared_experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.shared_expert."))

    expected_counts = Counter(expected)
    actual_counts = Counter((adapter.kind, adapter.hf_prefix) for adapter in adapters)
    if actual_counts != expected_counts:
        missing = list((expected_counts - actual_counts).elements())
        unexpected = list((actual_counts - expected_counts).elements())
        raise RuntimeError(
            "Qwen3.8-Next LoRA adapter layout is incomplete: "
            f"expected={sum(expected_counts.values())}, actual={sum(actual_counts.values())}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def split_qgkv_rows(
    qgkv: torch.Tensor, *, num_query_groups: int, heads_per_group: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of the mbridge qkv packing (``mbridge/qwen3_5.py``) on the row axis.

    Megatron rows per query group: ``[2, heads_per_group, head_dim]`` (query, then output
    gate) followed by ``k`` and ``v`` of ``head_dim`` each. HF rows: q ``[heads, 2, head_dim]``.
    """
    q_rows = 2 * heads_per_group * head_dim
    group_rows = q_rows + 2 * head_dim
    if qgkv.shape[0] != num_query_groups * group_rows:
        raise ValueError(
            f"fused qkv has {qgkv.shape[0]} rows, expected {num_query_groups} groups x {group_rows}"
        )
    trailing = qgkv.shape[1:]
    grouped = qgkv.reshape(num_query_groups, group_rows, *trailing)
    q = (
        grouped[:, :q_rows]
        .reshape(num_query_groups, 2, heads_per_group, head_dim, *trailing)
        .transpose(1, 2)
        .reshape(num_query_groups * heads_per_group * 2 * head_dim, *trailing)
    )
    k = grouped[:, q_rows : q_rows + head_dim].reshape(num_query_groups * head_dim, *trailing)
    v = grouped[:, q_rows + head_dim :].reshape(num_query_groups * head_dim, *trailing)
    return q, k, v


def _pad_rank(tensor: torch.Tensor, dim: int, rank: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current == rank:
        return tensor
    if current > rank:
        raise ValueError(f"LoRA tensor rank {current} exceeds export rank {rank}")
    pad_shape = list(tensor.shape)
    pad_shape[dim] = rank - current
    return torch.cat((tensor, tensor.new_zeros(pad_shape)), dim=dim)


def _export_gdn(adapter: Qwen38NextLoRAAdapter):
    prefix = adapter.hf_prefix
    plans = []
    for name in _GDN_PROJECTIONS:
        plans.append((f"{prefix}{name}.lora_A.weight", getattr(adapter, f"{name}_lora_A")))
        plans.append((f"{prefix}{name}.lora_B.weight", getattr(adapter, f"{name}_lora_B")))
    return plans


def _export_qsa(adapter: Qwen38NextLoRAAdapter):
    batch = _GatherBatch()
    prefix = adapter.hf_prefix
    qkv_b = batch.add("tp", adapter.qkv_lora_B, 0)
    o_a = batch.add("tp", adapter.o_lora_A, 1)
    batch.flush()
    q_b, k_b, v_b = split_qgkv_rows(
        qkv_b.get(),
        num_query_groups=adapter.num_query_groups,
        heads_per_group=adapter.heads_per_group,
        head_dim=adapter.head_dim,
    )
    qkv_a = adapter.qkv_lora_A
    return [
        (f"{prefix}q_proj.lora_A.weight", qkv_a),
        (f"{prefix}q_proj.lora_B.weight", q_b),
        (f"{prefix}k_proj.lora_A.weight", qkv_a),
        (f"{prefix}k_proj.lora_B.weight", k_b),
        (f"{prefix}v_proj.lora_A.weight", qkv_a),
        (f"{prefix}v_proj.lora_B.weight", v_b),
        (f"{prefix}o_proj.lora_A.weight", o_a.get),
        (f"{prefix}o_proj.lora_B.weight", adapter.o_lora_B),
    ]


def _export_shared_expert(adapter: Qwen38NextLoRAAdapter):
    batch = _GatherBatch()
    prefix = adapter.hf_prefix
    fc1_a = adapter.fc1_lora_A
    gate_b_local, up_b_local = adapter.fc1_lora_B.chunk(2, dim=0)
    gate_b = batch.add("tp", gate_b_local, 0)
    up_b = batch.add("tp", up_b_local, 0)
    down_a = batch.add("tp", adapter.fc2_lora_A, 1)
    batch.flush()
    return [
        (f"{prefix}gate_proj.lora_A.weight", fc1_a),
        (f"{prefix}gate_proj.lora_B.weight", gate_b.get),
        (f"{prefix}up_proj.lora_A.weight", fc1_a),
        (f"{prefix}up_proj.lora_B.weight", up_b.get),
        (f"{prefix}down_proj.lora_A.weight", down_a.get),
        (f"{prefix}down_proj.lora_B.weight", adapter.fc2_lora_B),
    ]


def _export_experts(adapter: Qwen38NextLoRAAdapter, rank: int):
    """Per-expert: every tensor is EP-gathered on the expert axis into [num_experts, ...]."""
    batch = _GatherBatch()
    prefix = adapter.hf_prefix
    fc1_a = batch.add("ep", _pad_rank(adapter.fc1_lora_A, 1, rank), 0)
    fc1_b = batch.add("ep", _pad_rank(adapter.fc1_lora_B, 2, rank), 0)
    fc2_a = batch.add("ep", _pad_rank(adapter.fc2_lora_A, 1, rank), 0)
    fc2_b = batch.add("ep", _pad_rank(adapter.fc2_lora_B, 2, rank), 0)
    batch.flush()
    return [
        (f"{prefix}gate_up_proj.lora_A.weight", fc1_a.get),
        (f"{prefix}gate_up_proj.lora_B.weight", fc1_b.get),
        (f"{prefix}down_proj.lora_A.weight", fc2_a.get),
        (f"{prefix}down_proj.lora_B.weight", fc2_b.get),
    ]


def export_qwen3_8_next_lora_hf_chunks(model_chunks):
    models = list(_unwrap_model_chunks(model_chunks))
    adapters: list[Qwen38NextLoRAAdapter] = []
    for model in models:
        adapters.extend(module for module in model.modules() if isinstance(module, Qwen38NextLoRAAdapter))
    if not adapters:
        raise RuntimeError("Qwen3.8-Next native LoRA export found no adapters")
    _validate_adapter_layout(models, adapters)
    # The adapter-wide r is the attention rank; experts are padded up to it.
    rank = max(
        parameter.shape[-1]
        for adapter in adapters
        if adapter.kind != "experts"
        for name, parameter in adapter.named_parameters()
        if name.endswith("_lora_B")
    )

    for adapter in adapters:
        plans: list[tuple[str, torch.Tensor | Callable[[], torch.Tensor]]]
        if adapter.kind == "gdn_attention":
            plans = _export_gdn(adapter)
        elif adapter.kind == "qsa_attention":
            plans = _export_qsa(adapter)
        elif adapter.kind == "shared_experts":
            plans = _export_shared_expert(adapter)
        elif adapter.kind == "experts":
            plans = _export_experts(adapter, rank)
        else:
            raise ValueError(f"Unknown Qwen3.8-Next LoRA adapter kind: {adapter.kind}")
        yield [
            (name, (value() if callable(value) else value).detach().to(torch.bfloat16).contiguous())
            for name, value in plans
        ]
