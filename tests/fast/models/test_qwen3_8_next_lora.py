"""CPU tests for the Qwen3.8-Next (qwen4_exp) native LoRA plugin: target resolution,
forward deltas, the fused-qkv un-interleave and the HF export layout SGLang loads."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from miles.utils.lora.hf_lora_targets import get_hf_lora_targets, resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules, is_qwen3_8_next_model, matches_lora_target
from miles_plugins.models.qwen3_8_next import lora as qwen_lora
from miles_plugins.models.qwen3_8_next.lora import (
    HF_LAYER_PREFIX,
    Qwen38NextLoRAAdapter,
    _apply_expert_lora,
    _apply_gdn_lora,
    _apply_qsa_lora,
    _apply_shared_expert_lora,
    _enable_full_recompute_input_grads,
    _grouped_linear,
    expert_lora_rank,
    export_qwen3_8_next_lora_hf_chunks,
    resolve_qwen3_8_next_adapter_targets,
    split_qgkv_rows,
)

# Shrunk 4-layer variant: 3 GDN layers + 1 QSA layer, all-MoE with a shared expert.
HIDDEN, HEAD_DIM, HEADS, KV_HEADS = 32, 8, 4, 2
LIN_K_HEADS, LIN_V_HEADS, LIN_HEAD_DIM = 2, 4, 8
EXPERTS, MOE_INTER, SHARED_INTER = 4, 16, 16


def _text_config():
    return dict(
        num_hidden_layers=4,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=LIN_K_HEADS,
        linear_num_value_heads=LIN_V_HEADS,
        linear_key_head_dim=LIN_HEAD_DIM,
        linear_value_head_dim=LIN_HEAD_DIM,
        num_experts=EXPERTS,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        attn_output_gate=True,
    )


def _hf_config():
    return dict(model_type="qwen4_exp", text_config=_text_config())


def _default_hf_targets():
    return resolve_hf_lora_targets(_hf_config())


def _args(rank=4, expert_rank=0, alpha=8, dropout=0.0):
    return SimpleNamespace(lora_rank=rank, lora_expert_rank=expert_rank, lora_alpha=alpha, lora_dropout=dropout)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def test_hf_layout_covers_gdn_qsa_experts_and_shared_expert():
    layout = get_hf_lora_targets(_hf_config())
    suffixes = {target.removeprefix("model.language_model.layers.*.") for target in layout.attention + layout.mlp}
    assert suffixes == qwen_lora._REQUIRED_TARGET_SUFFIXES
    assert all(target.startswith("model.language_model.layers.*.") for target in layout.attention + layout.mlp)
    assert not any("indexer" in target for target in suffixes)
    # The text-only alias shares the group builder but has no multimodal prefix.
    text_layout = get_hf_lora_targets(dict(model_type="qwen4_exp_text", **_text_config()))
    assert {t.removeprefix("model.layers.*.") for t in text_layout.attention + text_layout.mlp} == suffixes


@pytest.mark.parametrize("case", ["ok", "missing", "extra", "indexer", "canonical", "shared-outer"])
def test_native_target_resolution(case):
    targets = _default_hf_targets()
    if case == "missing":
        targets = [target for target in targets if not target.endswith("mlp.experts.down_proj")]
    elif case == "extra":
        targets.append("model.language_model.layers.*.mlp.gate")
    elif case == "indexer":
        targets.append("model.language_model.layers.*.self_attn.indexer.index_qk_proj")
    kwargs = dict(canonical=case == "canonical", experts_shared_outer_loras=case == "shared-outer")
    if case == "ok":
        assert resolve_qwen3_8_next_adapter_targets(targets, **kwargs) == targets
        return
    error = AssertionError if case == "canonical" else NotImplementedError
    message = {
        "missing": "missing=",
        "extra": "unsupported=",
        "indexer": "deferred",
        "canonical": "canonical_lora",
        "shared-outer": "per-expert",
    }[case]
    with pytest.raises(error, match=message):
        resolve_qwen3_8_next_adapter_targets(targets, **kwargs)


@pytest.mark.parametrize("rank,expert_rank,expected", [(4, 0, 4), (4, 2, 2), (4, 4, 4)])
def test_expert_rank_defaults_to_lora_rank_and_must_not_exceed_it(rank, expert_rank, expected):
    assert expert_lora_rank(_args(rank=rank, expert_rank=expert_rank)) == expected


@pytest.mark.parametrize("expert_rank", [-1, 5])
def test_expert_rank_out_of_range_is_rejected(expert_rank):
    with pytest.raises(ValueError, match="lora-expert-rank"):
        expert_lora_rank(_args(rank=4, expert_rank=expert_rank))


def test_model_detection_uses_provider_path_or_model_name():
    assert is_qwen3_8_next_model(
        SimpleNamespace(
            custom_model_provider_path="miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider",
            model_name=None,
        )
    )
    assert is_qwen3_8_next_model(SimpleNamespace(custom_model_provider_path=None, model_name="qwen4_exp"))
    assert is_qwen3_8_next_model(SimpleNamespace(custom_model_provider_path=None, model_name="Qwen3.8-Flash-Next-4layer"))
    assert not is_qwen3_8_next_model(SimpleNamespace(custom_model_provider_path=None, model_name="kimi_k3"))


# ---------------------------------------------------------------------------
# fused qkv rows
# ---------------------------------------------------------------------------


def _mbridge_pack_qkv(q, k, v, *, num_kv_heads, heads_per_group, head_dim):
    """The forward packing of mbridge/qwen3_5.py `_weight_to_mcore_format` (q rows carry the output gate)."""
    q = q.view([num_kv_heads, heads_per_group, 2, head_dim, -1]).transpose(1, 2).flatten(1, 3)
    k = k.view([num_kv_heads, head_dim, -1])
    v = v.view([num_kv_heads, head_dim, -1])
    return torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1])


def test_split_qgkv_rows_inverts_mbridge_packing():
    heads_per_group = HEADS // KV_HEADS
    rank = 3
    q = torch.randn(HEADS * 2 * HEAD_DIM, rank)
    k = torch.randn(KV_HEADS * HEAD_DIM, rank)
    v = torch.randn(KV_HEADS * HEAD_DIM, rank)
    packed = _mbridge_pack_qkv(q, k, v, num_kv_heads=KV_HEADS, heads_per_group=heads_per_group, head_dim=HEAD_DIM)
    assert packed.shape == (KV_HEADS * (2 * heads_per_group * HEAD_DIM + 2 * HEAD_DIM), rank)

    q_out, k_out, v_out = split_qgkv_rows(
        packed, num_query_groups=KV_HEADS, heads_per_group=heads_per_group, head_dim=HEAD_DIM
    )
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)

    with pytest.raises(ValueError, match="rows"):
        split_qgkv_rows(packed[1:], num_query_groups=KV_HEADS, heads_per_group=heads_per_group, head_dim=HEAD_DIM)


# ---------------------------------------------------------------------------
# forward deltas on stand-in modules
# ---------------------------------------------------------------------------


class _TELinear(nn.Module):
    """TE linears return (output, bias)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, inputs):
        return F.linear(inputs, self.weight), None


class _GroupedLinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        for idx in range(num_experts):
            self.register_parameter(f"weight{idx}", nn.Parameter(torch.randn(out_features, in_features)))
        self.num_experts = num_experts

    def forward(self, inputs, tokens_per_expert):
        weights = torch.stack([getattr(self, f"weight{idx}") for idx in range(self.num_experts)])
        return _grouped_linear(inputs, weights, tokens_per_expert), None


def _identity_mappings():
    return SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        reduce_from_tensor_model_parallel_region=lambda x, group=None: x,
        reduce_scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )


def _fill_lora_b(adapter):
    """B is zero-initialised (no delta at step 0); fill it so the delta is observable."""
    for name, parameter in adapter.named_parameters():
        if name.endswith("_lora_B"):
            with torch.no_grad():
                parameter.normal_()


def test_gdn_lora_adds_scaled_delta_to_every_projection():
    gdn = nn.Module()
    gdn.in_proj_qkv = nn.Linear(HIDDEN, 2 * LIN_K_HEADS * LIN_HEAD_DIM + LIN_V_HEADS * LIN_HEAD_DIM, bias=False)
    gdn.in_proj_z = nn.Linear(HIDDEN, LIN_V_HEADS * LIN_HEAD_DIM, bias=False)
    gdn.in_proj_b = nn.Linear(HIDDEN, LIN_V_HEADS, bias=False)
    gdn.in_proj_a = nn.Linear(HIDDEN, LIN_V_HEADS, bias=False)
    gdn.out_proj = nn.Linear(LIN_V_HEADS * LIN_HEAD_DIM, HIDDEN, bias=False)
    attention = nn.Module()
    attention.linear_attn = gdn
    base = {name: getattr(gdn, name).weight.detach().clone() for name in qwen_lora._GDN_PROJECTIONS}

    _apply_gdn_lora(attention, _args(rank=4, alpha=8), layer_idx=1, scale=2.0, dropout=0.0)
    adapter = attention.lora_adapter
    assert adapter.kind == "gdn_attention" and adapter.hf_prefix == f"{HF_LAYER_PREFIX}1.linear_attn."
    _fill_lora_b(adapter)
    for name in qwen_lora._GDN_PROJECTIONS:
        module = getattr(gdn, name)
        inputs = torch.randn(3, module.weight.shape[1])
        a = getattr(adapter, f"{name}_lora_A")
        b = getattr(adapter, f"{name}_lora_B")
        assert a.shape == (4, module.weight.shape[1]) and b.shape == (module.weight.shape[0], 4)
        assert not a.tensor_model_parallel and not hasattr(a, "sum_gradients_across_tp_domain")
        expected = F.linear(inputs, base[name]) + 2.0 * F.linear(F.linear(inputs, a), b)
        torch.testing.assert_close(module(inputs), expected)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_qsa_lora_wraps_fused_qkv_and_proj(monkeypatch, sequence_parallel):
    monkeypatch.setattr(qwen_lora, "_tp_mappings", _identity_mappings)
    heads_per_group = HEADS // KV_HEADS
    qkv_out = KV_HEADS * (2 * heads_per_group * HEAD_DIM + 2 * HEAD_DIM)
    attention = nn.Module()
    attention.config = SimpleNamespace(
        hidden_size=HIDDEN,
        sequence_parallel=sequence_parallel,
        num_query_groups=KV_HEADS,
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        attention_output_gate=True,
    )
    attention.linear_qkv = _TELinear(HIDDEN, qkv_out)
    attention.linear_proj = _TELinear(HEADS * HEAD_DIM, HIDDEN)
    base_qkv = attention.linear_qkv.weight.detach().clone()
    base_proj = attention.linear_proj.weight.detach().clone()

    _apply_qsa_lora(attention, _args(rank=4), layer_idx=3, scale=0.5, dropout=0.0)
    adapter = attention.lora_adapter
    assert (adapter.num_query_groups, adapter.heads_per_group, adapter.head_dim) == (KV_HEADS, heads_per_group, HEAD_DIM)
    assert adapter.qkv_lora_A.sum_gradients_across_tp_domain
    assert hasattr(adapter.o_lora_B, "sum_gradients_across_tp_domain") == sequence_parallel
    _fill_lora_b(adapter)

    inputs = torch.randn(5, HIDDEN)
    output, bias = attention.linear_qkv(inputs)
    assert bias is None
    expected = F.linear(inputs, base_qkv) + 0.5 * F.linear(F.linear(inputs, adapter.qkv_lora_A), adapter.qkv_lora_B)
    torch.testing.assert_close(output, expected)

    attn_out = torch.randn(5, HEADS * HEAD_DIM)
    output, _ = attention.linear_proj(attn_out)
    expected = F.linear(attn_out, base_proj) + 0.5 * F.linear(F.linear(attn_out, adapter.o_lora_A), adapter.o_lora_B)
    torch.testing.assert_close(output, expected)


def test_qsa_lora_requires_output_gate(monkeypatch):
    monkeypatch.setattr(qwen_lora, "_tp_mappings", _identity_mappings)
    attention = nn.Module()
    attention.config = SimpleNamespace(
        hidden_size=HIDDEN, sequence_parallel=False, num_query_groups=KV_HEADS, num_attention_heads=HEADS,
        kv_channels=HEAD_DIM, attention_output_gate=False,
    )
    attention.linear_qkv = _TELinear(HIDDEN, 8)
    attention.linear_proj = _TELinear(8, HIDDEN)
    with pytest.raises(NotImplementedError, match="attention_output_gate"):
        _apply_qsa_lora(attention, _args(), layer_idx=0, scale=1.0, dropout=0.0)


def _fake_moe(num_local_experts=EXPERTS, etp=1):
    moe = nn.Module()
    moe.config = SimpleNamespace(
        hidden_size=HIDDEN, moe_ffn_hidden_size=MOE_INTER, expert_tensor_parallel_size=etp, sequence_parallel=False
    )
    experts = nn.Module()
    experts.num_local_experts = num_local_experts
    experts.linear_fc1 = _GroupedLinear(num_local_experts, HIDDEN, 2 * MOE_INTER)
    experts.linear_fc2 = _GroupedLinear(num_local_experts, MOE_INTER, HIDDEN)
    moe.experts = experts
    shared = nn.Module()
    shared.config = moe.config
    shared.linear_fc1 = _TELinear(HIDDEN, 2 * SHARED_INTER)
    shared.linear_fc2 = _TELinear(SHARED_INTER, HIDDEN)
    moe.shared_experts = shared
    return moe


def test_expert_lora_is_per_expert_and_matches_dense_reference():
    moe = _fake_moe()
    base_fc1 = torch.stack([getattr(moe.experts.linear_fc1, f"weight{i}").detach().clone() for i in range(EXPERTS)])
    base_fc2 = torch.stack([getattr(moe.experts.linear_fc2, f"weight{i}").detach().clone() for i in range(EXPERTS)])

    _apply_expert_lora(moe, _args(rank=4, expert_rank=2), layer_idx=2, scale=1.5, dropout=0.0)
    adapter = moe.experts.lora_adapter
    assert adapter.kind == "experts" and adapter.hf_prefix == f"{HF_LAYER_PREFIX}2.mlp.experts."
    assert adapter.fc1_lora_A.shape == (EXPERTS, 2, HIDDEN)
    assert adapter.fc1_lora_B.shape == (EXPERTS, 2 * MOE_INTER, 2)
    assert adapter.fc2_lora_A.shape == (EXPERTS, 2, MOE_INTER)
    assert adapter.fc2_lora_B.shape == (EXPERTS, HIDDEN, 2)
    for parameter in adapter.parameters():
        assert parameter.allreduce is False and not hasattr(parameter, "_lora_grad_sum_group")
    _fill_lora_b(adapter)

    tokens_per_expert = [2, 0, 3, 1]
    inputs = torch.randn(sum(tokens_per_expert), HIDDEN)
    output, _ = moe.experts.linear_fc1(inputs, tokens_per_expert)
    expected = []
    start = 0
    for idx, count in enumerate(tokens_per_expert):
        segment = inputs[start : start + count]
        start += count
        lora = F.linear(F.linear(segment, adapter.fc1_lora_A[idx]), adapter.fc1_lora_B[idx])
        expected.append(F.linear(segment, base_fc1[idx]) + 1.5 * lora)
    torch.testing.assert_close(output, torch.cat(expected))

    inner = torch.randn(sum(tokens_per_expert), MOE_INTER)
    output, _ = moe.experts.linear_fc2(inner, tokens_per_expert)
    expected = []
    start = 0
    for idx, count in enumerate(tokens_per_expert):
        segment = inner[start : start + count]
        start += count
        lora = F.linear(F.linear(segment, adapter.fc2_lora_A[idx]), adapter.fc2_lora_B[idx])
        expected.append(F.linear(segment, base_fc2[idx]) + 1.5 * lora)
    torch.testing.assert_close(output, torch.cat(expected))

    output.sum().backward()
    assert adapter.fc2_lora_A.grad is not None and adapter.fc2_lora_B.grad is not None
    # expert 1 received no tokens: its adapter gets a zero gradient, not a missing one
    assert torch.count_nonzero(adapter.fc2_lora_A.grad[1]) == 0


def test_expert_lora_requires_etp1():
    with pytest.raises(NotImplementedError, match="ETP=1"):
        _apply_expert_lora(_fake_moe(etp=2), _args(), layer_idx=0, scale=1.0, dropout=0.0)


def test_shared_expert_lora_delta(monkeypatch):
    monkeypatch.setattr(qwen_lora, "_tp_mappings", _identity_mappings)
    moe = _fake_moe()
    shared = moe.shared_experts
    base_fc1 = shared.linear_fc1.weight.detach().clone()
    _apply_shared_expert_lora(shared, _args(rank=4), layer_idx=0, scale=1.0, dropout=0.0)
    adapter = shared.lora_adapter
    assert adapter.hf_prefix == f"{HF_LAYER_PREFIX}0.mlp.shared_expert."
    _fill_lora_b(adapter)
    inputs = torch.randn(2, HIDDEN)
    output, _ = shared.linear_fc1(inputs)
    torch.testing.assert_close(
        output, F.linear(inputs, base_fc1) + F.linear(F.linear(inputs, adapter.fc1_lora_A), adapter.fc1_lora_B)
    )


def test_grouped_linear_uses_expert_token_boundaries():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    torch.testing.assert_close(_grouped_linear(inputs, weights, [1, 2]), torch.tensor([[1.0], [4.0], [6.0]]))


def test_full_recompute_keeps_native_lora_in_autograd_graph():
    model = nn.Module()
    model.embedding = nn.Embedding.from_pretrained(torch.ones(8, 4), freeze=True)
    model.adapter = nn.Parameter(torch.ones(4, 4))
    model.config = SimpleNamespace(recompute_granularity="full")
    model.pre_process = True
    _enable_full_recompute_input_grads(model)

    model.train()
    hidden_states = model.embedding(torch.tensor([[1, 2]]))
    output = checkpoint(lambda inputs: inputs @ model.adapter, hidden_states, use_reentrant=True)
    output.sum().backward()
    assert hidden_states.requires_grad
    assert model.embedding.weight.grad is None
    torch.testing.assert_close(model.adapter.grad, torch.full_like(model.adapter, 2.0))
    model.eval()
    assert not model.embedding(torch.tensor([[1, 2]])).requires_grad


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _parameter(*shape):
    return nn.Parameter(torch.arange(torch.tensor(shape).prod()).reshape(shape).float())


def _gdn_adapter(layer_idx, rank=4):
    adapter = Qwen38NextLoRAAdapter("gdn_attention", f"{HF_LAYER_PREFIX}{layer_idx}.linear_attn.")
    for name, out in (("in_proj_qkv", 48), ("in_proj_z", 32), ("in_proj_b", 4), ("in_proj_a", 4)):
        adapter.register_parameter(f"{name}_lora_A", _parameter(rank, HIDDEN))
        adapter.register_parameter(f"{name}_lora_B", _parameter(out, rank))
    adapter.register_parameter("out_proj_lora_A", _parameter(rank, 32))
    adapter.register_parameter("out_proj_lora_B", _parameter(HIDDEN, rank))
    return adapter


def _qsa_adapter(layer_idx, rank=4):
    adapter = Qwen38NextLoRAAdapter("qsa_attention", f"{HF_LAYER_PREFIX}{layer_idx}.self_attn.")
    heads_per_group = HEADS // KV_HEADS
    adapter.num_query_groups, adapter.heads_per_group, adapter.head_dim = KV_HEADS, heads_per_group, HEAD_DIM
    qkv_out = KV_HEADS * (2 * heads_per_group * HEAD_DIM + 2 * HEAD_DIM)
    adapter.register_parameter("qkv_lora_A", _parameter(rank, HIDDEN))
    adapter.register_parameter("qkv_lora_B", _parameter(qkv_out, rank))
    adapter.register_parameter("o_lora_A", _parameter(rank, HEADS * HEAD_DIM))
    adapter.register_parameter("o_lora_B", _parameter(HIDDEN, rank))
    return adapter


def _experts_adapter(layer_idx, rank=2):
    adapter = Qwen38NextLoRAAdapter("experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.experts.")
    adapter.register_parameter("fc1_lora_A", _parameter(EXPERTS, rank, HIDDEN))
    adapter.register_parameter("fc1_lora_B", _parameter(EXPERTS, 2 * MOE_INTER, rank))
    adapter.register_parameter("fc2_lora_A", _parameter(EXPERTS, rank, MOE_INTER))
    adapter.register_parameter("fc2_lora_B", _parameter(EXPERTS, HIDDEN, rank))
    return adapter


def _shared_adapter(layer_idx, rank=4):
    adapter = Qwen38NextLoRAAdapter("shared_experts", f"{HF_LAYER_PREFIX}{layer_idx}.mlp.shared_expert.")
    adapter.register_parameter("fc1_lora_A", _parameter(rank, HIDDEN))
    adapter.register_parameter("fc1_lora_B", _parameter(2 * SHARED_INTER, rank))
    adapter.register_parameter("fc2_lora_A", _parameter(rank, SHARED_INTER))
    adapter.register_parameter("fc2_lora_B", _parameter(HIDDEN, rank))
    return adapter


def _model_with_adapters(*, include_shared=True, expert_rank=2):
    """Two layers: layer 2 (GDN) and layer 3 (QSA), each with routed + shared experts."""
    adapters = [_gdn_adapter(2), _experts_adapter(2, expert_rank), _shared_adapter(2), _qsa_adapter(3), _experts_adapter(3, expert_rank)]
    if include_shared:
        adapters.append(_shared_adapter(3))
    model = nn.Module()
    model.adapters = nn.ModuleList(adapters)
    model.decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(layer_number=3, self_attention=SimpleNamespace(linear_attn=object()), mlp=SimpleNamespace()),
            SimpleNamespace(layer_number=4, self_attention=SimpleNamespace(), mlp=SimpleNamespace()),
        ]
    )
    return model


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(
        qwen_lora,
        "_parallel_state",
        lambda: SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_expert_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_group=lambda: None,
            get_expert_model_parallel_group=lambda: None,
        ),
    )


def test_native_export_matches_hf_targets_and_pads_expert_rank(single_rank):
    """A wrong expert dim is a shape mismatch in SGLang's pool; a wrong HF name is silently dropped."""
    model = _model_with_adapters(expert_rank=2)
    targets = _default_hf_targets()
    adapter_targets = resolve_qwen3_8_next_adapter_targets(targets, canonical=False, experts_shared_outer_loras=False)
    chunks = list(export_qwen3_8_next_lora_hf_chunks([model]))
    assert len(chunks) == 6
    names = [name for chunk in chunks for name, _ in chunk]
    assert len(names) == len(set(names))
    exported = get_adapter_target_modules(names)
    assert all(any(matches_lora_target(name, target) for target in adapter_targets) for name in exported)
    assert all(any(matches_lora_target(name, target) for name in exported) for target in adapter_targets)
    assert all(tensor.dtype == torch.bfloat16 and tensor.is_contiguous() for chunk in chunks for _, tensor in chunk)

    gdn = dict(chunks[0])
    prefix = f"{HF_LAYER_PREFIX}2.linear_attn."
    assert gdn[f"{prefix}in_proj_qkv.lora_A.weight"].shape == (4, HIDDEN)
    assert gdn[f"{prefix}in_proj_qkv.lora_B.weight"].shape == (48, 4)
    assert gdn[f"{prefix}in_proj_a.lora_B.weight"].shape == (4, 4)
    assert gdn[f"{prefix}out_proj.lora_A.weight"].shape == (4, 32)

    experts = dict(chunks[1])
    prefix = f"{HF_LAYER_PREFIX}2.mlp.experts."
    # per-expert: the expert axis is the global expert count (not 1), rank padded 2 -> 4
    assert experts[f"{prefix}gate_up_proj.lora_A.weight"].shape == (EXPERTS, 4, HIDDEN)
    assert experts[f"{prefix}gate_up_proj.lora_B.weight"].shape == (EXPERTS, 2 * MOE_INTER, 4)
    assert experts[f"{prefix}down_proj.lora_A.weight"].shape == (EXPERTS, 4, MOE_INTER)
    assert experts[f"{prefix}down_proj.lora_B.weight"].shape == (EXPERTS, HIDDEN, 4)
    source = model.adapters[1]
    torch.testing.assert_close(experts[f"{prefix}gate_up_proj.lora_A.weight"][:, :2].float(), source.fc1_lora_A.detach())
    assert torch.count_nonzero(experts[f"{prefix}gate_up_proj.lora_A.weight"][:, 2:]) == 0
    assert torch.count_nonzero(experts[f"{prefix}down_proj.lora_B.weight"][:, :, 2:]) == 0
    # padded A rows x padded B columns contribute nothing: A^T B equals the unpadded product
    a_pad = experts[f"{prefix}down_proj.lora_A.weight"].float()
    b_pad = experts[f"{prefix}down_proj.lora_B.weight"].float()
    torch.testing.assert_close(
        torch.bmm(b_pad, a_pad), torch.bmm(source.fc2_lora_B.detach(), source.fc2_lora_A.detach())
    )

    shared = dict(chunks[2])
    prefix = f"{HF_LAYER_PREFIX}2.mlp.shared_expert."
    assert shared[f"{prefix}gate_proj.lora_B.weight"].shape == (SHARED_INTER, 4)
    assert shared[f"{prefix}up_proj.lora_B.weight"].shape == (SHARED_INTER, 4)
    torch.testing.assert_close(shared[f"{prefix}gate_proj.lora_A.weight"], shared[f"{prefix}up_proj.lora_A.weight"])
    assert shared[f"{prefix}down_proj.lora_A.weight"].shape == (4, SHARED_INTER)

    qsa = dict(chunks[3])
    prefix = f"{HF_LAYER_PREFIX}3.self_attn."
    assert qsa[f"{prefix}q_proj.lora_B.weight"].shape == (HEADS * 2 * HEAD_DIM, 4)
    assert qsa[f"{prefix}k_proj.lora_B.weight"].shape == (KV_HEADS * HEAD_DIM, 4)
    assert qsa[f"{prefix}v_proj.lora_B.weight"].shape == (KV_HEADS * HEAD_DIM, 4)
    for name in ("q_proj", "k_proj", "v_proj"):
        torch.testing.assert_close(qsa[f"{prefix}{name}.lora_A.weight"].float(), model.adapters[3].qkv_lora_A.detach())
    assert qsa[f"{prefix}o_proj.lora_A.weight"].shape == (4, HEADS * HEAD_DIM)
    # row de-interleave agrees with re-packing through the mbridge formula
    repacked = _mbridge_pack_qkv(
        qsa[f"{prefix}q_proj.lora_B.weight"].float(),
        qsa[f"{prefix}k_proj.lora_B.weight"].float(),
        qsa[f"{prefix}v_proj.lora_B.weight"].float(),
        num_kv_heads=KV_HEADS, heads_per_group=HEADS // KV_HEADS, head_dim=HEAD_DIM,
    )
    # the export is bf16; compare against the bf16-rounded source
    torch.testing.assert_close(repacked, model.adapters[3].qkv_lora_B.detach().to(torch.bfloat16).float())


def test_native_export_rejects_missing_shared_expert_adapter(single_rank):
    model = _model_with_adapters(include_shared=False)
    with pytest.raises(RuntimeError, match="adapter layout is incomplete"):
        list(export_qwen3_8_next_lora_hf_chunks([model]))


def test_sglang_target_leaves_are_known_modules():
    """SGLang normalises the leaf names; every leaf Miles sends must map to a pool module."""
    leaves = {target.rsplit(".", 1)[-1] for target in _default_hf_targets()}
    sglang_params_mapping = {
        "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "o_proj": "o_proj",
        "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "gate_up_proj": "gate_up_proj", "down_proj": "down_proj",
        "in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz", "in_proj_b": "in_proj_ba", "in_proj_a": "in_proj_ba",
        "out_proj": "out_proj",
    }
    assert {sglang_params_mapping[leaf] for leaf in leaves} == {
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj",
    }
