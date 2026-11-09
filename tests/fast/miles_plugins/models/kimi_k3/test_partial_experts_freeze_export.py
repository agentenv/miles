"""Freeze set, HF export names and the Direct weight-update filter under --partial-experts-selection.

The Megatron-dependent modules are read as source (one function via ast, one module via its file) so the
tests run on CPU without megatron or sglang installed.
"""

import ast
import importlib.util
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.miles_plugins.models.kimi_k3.partial_experts_fixtures import make_selection

from miles.utils.types import ParamInfo
from miles_plugins.models.kimi_k3.partial_experts import (
    TRAINABLE_PARAM_ATTR,
    freeze_all_but_trainable_experts,
    hf_expert_id_for_export,
    is_split_expert_param_name,
    is_trainable_expert_param_name,
)
from miles_plugins.models.kimi_k3.permute import SlotLayout, set_active_slot_layout

REPO = Path(__file__).resolve().parents[5]
E, K, EP, LAYERS = 64, 16, 4, [1, 2]
N_LOCAL, K_LOCAL = E // EP, K // EP


@pytest.fixture
def layout():
    layout = SlotLayout.from_selection(make_selection(num_experts=E, num_trainable=K, ep_size=EP, layers=LAYERS))
    set_active_slot_layout(layout)
    yield layout
    set_active_slot_layout(None)


class _Half(torch.nn.Module):
    def __init__(self, n, in_f=8, out_f=4):
        super().__init__()
        self.linear_fc1 = torch.nn.Module()
        self.linear_fc2 = torch.nn.Module()
        for i in range(n):
            self.linear_fc1.register_parameter(f"weight{i}", torch.nn.Parameter(torch.zeros(2 * out_f, in_f)))
            self.linear_fc2.register_parameter(f"weight{i}", torch.nn.Parameter(torch.zeros(in_f, out_f)))


def _fake_model():
    """The parameter tree of one MoE layer plus the pieces the freeze must reach."""
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Linear(4, 4)
    layer.mlp = torch.nn.Module()
    layer.mlp.router = torch.nn.Linear(4, E, bias=False)
    layer.mlp.router.register_buffer("expert_bias", torch.zeros(E))
    layer.mlp.fc1_latent_proj = torch.nn.Linear(4, 4)
    layer.mlp.shared_experts = torch.nn.Module()
    layer.mlp.shared_experts.linear_fc1 = torch.nn.Linear(4, 4)
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.trainable = _Half(K_LOCAL)
    layer.mlp.experts.frozen = _Half(N_LOCAL - K_LOCAL)
    model.decoder.layers.append(layer)
    model.embedding = torch.nn.Embedding(10, 4)
    return model


def test_freeze_keeps_exactly_the_trainable_experts():
    model = _fake_model()
    trainable, frozen = freeze_all_but_trainable_experts(model)
    requires_grad = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {
        f"decoder.layers.0.mlp.experts.trainable.linear_fc{fc}.weight{i}" for fc in (1, 2) for i in range(K_LOCAL)
    }
    assert requires_grad == expected
    assert trainable == len(expected) and trainable + frozen == len(list(model.parameters()))
    assert all(getattr(p, TRAINABLE_PARAM_ATTR, False) == p.requires_grad for p in model.parameters())
    assert not model.decoder.layers[0].mlp.router.weight.requires_grad


def test_name_predicates():
    assert is_trainable_expert_param_name("module.module.decoder.layers.3.mlp.experts.trainable.linear_fc1.weight5")
    assert not is_trainable_expert_param_name("module.module.decoder.layers.3.mlp.experts.frozen.linear_fc1.weight5")
    assert is_split_expert_param_name("decoder.layers.3.mlp.experts.frozen.linear_fc2.weight0")
    assert not is_split_expert_param_name("decoder.layers.3.mlp.experts.linear_fc2.weight0")
    assert not is_trainable_expert_param_name("decoder.layers.3.mlp.shared_experts.linear_fc1.weight")


def _load_converter():
    path = REPO / "miles/backends/megatron_utils/megatron_to_hf/kimi_k3.py"
    spec = importlib.util.spec_from_file_location("kimi_k3_converter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_kimi_k3_to_hf


def test_export_names_carry_original_expert_ids(layout):
    convert = _load_converter()
    layer = LAYERS[1]
    exported = {}
    for ep_rank in range(EP):
        for j in range(K_LOCAL):
            global_slot = ep_rank * N_LOCAL + j  # what named_params_and_buffers makes of local index j
            param = torch.zeros(4, 2)
            for hf_name, _ in convert(None, f"module.module.decoder.layers.{layer}.mlp.experts.trainable.linear_fc1.weight{global_slot}", param):
                exported.setdefault(hf_name, []).append(global_slot)
            (fc2_name, _), = convert(None, f"module.module.decoder.layers.{layer}.mlp.experts.trainable.linear_fc2.weight{global_slot}", param)
            exported.setdefault(fc2_name, []).append(global_slot)
    ids = {int(re.search(r"experts\.(\d+)\.", name).group(1)) for name in exported}
    assert ids == set(layout.perm[layer][[s for s in range(E) if layout.is_trainable_slot(s)]].tolist())
    assert ids == {e for r in range(EP) for e in layout.trainable_original_ids(layer, r)}
    assert all(len(slots) == 1 for slots in exported.values())  # every HF name is produced by exactly one slot
    assert {name.rsplit(".", 2)[-2] for name in exported} == {"w1", "w2", "w3"}
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts."
    assert all(name.startswith(prefix) for name in exported)


def test_export_of_unsplit_names_is_unchanged_and_frozen_half_is_refused(layout):
    convert = _load_converter()
    (name, _), = convert(None, "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight7", torch.zeros(2, 2))
    assert name == "language_model.model.layers.1.block_sparse_moe.experts.7.w2.weight"
    with pytest.raises(ValueError):
        convert(None, "module.module.decoder.layers.1.mlp.experts.frozen.linear_fc2.weight0", torch.zeros(2, 2))
    with pytest.raises(AssertionError, match="frozen"):
        hf_expert_id_for_export(layout, 1, K_LOCAL)  # local slot k_local of rank 0 is frozen


def test_export_without_active_layout_fails_loudly():
    set_active_slot_layout(None)
    with pytest.raises(AssertionError, match="partial-experts-selection"):
        _load_converter()(None, "module.module.decoder.layers.1.mlp.experts.trainable.linear_fc2.weight0", torch.zeros(2, 2))


def _load_partition_expert_infos():
    path = REPO / "miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_partition_expert_infos")
    namespace = {"re": re, "ParamInfo": ParamInfo, "Sequence": Sequence}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_partition_expert_infos"]


def _info(name):
    return ParamInfo(name=name, dtype=torch.bfloat16, shape=torch.Size([2, 2]), attrs={}, size=8, src_rank=0)


def _direct_filter(names, *, partial):
    """What _get_megatron_local_param_infos keeps of named_params_and_buffers under the flag."""
    return [n for n in names if not partial or is_trainable_expert_param_name(n)]


def test_direct_sends_only_the_trainable_half_and_partitions_present_slots_evenly():
    partition = _load_partition_expert_infos()
    ep_rank, edp = 2, 2
    names = ["module.module.decoder.layers.1.mlp.router.weight", "module.module.decoder.layers.1.mlp.router.expert_bias"]
    for fc in (1, 2):
        for j in range(K_LOCAL):
            names.append(f"module.module.decoder.layers.1.mlp.experts.trainable.linear_fc{fc}.weight{ep_rank * N_LOCAL + j}")
        for j in range(N_LOCAL - K_LOCAL):
            names.append(f"module.module.decoder.layers.1.mlp.experts.frozen.linear_fc{fc}.weight{ep_rank * N_LOCAL + j}")
    kept = _direct_filter(names, partial=True)
    assert len(kept) == 2 * K_LOCAL and all(".experts.trainable." in n for n in kept)
    assert _direct_filter(names, partial=False) == names

    partitions = partition([_info(n) for n in kept], num_local_experts=N_LOCAL, edp_size=edp)
    sizes = [len(p) for p in partitions]
    assert sum(sizes) == len(kept) and max(sizes) - min(sizes) <= 2  # fc1+fc2 of one slot stay together
    for part in partitions:
        slots = {int(re.search(r"weight(\d+)$", i.name).group(1)) for i in part}
        assert all(i.name.replace("linear_fc1", "linear_fc2") in {x.name for x in part} for i in part if "fc1" in i.name)
        assert len(part) == 2 * len(slots)


def test_partition_of_all_local_experts_is_unchanged():
    """Without the flag every local slot is present and the split is the classic contiguous one."""
    partition = _load_partition_expert_infos()
    infos = [_info(f"module.module.decoder.layers.1.mlp.experts.linear_fc1.weight{s}") for s in range(N_LOCAL)]
    partitions = partition(infos, num_local_experts=N_LOCAL, edp_size=3)
    assert [len(p) for p in partitions] == [6, 5, 5]
    assert [int(i.name[-1]) if i.name[-2] == "t" else int(i.name[-2:]) for i in partitions[0]] == list(range(6))


def test_split_moe_spec_swaps_the_experts_builder():
    pytest.importorskip("megatron")
    pytest.importorskip("transformer_engine")
    import functools

    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from miles_plugins.models.kimi_k3.split_moe import PartiallyTrainableGroupedMLP, partial_experts_spec

    submodules = SimpleNamespace()
    spec = partial_experts_spec(functools.partial(TEGroupedMLP, submodules=submodules), 4)
    assert spec.func is PartiallyTrainableGroupedMLP
    assert spec.keywords == {"submodules": submodules, "num_local_trainable": 4}
