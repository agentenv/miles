"""Rollout routing replay (R3) maps replay streams to the right MoE layers.

A hybrid model (Nemotron-H: Mamba, attention and MoE layers) keeps
``moe_layer_freq = 1`` in its bridge config, so the frequency rule maps all
52 layers to 23 router streams.  The routers' own layer numbers must decide
then, and a mapping that still does not fit must fail loudly.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def replay_utils(monkeypatch):
    block = types.ModuleType("megatron.core.transformer.transformer_block")
    layer = types.ModuleType("megatron.core.transformer.transformer_layer")
    block.get_num_layers_to_build = lambda config, vp_stage=None: config.num_layers
    layer.get_transformer_layer_offset = lambda config, vp_stage=None: 0
    for name in ("megatron", "megatron.core", "megatron.core.transformer"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_block", block)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_layer", layer)
    monkeypatch.delitem(sys.modules, "miles.backends.megatron_utils.replay_utils", raising=False)
    return importlib.import_module("miles.backends.megatron_utils.replay_utils")


class _Stream:
    def __init__(self):
        self.recorded = []

    def record(self, data):
        self.recorded.append(data)


class _Router(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.routing_replay = object()


class _Model(torch.nn.Module):
    def __init__(self, pattern, *, moe_layer_freq=1):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Router(index + 1) if symbol == "E" else torch.nn.Identity() for index, symbol in enumerate(pattern)])
        self.module = SimpleNamespace(config=SimpleNamespace(num_layers=len(pattern), moe_layer_freq=moe_layer_freq))


# A hybrid pattern with the Nemotron-3 Nano 30B-A3B counts: 52 layers, 23 MoE (E);
# the order of Mamba (M) and attention (*) layers here is illustrative.
NANO = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


def _data(layers, tokens=3, topk=6):
    # value = layer id, so every stream shows which layer it received
    return torch.arange(layers).view(1, layers, 1).expand(tokens, layers, topk).clone()


def test_hybrid_routers_get_their_own_layers(replay_utils):
    assert len(NANO) == 52 and NANO.count("E") == 23
    model = _Model(NANO)
    streams = [_Stream() for _ in range(23)]
    replay_utils.register_replay_list_moe(streams, _data(52), models=[model])
    want = [index for index, symbol in enumerate(NANO) if symbol == "E"]
    assert [int(stream.recorded[0][0, 0]) for stream in streams] == want


def test_a_dense_moe_model_keeps_the_frequency_rule(replay_utils):
    model = _Model("EEEE")
    model.layers[2].layer_number = 99  # unused: the frequency rule fits
    streams = [_Stream() for _ in range(4)]
    replay_utils.register_replay_list_moe(streams, _data(4), models=[model])
    assert [int(stream.recorded[0][0, 0]) for stream in streams] == [0, 1, 2, 3]

    every_other = _Model("EMEM", moe_layer_freq=2)
    streams = [_Stream() for _ in range(2)]
    replay_utils.register_replay_list_moe(streams, _data(4), models=[every_other])
    assert [int(stream.recorded[0][0, 0]) for stream in streams] == [0, 2]


def test_a_mapping_that_does_not_fit_fails_loudly(replay_utils):
    model = _Model(NANO)
    with pytest.raises(RuntimeError, match="22 replay streams but 52 local MoE layers"):
        replay_utils.register_replay_list_moe([_Stream() for _ in range(22)], _data(52), models=[model])
    with pytest.raises(RuntimeError, match="outside the rollout"):
        replay_utils.register_replay_list_moe([_Stream() for _ in range(23)], _data(40), models=[model])
