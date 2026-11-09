"""--num-critic-epochs / --critic-updates-per-step and --critic-freeze-attention (SAO 16a9bea4 port)."""

import argparse

import pytest
import torch

from miles.backends.training_utils.critic_freeze import freeze_attention_parameters
from miles.utils.arguments import get_miles_extra_args_provider


def _parse(argv):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    ns, _ = parser.parse_known_args(argv)
    return ns


def test_defaults_unchanged():
    ns = _parse([])
    assert ns.num_critic_epochs == 1
    assert ns.critic_freeze_attention is False
    assert not hasattr(ns, "critic_updates_per_step")


@pytest.mark.parametrize("flag", ["--num-critic-epochs", "--critic-updates-per-step"])
def test_alias_shares_one_dest(flag):
    assert _parse([flag, "2"]).num_critic_epochs == 2


def test_last_flag_wins_single_implementation():
    assert _parse(["--num-critic-epochs", "2", "--critic-updates-per-step", "3"]).num_critic_epochs == 3


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = torch.nn.Linear(4, 12)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _Attn()
        self.mlp = torch.nn.Linear(4, 4)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.output_layer = torch.nn.Linear(4, 1)


def test_freeze_attention_only_touches_attention():
    m = _Model()
    tensors, elements = freeze_attention_parameters(m)
    assert tensors == 4 and elements == 2 * (4 * 12 + 12)
    for name, p in m.named_parameters():
        assert p.requires_grad == ("self_attention" not in name), name


def test_freeze_attention_rejects_no_match():
    with pytest.raises(RuntimeError):
        freeze_attention_parameters(torch.nn.Linear(2, 2))
