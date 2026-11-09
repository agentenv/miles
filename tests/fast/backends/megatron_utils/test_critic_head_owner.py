"""Critic value head placement for bridge-built multimodal wrappers (Qwen3.5 tied).

Regression: the critic put ``LinearForLastLayer`` on the ``Qwen3VLModel``
wrapper, leaving ``language_model.output_layer`` (vocab head) live and making
the HF load demand ``lm_head.weight``, absent in tied checkpoints.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.megatron_utils.critic_head import critic_head_owner, iter_value_heads


class _Head(torch.nn.Linear):
    pass


class _GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 2)
        self.output_layer = torch.nn.Linear(2, 5)


class _VL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = torch.nn.Linear(2, 2)
        self.language_model = _GPT()


def _names(m):
    return sorted(n for n, _ in m.named_parameters())


def test_plain_gpt_owner_is_model_itself():
    m = _GPT()
    assert critic_head_owner(m) is m


def test_vl_wrapper_owner_is_language_model():
    m = _VL()
    assert critic_head_owner(m) is m.language_model


def test_vl_without_head_or_none_language_model_falls_back():
    m = _VL()
    del m.language_model.output_layer
    assert critic_head_owner(m) is m
    m2 = torch.nn.Module()
    m2.language_model = None
    assert critic_head_owner(m2) is m2


def _hide(chunk):
    heads = list(iter_value_heads(chunk, _Head))
    for parent, name, _ in heads:
        delattr(parent, name)
    return heads


def test_vl_value_head_replaces_vocab_head_and_hides_fully():
    m = _VL()
    critic_head_owner(m).output_layer = _Head(2, 1)
    assert m.language_model.output_layer.out_features == 1
    before = _names(m)
    heads = _hide(m)
    # No output_layer parameter left -> bridge needs no lm_head.weight mapping.
    assert not any("output_layer" in n for n in _names(m))
    for parent, name, head in heads:
        setattr(parent, name, head)
    assert _names(m) == before


def test_plain_gpt_hide_matches_legacy_top_level_behaviour():
    m = _GPT()
    m.output_layer = _Head(2, 1)
    legacy = [(m, n, h) for n, h in m.named_children() if isinstance(h, _Head)]
    assert [(p, n) for p, n, _ in iter_value_heads(m, _Head)] == [(p, n) for p, n, _ in legacy]


def test_actor_vocab_head_never_hidden():
    m = _VL()
    assert list(iter_value_heads(m, _Head)) == []
    assert list(iter_value_heads(_GPT(), _Head)) == []
