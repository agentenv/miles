"""A ChainedOptimizer stub child (Megatron's ``is_stub_optimizer``: no inner
optimizer, no parameters) is skipped by every full-parameter traversal."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps


def _child(params):
    inner = SimpleNamespace(param_groups=[{"params": params}], state={})
    return SimpleNamespace(optimizer=inner, is_stub_optimizer=False)


def _stub():
    return SimpleNamespace(optimizer=None, is_stub_optimizer=True)


def test_stub_child_is_skipped_and_layout_logged(caplog):
    fps._LOGGED_OPTIMIZER_LAYOUTS.clear()
    dense, expert = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))
    chained = SimpleNamespace(chained_optimizers=[_child([dense]), _stub(), _child([expert])])
    with caplog.at_level(logging.WARNING, logger=fps.__name__):
        bases = fps._optimizer_bases(chained)
        owned = fps._optimizer_owned_parameter_ids(chained)
    assert len(bases) == 2 and all(base is not None for base in bases)
    assert {id(dense), id(expert)} <= set(owned)
    lines = [r.getMessage() for r in caplog.records if "[full-parameter-optimizer]" in r.getMessage()]
    assert len(lines) == 1  # logged once per layout
    assert "stub=True, param_groups=0" in lines[0] and lines[0].count("stub=False, param_groups=1") == 2


def test_only_stubs_still_refused():
    fps._LOGGED_OPTIMIZER_LAYOUTS.clear()
    with pytest.raises(RuntimeError, match="no base optimizer"):
        fps._optimizer_bases(SimpleNamespace(chained_optimizers=[_stub()]))


def test_unchained_optimizer_is_its_own_base():
    parameter = torch.nn.Parameter(torch.zeros(1))
    child = _child([parameter])
    assert fps._optimizer_bases(child) == (child.optimizer,)
