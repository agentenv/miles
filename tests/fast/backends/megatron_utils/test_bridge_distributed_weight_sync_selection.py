from argparse import Namespace

import pytest

from miles.backends.megatron_utils.actor import _select_update_weight_cls
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor


def _args(**overrides):
    values = {
        "bridge_distributed_weight_sync": False,
        "colocate": False,
        "megatron_to_hf_mode": "bridge",
        "update_weight_transfer_mode": "broadcast",
    }
    values.update(overrides)
    return Namespace(**values)


def test_bridge_distributed_full_parameter_selects_bridge_iterator_path():
    selected = _select_update_weight_cls(
        _args(bridge_distributed_weight_sync=True),
        is_lora=False,
    )

    assert selected is UpdateWeightFromTensor


@pytest.mark.parametrize(
    ("overrides", "is_lora", "message"),
    [
        ({"colocate": True}, False, "only for non-colocated"),
        ({"megatron_to_hf_mode": "raw"}, False, "requires --megatron-to-hf-mode bridge"),
        ({"update_weight_transfer_mode": "p2p"}, False, "requires --update-weight-transfer-mode broadcast"),
        ({}, True, "only for full-parameter"),
    ],
)
def test_bridge_distributed_full_parameter_rejects_incompatible_modes(overrides, is_lora, message):
    with pytest.raises(AssertionError, match=message):
        _select_update_weight_cls(
            _args(bridge_distributed_weight_sync=True, **overrides),
            is_lora=is_lora,
        )


def test_default_non_colocated_broadcast_selection_is_unchanged():
    assert _select_update_weight_cls(_args(), is_lora=False) is UpdateWeightFromDistributed
