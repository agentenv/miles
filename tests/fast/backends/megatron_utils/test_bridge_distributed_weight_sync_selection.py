from argparse import Namespace

import pytest

from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as tensor_update_module
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


def test_yeto_policy_weight_versions_follow_rollout_ids():
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = _args(rollout_weight_version_format="yeto-policy", start_rollout_id=0)

    updater.weight_version = 1
    assert updater.published_weight_version() == "yeto:0"
    updater.weight_version = 2
    assert updater.published_weight_version() == "yeto:1"


def test_yeto_policy_weight_versions_honor_resume_offset():
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = _args(rollout_weight_version_format="yeto-policy", start_rollout_id=7)
    updater.weight_version = 1
    assert updater.published_weight_version() == "yeto:7"


def test_bridge_send_publishes_yeto_policy_token_to_every_engine_path(monkeypatch):
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = _args(rollout_weight_version_format="yeto-policy", start_rollout_id=0)
    updater.weight_version = 1
    updater._ipc_engine = object()
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = object()
    updater.use_distribute = True
    updater._is_distributed_src_rank = True
    updater._group_name = "miles"
    updater._model_update_groups = object()
    updater.distributed_rollout_engines = [object()]
    observed = []

    def fake_colocated(**kwargs):
        observed.append(("colocated", kwargs["weight_version"]))
        return ["colocated-ref"], "keepalive"

    def fake_distributed(_group_name, _groups, version, _engines, _tensors):
        observed.append(("distributed", version))
        return ["distributed-ref"]

    monkeypatch.setattr(tensor_update_module, "_send_to_colocated_engine", fake_colocated)
    monkeypatch.setattr(tensor_update_module, "update_weights_from_distributed", fake_distributed)

    refs, keepalive = updater._send_base_params([])

    assert observed == [("colocated", "yeto:0"), ("distributed", "yeto:0")]
    assert refs == ["colocated-ref", "distributed-ref"]
    assert keepalive == "keepalive"


@pytest.mark.parametrize(
    ("weight_version", "start_rollout_id"),
    [(0, 0), (1, None), (1, -1), (1, True)],
)
def test_yeto_policy_weight_versions_fail_closed(weight_version, start_rollout_id):
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = _args(
        rollout_weight_version_format="yeto-policy",
        start_rollout_id=start_rollout_id,
    )
    updater.weight_version = weight_version
    with pytest.raises(RuntimeError):
        updater.published_weight_version()
