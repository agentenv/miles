from argparse import Namespace

import pytest

from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as tensor_update_module
from miles.backends.megatron_utils.actor import _published_weight_version, _select_update_weight_cls
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    broadcast as distributed_update_module,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
    _colocated_engine_count,
)


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


def test_actor_reads_published_version_by_capability():
    updater = Namespace(
        weight_version=99,
        published_weight_version=lambda: "yeto:7",
    )

    assert _published_weight_version(updater) == "yeto:7"
    assert _published_weight_version(Namespace(weight_version=3)) == "3"


def test_bridge_tp2_to_dedicated_tp2_is_never_misclassified_as_colocated():
    args = _args(
        bridge_distributed_weight_sync=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
    )

    # RolloutServer reports offsets relative to its own placement-group slice,
    # so the dedicated pair begins at zero even though it is physically after
    # the actor pair.
    assert _colocated_engine_count(args, [0], [2]) == 0


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


def test_raw_distributed_yeto_policy_weight_versions_follow_rollout_ids():
    updater = object.__new__(UpdateWeightFromDistributed)
    updater.args = _args(
        megatron_to_hf_mode="raw",
        rollout_weight_version_format="yeto-policy",
        start_rollout_id=4,
    )

    updater.weight_version = 1
    assert updater.published_weight_version() == "yeto:4"
    updater.weight_version = 2
    assert updater.published_weight_version() == "yeto:5"


def test_raw_distributed_send_publishes_yeto_policy_token(monkeypatch):
    observed = []

    class RemoteCall:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def remote(self):
            observed.append(self.name)
            return self.value

    updater = object.__new__(UpdateWeightFromDistributed)
    updater.args = _args(
        megatron_to_hf_mode="raw",
        rollout_weight_version_format="yeto-policy",
        start_rollout_id=0,
    )
    updater.weight_version = 1
    updater._group_name = "miles"
    updater._model_update_groups = object()
    updater.rollout_engines = [object()]
    updater.rollout_engine_lock = Namespace(
        acquire=RemoteCall("lock", True),
        release=RemoteCall("unlock", True),
    )
    converted_named_tensors = [("weight", object())]

    def fake_distributed(_group_name, _groups, version, _engines, _tensors):
        observed.append(("distributed", version))
        return [True]

    monkeypatch.setattr(distributed_update_module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        distributed_update_module,
        "update_weights_from_distributed",
        fake_distributed,
    )

    updater._update_weight_implementation(converted_named_tensors)

    assert observed == ["lock", ("distributed", "yeto:0"), "unlock"]
    assert converted_named_tensors == []


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

    class RemoteCall:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def remote(self):
            observed.append(self.name)
            return self.value

    updater.rollout_engine_lock = Namespace(
        acquire=RemoteCall("lock", True),
        release=RemoteCall("unlock", True),
    )
    monkeypatch.setattr(tensor_update_module.ray, "get", lambda value: value)

    def fake_colocated(**kwargs):
        observed.append(("colocated", kwargs["weight_version"]))
        return ["colocated-ref"], "keepalive"

    def fake_distributed(_group_name, _groups, version, _engines, _tensors):
        observed.append(("distributed", version))
        return ["distributed-ref"]

    monkeypatch.setattr(tensor_update_module, "_send_to_colocated_engine", fake_colocated)
    monkeypatch.setattr(tensor_update_module, "update_weights_from_distributed", fake_distributed)

    refs, keepalive = updater._send_base_params([])
    updater._release_distributed_engine_lock()

    assert observed == [
        ("colocated", "yeto:0"),
        "lock",
        ("distributed", "yeto:0"),
        "unlock",
    ]
    assert refs == ["colocated-ref", "distributed-ref"]
    assert keepalive == "keepalive"


def test_bridge_distributed_only_engine_gets_full_publication_lifecycle(monkeypatch):
    observed = []

    class RemoteCall:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            observed.append((self.name, args, kwargs))
            return True

    engine = Namespace(
        pause_generation=RemoteCall("pause"),
        flush_cache=RemoteCall("flush"),
        continue_generation=RemoteCall("continue"),
    )
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = _args(
        pause_generation_mode="abort",
        check_weight_update_equal=False,
    )
    updater.weight_version = 0
    updater.is_lora = False
    updater.use_distribute = True
    updater.rollout_engines = []
    updater.distributed_rollout_engines = [engine]
    updater.all_rollout_engines = (engine,)
    updater.weights_getter = lambda: {}
    updater._hf_weight_iterator = Namespace(
        get_hf_weight_chunks=lambda *_args, **_kwargs: ()
    )

    monkeypatch.setattr(tensor_update_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tensor_update_module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(tensor_update_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(tensor_update_module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        tensor_update_module,
        "begin_weight_update",
        lambda engines: observed.append(("begin", tuple(engines))),
    )
    monkeypatch.setattr(
        tensor_update_module,
        "end_weight_update",
        lambda engines: observed.append(("end", tuple(engines))),
    )

    updater.update_weights()

    assert observed == [
        ("pause", (), {"mode": "abort"}),
        ("flush", (), {}),
        ("begin", (engine,)),
        ("end", (engine,)),
        ("continue", (), {}),
    ]


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
