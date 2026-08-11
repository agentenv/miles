import sys
import types
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import trainable_state

NAME = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
REMOTE_NAME = "base_model.model.layers.1.self_attn.q_proj.lora_A.weight"


class _Mapping:
    def megatron_to_hf(self, value, _module):
        return {NAME: value}

    def hf_to_megatron(self, value, _module):
        return value


class _RemoteMapping:
    def __init__(self, value):
        self.value = value

    def megatron_to_hf(self, value, module):
        assert value is None
        assert module is None
        return {REMOTE_NAME: self.value}

    def hf_to_megatron(self, *_args):
        raise AssertionError("remote pipeline tensor must not be applied locally")


def _remote_side(value):
    return SimpleNamespace(
        mapping=_RemoteMapping(value),
        megatron_module=None,
        param_name="remote.adapter.linear_in.weight",
        param_weight=None,
    )


def _actor():
    parameter = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    parameter.main_param = torch.zeros(1, 2, dtype=torch.float32)
    side = SimpleNamespace(
        mapping=_Mapping(),
        megatron_module=None,
        param_weight=parameter,
    )
    unrelated = torch.nn.Parameter(torch.ones(1))
    inner = SimpleNamespace(
        state={
            parameter.main_param: {"step": torch.tensor(3.0)},
            unrelated: {"step": torch.tensor(4.0)},
        }
    )

    def copy_main_to_model():
        parameter.copy_(parameter.main_param)

    optimizer = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(
                optimizer=inner,
                _copy_main_params_to_model_params=copy_main_to_model,
            )
        ]
    )

    class Scheduler:
        num_steps = 0

        def step(self, increment):
            self.num_steps += increment

    backups = []
    actor = SimpleNamespace(
        args=SimpleNamespace(global_batch_size=4, num_steps_per_rollout=2),
        optimizer=optimizer,
        opt_param_scheduler=Scheduler(),
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    return actor, side, unrelated, inner, backups


def test_export_trainable_state_is_canonical(monkeypatch):
    actor, side, *_ = _actor()
    side.param_weight.main_param.copy_(torch.tensor([[1.25, 2.5]]))
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])

    state = trainable_state.export_trainable_state(actor, policy_version=3)

    assert state.policy_version == 3
    assert list(state.tensors) == [NAME]
    assert state.tensors[NAME].device.type == "cpu"
    assert state.tensors[NAME].dtype == torch.float32
    assert state.tensors[NAME].is_contiguous()
    assert torch.equal(state.tensors[NAME], torch.tensor([[1.25, 2.5]]))


def test_export_trainable_state_carries_external_round_stats(monkeypatch):
    actor, side, *_ = _actor()
    actor.args.external_policy_sync_path = "project.sync.create"
    actor.args._external_train_metrics = {
        "train/train_rollout_kl": 0.1,
        "train/ess_ratio": 0.8,
        "train/pg_clipfrac": 0.25,
    }
    actor.args._external_train_seconds = 1.5
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])

    state = trainable_state.export_trainable_state(actor, policy_version=3)

    assert state.train_rollout_kl == 0.1
    assert state.ess_ratio == 0.8
    assert state.pg_clipfrac == 0.25
    assert state.train_seconds == 1.5


def test_external_train_metric_capture_uses_caller_selected_rank(monkeypatch):
    args = SimpleNamespace(external_policy_sync_path="project.sync.create")
    metrics = {
        "train/train_rollout_kl": 0.1,
        "train/ess_ratio": 0.8,
        "train/pg_clipfrac": 0.25,
    }
    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 1)

    trainable_state._capture_external_train_metrics(args, metrics)

    assert args._external_train_metrics == metrics


def test_adapter_discovery_accepts_remote_pipeline_tasks(monkeypatch):
    local_a = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    local_a.main_param = torch.zeros(1, 2, dtype=torch.float32)
    local_b = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    local_b.main_param = torch.zeros(1, 2, dtype=torch.float32)
    chunk = torch.nn.Module()
    chunk.register_parameter("local_a", local_a)
    chunk.register_parameter("local_b", local_b)

    def side(name, parameter):
        mapping = SimpleNamespace(
            megatron_to_hf=lambda value, _module: {name: value},
        )
        return SimpleNamespace(
            mapping=mapping,
            megatron_module=None,
            param_name=name,
            param_weight=parameter,
        )

    tasks = {
        "local": [
            SimpleNamespace(
                adapter_key=None,
                linear_in_task=side(NAME, local_a),
                linear_out_task=side(
                    "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    local_b,
                ),
            )
        ],
        "remote": [
            SimpleNamespace(
                adapter_key=None,
                linear_in_task=_remote_side(torch.ones(1, 2)),
                linear_out_task=SimpleNamespace(
                    mapping=SimpleNamespace(
                        megatron_to_hf=lambda value, module: {
                            "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.ones(1, 2)
                        }
                    ),
                    megatron_module=None,
                    param_name="remote.adapter.linear_out.weight",
                    param_weight=None,
                ),
            )
        ],
    }
    model_bridge = SimpleNamespace(build_adapter_conversion_tasks=lambda _model: tasks)
    bridge = SimpleNamespace(_model_bridge=model_bridge)
    bridge_module = types.ModuleType("megatron.bridge")
    bridge_module.AutoBridge = SimpleNamespace(
        from_hf_pretrained=lambda *_args, **_kwargs: bridge
    )
    megatron_module = types.ModuleType("megatron")
    megatron_module.bridge = bridge_module
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_module)
    actor = SimpleNamespace(
        args=SimpleNamespace(hf_checkpoint="/model"),
        model=[chunk],
    )

    sides = trainable_state._adapter_sides(actor)

    assert [name for name, _side in sides] == [
        NAME,
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        REMOTE_NAME,
        "base_model.model.layers.1.self_attn.q_proj.lora_B.weight",
    ]


def test_export_trainable_state_collects_remote_pipeline_stage(monkeypatch):
    actor, side, *_ = _actor()
    side.param_weight.main_param.copy_(torch.tensor([[1.0, 2.0]]))
    remote = _remote_side(torch.tensor([[3.0, 4.0]]))
    monkeypatch.setattr(
        trainable_state,
        "_adapter_sides",
        lambda _actor: [(NAME, side), (REMOTE_NAME, remote)],
    )

    state = trainable_state.export_trainable_state(actor, policy_version=1)

    assert set(state.tensors) == {NAME, REMOTE_NAME}
    assert torch.equal(state.tensors[REMOTE_NAME], torch.tensor([[3.0, 4.0]]))


def test_apply_trainable_state_resets_only_lora_and_preserves_scheduler(monkeypatch):
    actor, side, unrelated, inner, backups = _actor()
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])
    state = trainable_state.make_trainable_state(
        3,
        {NAME: torch.tensor([[5.0, 7.0]])},
    )

    reset_count = trainable_state.apply_trainable_state(
        actor,
        state,
        reset_optimizer=True,
    )

    assert reset_count == 1
    assert side.param_weight.main_param not in inner.state
    assert unrelated in inner.state
    assert actor.opt_param_scheduler.num_steps == 24
    assert torch.equal(side.param_weight.main_param, torch.tensor([[5.0, 7.0]]))
    assert torch.equal(side.param_weight.float(), torch.tensor([[5.0, 7.0]]))
    assert backups == ["actor"]


def test_apply_trainable_state_updates_only_local_pipeline_stage(monkeypatch):
    actor, side, unrelated, inner, backups = _actor()
    remote = _remote_side(torch.tensor([[3.0, 4.0]]))
    monkeypatch.setattr(
        trainable_state,
        "_adapter_sides",
        lambda _actor: [(NAME, side), (REMOTE_NAME, remote)],
    )
    state = trainable_state.make_trainable_state(
        1,
        {
            NAME: torch.tensor([[5.0, 7.0]]),
            REMOTE_NAME: torch.tensor([[11.0, 13.0]]),
        },
    )

    reset_count = trainable_state.apply_trainable_state(
        actor,
        state,
        reset_optimizer=True,
    )

    assert reset_count == 2
    assert side.param_weight.main_param not in inner.state
    assert unrelated in inner.state
    assert torch.equal(side.param_weight.main_param, torch.tensor([[5.0, 7.0]]))
    assert backups == ["actor"]


def test_partial_trainable_state_applies_preserve_optimizer_and_advance_scheduler(
    monkeypatch,
):
    actor, side, unrelated, inner, backups = _actor()
    adapter_state = inner.state[side.param_weight.main_param]
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])

    for policy_version, value in ((1, 5.0), (2, 7.0)):
        reset_count = trainable_state.apply_trainable_state(
            actor,
            trainable_state.make_trainable_state(
                policy_version,
                {NAME: torch.full((1, 2), value)},
            ),
            reset_optimizer=False,
        )
        assert reset_count == 0
        assert inner.state[side.param_weight.main_param] is adapter_state
        assert unrelated in inner.state

    assert actor.opt_param_scheduler.num_steps == 16
    assert torch.equal(side.param_weight.main_param, torch.tensor([[7.0, 7.0]]))
    assert backups == ["actor", "actor"]


def test_apply_trainable_state_rejects_layout_change(monkeypatch):
    actor, side, *_ = _actor()
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])
    state = trainable_state.make_trainable_state(
        0,
        {"base_model.model.other.lora_A.weight": torch.ones(1, 2)},
    )

    with pytest.raises(RuntimeError, match="mapping mismatch"):
        trainable_state.apply_trainable_state(actor, state, reset_optimizer=True)


def test_apply_trainable_state_rejects_scheduler_rewind(monkeypatch):
    actor, side, *_ = _actor()
    actor.opt_param_scheduler.num_steps = 12
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])
    state = trainable_state.make_trainable_state(1, {NAME: torch.ones(1, 2)})

    with pytest.raises(RuntimeError, match="ahead of the committed policy"):
        trainable_state.apply_trainable_state(actor, state, reset_optimizer=True)
