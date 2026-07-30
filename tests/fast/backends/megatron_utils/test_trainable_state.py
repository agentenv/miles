from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import trainable_state

NAME = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"


class _Mapping:
    def megatron_to_hf(self, value, _module):
        return {NAME: value}

    def hf_to_megatron(self, value, _module):
        return value


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
