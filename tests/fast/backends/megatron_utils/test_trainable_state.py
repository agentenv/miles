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


def test_model_parallel_export_is_collective_and_only_rank_zero_returns(monkeypatch):
    actor, side, *_ = _actor()
    actor.args.tensor_model_parallel_size = 2
    actor.args.pipeline_model_parallel_size = 2
    monkeypatch.setattr(
        trainable_state,
        "_collective_adapter_tensors",
        lambda _actor: {NAME: torch.tensor([[1.0, 2.0]])},
    )
    monkeypatch.setattr(trainable_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 1)

    assert trainable_state.export_trainable_state(actor, policy_version=4) is None

    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 0)
    state = trainable_state.export_trainable_state(actor, policy_version=4)
    assert state is not None
    assert state.policy_version == 4
    assert torch.equal(state.tensors[NAME], torch.tensor([[1.0, 2.0]]))


def test_collective_conversion_reads_f32_master_and_restores_model_parameter(monkeypatch):
    actor, side, *_ = _actor()
    original = side.param_weight.data
    side.param_weight.main_param.copy_(torch.tensor([[3.0, 4.0]]))
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])

    with trainable_state._optimizer_masters_as_model_parameters(actor):
        assert side.param_weight.dtype == torch.float32
        assert side.param_weight.data.data_ptr() == side.param_weight.main_param.data_ptr()

    assert side.param_weight.dtype == torch.bfloat16
    assert side.param_weight.data.data_ptr() == original.data_ptr()


def test_external_train_metric_capture_is_rank_zero_only(monkeypatch):
    args = SimpleNamespace(external_policy_sync_path="project.sync.create")
    metrics = {
        "train/train_rollout_kl": 0.1,
        "train/ess_ratio": 0.8,
        "train/pg_clipfrac": 0.25,
    }
    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 0)

    trainable_state._capture_external_train_metrics(args, metrics)

    assert args._external_train_metrics == metrics
    del args._external_train_metrics
    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 1)
    trainable_state._capture_external_train_metrics(args, metrics)
    assert not hasattr(args, "_external_train_metrics")


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
