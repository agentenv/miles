from contextlib import nullcontext
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


def test_collective_export_nonzero_rank_does_not_retain_canonical_tensors(monkeypatch):
    actor, side, *_ = _actor()
    actor.args.hf_checkpoint = "/synthetic/model"
    actor.model = []
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: [(NAME, side)])

    class _UnmaterializedWeight:
        def detach(self):
            raise AssertionError("non-retaining rank materialized an adapter tensor")

    class _Bridge:
        def export_adapter_weights(self, *_args, **_kwargs):
            yield NAME, _UnmaterializedWeight(), "synthetic"

    from megatron.bridge import AutoBridge
    from miles.utils import megatron_bridge_utils

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )
    monkeypatch.setattr(
        megatron_bridge_utils,
        "patch_megatron_model",
        lambda _model: nullcontext(),
    )
    monkeypatch.setattr(trainable_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainable_state.dist, "get_rank", lambda: 1)

    assert trainable_state._collective_adapter_tensors(actor) == {}


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


def _tiny_clone_policy():
    tensors = {NAME: torch.tensor([[9.0, 10.0]])}
    for expert in range(2, 4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for side in ("A", "B"):
                if projection in {"gate_proj", "up_proj"}:
                    shape = (2, 3) if side == "A" else (4, 2)
                else:
                    shape = (2, 4) if side == "A" else (3, 2)
                tensors[
                    "base_model.model.model.layers.0.mlp.experts."
                    f"{expert}.{projection}.lora_{side}.weight"
                ] = torch.full(shape, float(expert + 1))
    return tensors


def _packed_side(name, shape):
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    parameter.main_param = torch.zeros(shape, dtype=torch.float32)
    return SimpleNamespace(
        mapping=_Mapping(),
        megatron_module=None,
        param_weight=parameter,
    )


def test_clone_only_apply_packs_sparse_clone_experts_and_resets_optimizer(
    monkeypatch,
):
    monkeypatch.setattr(trainable_state, "_CLONE_LAYERS", 1)
    monkeypatch.setattr(trainable_state, "_CLONE_ORIGINAL_EXPERTS", 2)
    monkeypatch.setattr(trainable_state, "_CLONE_TOTAL_EXPERTS", 4)
    tensors = _tiny_clone_policy()
    state = trainable_state.make_trainable_state(1, tensors)

    attention = _packed_side(NAME, (1, 2))
    prefix = "base_model.model.model.layers.0.mlp.experts.2"
    expert_sides = {
        f"{prefix}.gate_proj.lora_A.weight": _packed_side("fc1_a", (2, 2, 3)),
        f"{prefix}.gate_proj.lora_B.weight": _packed_side("fc1_b", (2, 8, 2)),
        f"{prefix}.down_proj.lora_A.weight": _packed_side("fc2_a", (2, 2, 4)),
        f"{prefix}.down_proj.lora_B.weight": _packed_side("fc2_b", (2, 3, 2)),
    }
    sides = [(NAME, attention), *expert_sides.items()]
    parameters = [side.param_weight for _name, side in sides]
    inner = SimpleNamespace(
        state={parameter.main_param: {"step": torch.tensor(1.0)} for parameter in parameters}
    )

    def copy_main_to_model():
        for parameter in parameters:
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
        args=SimpleNamespace(
            global_batch_size=1,
            num_steps_per_rollout=1,
            yeto_rl_clone_only_lora=True,
            yeto_rl_canonical_lora_names=tuple(sorted(tensors)),
            yeto_rl_expert_parallel_rank=1,
            yeto_rl_expert_parallel_size=2,
        ),
        optimizer=optimizer,
        opt_param_scheduler=Scheduler(),
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    monkeypatch.setattr(trainable_state, "_adapter_sides", lambda _actor: tuple(sides))
    monkeypatch.setattr(
        trainable_state,
        "export_trainable_state",
        lambda _actor, policy_version: state,
    )

    reset_count = trainable_state.apply_trainable_state(
        actor,
        state,
        reset_optimizer=True,
    )

    assert reset_count == len(tensors)
    assert torch.equal(attention.param_weight.main_param, tensors[NAME])
    for side in expert_sides.values():
        assert torch.all(side.param_weight.main_param[0] == 3)
        assert torch.all(side.param_weight.main_param[1] == 4)
    assert not inner.state
    assert backups == ["actor"]


def test_clone_only_policy_rejects_originals_and_split_fc1_a(monkeypatch):
    monkeypatch.setattr(trainable_state, "_CLONE_ORIGINAL_EXPERTS", 2)
    monkeypatch.setattr(trainable_state, "_CLONE_TOTAL_EXPERTS", 4)
    tensors = _tiny_clone_policy()
    original = (
        "base_model.model.model.layers.0.mlp.experts.0."
        "gate_proj.lora_A.weight"
    )
    tensors[original] = torch.zeros(2, 3)
    with pytest.raises(RuntimeError, match="original expert adapter is present"):
        trainable_state._validate_clone_canonical_tensors(tensors)

    tensors = _tiny_clone_policy()
    tensors[
        "base_model.model.model.layers.0.mlp.experts.2.up_proj.lora_A.weight"
    ].fill_(7)
    name = (
        "base_model.model.model.layers.0.mlp.experts.2."
        "gate_proj.lora_A.weight"
    )
    side = _packed_side(name, (2, 2, 3))
    with pytest.raises(RuntimeError, match="gate/up LoRA A tensors differ"):
        trainable_state._pack_expert_side(
            name,
            side,
            tensors,
            expert_parallel_rank=1,
            expert_parallel_size=2,
        )


def test_clone_only_collective_export_accepts_expanded_logical_experts(
    monkeypatch,
):
    """Packed Bridge originals are checked, then omitted from sparse policy."""

    monkeypatch.setattr(trainable_state, "_CLONE_LAYERS", 1)
    monkeypatch.setattr(trainable_state, "_CLONE_ORIGINAL_EXPERTS", 2)
    monkeypatch.setattr(trainable_state, "_CLONE_TOTAL_EXPERTS", 4)
    tensors = _tiny_clone_policy()
    exported_by_bridge = dict(tensors)
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for side in ("A", "B"):
                clone_name = (
                    "base_model.model.model.layers.0.mlp.experts.2."
                    f"{projection}.lora_{side}.weight"
                )
                original_name = clone_name.replace(".experts.2.", f".experts.{expert}.")
                exported_by_bridge[original_name] = torch.zeros_like(tensors[clone_name])
    attention = _packed_side(NAME, (1, 2))
    representative_name = (
        "base_model.model.model.layers.0.mlp.experts.0."
        "gate_proj.lora_A.weight"
    )
    representative = _packed_side(representative_name, (2, 2, 3))
    actor = SimpleNamespace(
        args=SimpleNamespace(
            hf_checkpoint="/synthetic/e288",
            yeto_rl_clone_only_lora=True,
            yeto_rl_canonical_lora_names=tuple(sorted(tensors)),
            yeto_rl_expert_parallel_rank=0,
            yeto_rl_expert_parallel_size=2,
        ),
        model=[],
    )
    monkeypatch.setattr(
        trainable_state,
        "_adapter_sides",
        lambda _actor: (
            (NAME, attention),
            (representative_name, representative),
        ),
    )

    bridge_calls = []

    class _Bridge:
        def export_adapter_weights(self, *_args, **kwargs):
            bridge_calls.append(kwargs)
            for name, value in sorted(exported_by_bridge.items()):
                yield name, value, "synthetic"

    from megatron.bridge import AutoBridge
    from miles.utils import megatron_bridge_utils

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )
    monkeypatch.setattr(
        megatron_bridge_utils,
        "patch_megatron_model",
        lambda _model: nullcontext(),
    )

    exported = trainable_state._collective_adapter_tensors(actor)
    assert set(exported) == set(tensors)
    assert torch.equal(exported[NAME], tensors[NAME])
    assert bridge_calls == [{"cpu": False, "show_progress": False}]
    trainable_state._validate_clone_canonical_tensors(exported)

    representative.param_weight.main_param[0].fill_(1)
    with pytest.raises(RuntimeError, match="original packed expert LoRA master"):
        trainable_state._collective_adapter_tensors(actor)


def test_clone_only_pack_reconstructs_original_slots_as_exact_zeros(monkeypatch):
    monkeypatch.setattr(trainable_state, "_CLONE_ORIGINAL_EXPERTS", 2)
    monkeypatch.setattr(trainable_state, "_CLONE_TOTAL_EXPERTS", 4)
    tensors = _tiny_clone_policy()
    name = (
        "base_model.model.model.layers.0.mlp.experts.0."
        "gate_proj.lora_B.weight"
    )
    side = _packed_side(name, (2, 8, 2))

    packed = trainable_state._pack_expert_side(
        name,
        side,
        tensors,
        expert_parallel_rank=0,
        expert_parallel_size=2,
    )

    assert packed.shape == side.param_weight.shape
    assert torch.count_nonzero(packed).item() == 0
