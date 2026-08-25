from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterLocalStepReceipt,
    FullParameterOptimizerState,
    FullParameterTopology,
)
from miles.ray.actor_group import RayTrainGroup

pytestmark = pytest.mark.asyncio


class _Remote:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _Actor:
    def __init__(self, exported, applied):
        self.export_trainable_state = _Remote(exported)
        self.apply_trainable_state = _Remote(applied)


async def test_public_trainable_state_api_hides_actor_broadcast(monkeypatch):
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Actor({"state": 1}, 2), _Actor(None, 2)]
    shared = object()
    puts = []
    monkeypatch.setattr(
        "miles.ray.actor_group.ray.put",
        lambda value: puts.append(value) or shared,
    )
    state = SimpleNamespace()

    assert await group.export_trainable_state() == {"state": 1}
    assert (
        await group.apply_trainable_state(
            state,
            reset_optimizer=True,
        )
        == 2
    )
    assert puts == [state]
    for actor in group._actor_handles:
        assert actor.apply_trainable_state.calls == [((shared,), {"reset_optimizer": True})]


async def test_public_trainable_state_api_rejects_rank_disagreement(monkeypatch):
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Actor({"state": 1}, 1), _Actor(None, 2)]
    monkeypatch.setattr("miles.ray.actor_group.ray.put", lambda value: value)

    with pytest.raises(RuntimeError, match="ranks disagree"):
        await group.apply_trainable_state(SimpleNamespace(), reset_optimizer=True)


def _optimizer_state(tp_rank: int, *, scheduler_num_steps: int = 1):
    return FullParameterOptimizerState(
        topology=FullParameterTopology(
            tp_rank=tp_rank,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            ep_rank=0,
            ep_size=1,
            cp_rank=0,
            cp_size=1,
            dp_rank=0,
            dp_size=1,
        ),
        role="actor",
        installed_policy_version=0,
        local_step_generation=1,
        last_rollout_id=0,
        scheduler_num_steps=scheduler_num_steps,
        populated_parameter_count=2,
        optimizer_state_tensor_count=6,
        optimizer_state_scalar_count=18,
        selected_wire_name=f"actor::tp{tp_rank}-of-2::model.norm.weight",
        selected_state_sha256=str(tp_rank + 1) * 64,
        model_master_parameter_count=2,
    )


async def test_full_parameter_optimizer_state_api_orders_complete_rank_proofs():
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [object(), object()]

    async def broadcast(name):
        assert name == "full_parameter_optimizer_state"
        return [_optimizer_state(1), _optimizer_state(0)]

    group._broadcast = broadcast
    states = await group.full_parameter_optimizer_states()

    assert [state.topology.tp_rank for state in states] == [0, 1]


async def test_full_parameter_optimizer_state_api_rejects_progress_disagreement():
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [object(), object()]

    async def broadcast(_name):
        return [_optimizer_state(0), _optimizer_state(1, scheduler_num_steps=2)]

    group._broadcast = broadcast
    with pytest.raises(RuntimeError, match="disagree"):
        await group.full_parameter_optimizer_states()


def _local_receipt(tp_rank: int, *, generation: int = 1):
    return FullParameterLocalStepReceipt(
        topology=_optimizer_state(tp_rank).topology,
        role="actor",
        base_policy_version=0,
        local_step_generation=generation,
        rollout_id=0,
        optimizer_steps=1,
        scheduler_start_steps=0,
        scheduler_end_steps=1,
    )


async def test_full_parameter_local_step_api_requires_rank_agreement():
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [object(), object()]

    async def broadcast(name, base, rollout):
        assert (name, base, rollout) == (
            "record_full_parameter_local_step",
            0,
            0,
        )
        return [_local_receipt(1), _local_receipt(0)]

    group._broadcast = broadcast
    receipts = await group.record_full_parameter_local_step(
        base_policy_version=0,
        rollout_id=0,
    )
    assert [receipt.topology.tp_rank for receipt in receipts] == [0, 1]

    async def disagree(_name, _base, _rollout):
        return [_local_receipt(0), _local_receipt(1, generation=2)]

    group._broadcast = disagree
    with pytest.raises(RuntimeError, match="disagree"):
        await group.record_full_parameter_local_step(
            base_policy_version=0,
            rollout_id=0,
        )
