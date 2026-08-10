from types import SimpleNamespace

import pytest

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
        assert actor.apply_trainable_state.calls == [
            ((shared,), {"reset_optimizer": True})
        ]


async def test_public_trainable_state_api_rejects_rank_disagreement(monkeypatch):
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Actor({"state": 1}, 1), _Actor(None, 2)]
    monkeypatch.setattr("miles.ray.actor_group.ray.put", lambda value: value)

    with pytest.raises(RuntimeError, match="ranks disagree"):
        await group.apply_trainable_state(SimpleNamespace(), reset_optimizer=True)
