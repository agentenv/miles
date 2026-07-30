from types import SimpleNamespace

import pytest

from miles.ray.actor_group import RayTrainGroup

pytestmark = pytest.mark.asyncio


class _Remote:
    def __init__(self, result):
        self.result = result

    async def remote(self, *_args, **_kwargs):
        return self.result


class _Actor:
    def __init__(self, exported, applied):
        self.export_trainable_state = _Remote(exported)
        self.apply_trainable_state = _Remote(applied)


async def test_public_trainable_state_api_hides_actor_broadcast():
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Actor({"state": 1}, 2), _Actor(None, 2)]

    assert await group.export_trainable_state() == {"state": 1}
    assert (
        await group.apply_trainable_state(
            SimpleNamespace(),
            reset_optimizer=True,
        )
        == 2
    )


async def test_public_trainable_state_api_rejects_rank_disagreement():
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Actor({"state": 1}, 1), _Actor(None, 2)]

    with pytest.raises(RuntimeError, match="ranks disagree"):
        await group.apply_trainable_state(SimpleNamespace(), reset_optimizer=True)
