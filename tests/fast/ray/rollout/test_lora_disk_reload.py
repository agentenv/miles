from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_manager import RolloutManager


class _Server:
    def __init__(self, name: str, *, update_weights: bool, events: list[str]):
        self.name = name
        self.update_weights = update_weights
        self.events = events

    async def onload(self, tags=None):
        self.events.append(f"{self.name}:onload:{','.join(tags or [])}")

    async def reload_weights_from_disk(self):
        self.events.append(f"{self.name}:reload")


@pytest.mark.asyncio
async def test_lora_disk_reload_happens_after_weight_storage_is_remapped():
    events = []
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(lora_base_disk_reload=True)
    manager.servers = {
        "actor": _Server("actor", update_weights=True, events=events),
        "reference": _Server("reference", update_weights=False, events=events),
    }

    await manager.onload_weights()

    assert events == [
        "actor:onload:weights",
        "reference:onload:weights",
        "actor:reload",
    ]


@pytest.mark.asyncio
async def test_default_weight_onload_does_not_reload_from_disk():
    events = []
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(lora_base_disk_reload=False)
    manager.servers = {
        "actor": _Server("actor", update_weights=True, events=events),
    }

    await manager.onload_weights()

    assert events == ["actor:onload:weights"]
