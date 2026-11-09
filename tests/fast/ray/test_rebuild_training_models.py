from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray import trainer_rebuild
from miles.ray.placement_group import PlacementGroupInfo
from miles.ray.trainer_rebuild import TrainerRebuildError, rebuild_training_models

_OLD_VIEW = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[7, 8], pg_reordered_gpu_ids=[7, 8])
_NEW_VIEW = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[0, 1, 3], pg_reordered_gpu_ids=[0, 1, 3])


class _Manager:
    def __init__(self, events: list, *, fail_at: str | None = None) -> None:
        self.events, self.fail_at = events, fail_at

    def _record(self, *event):
        self.events.append(event)
        if self.fail_at == event[0]:
            raise RuntimeError(f"injected failure at {event[0]}")

    async def stop_pools(self, pool_ids):
        self._record("stop_pools", tuple(pool_ids))

    async def get_pg_view(self, name):
        return _OLD_VIEW

    async def set_pg_view(self, name, info, *, replacing_pools=()):
        self._record("set_pg_view", name, tuple(info.pg_reordered_bundle_indices), tuple(replacing_pools))

    async def replace_pools(self, specs, *, scaling=None):
        self._record("replace_pools", tuple(spec.name for spec in specs), scaling)

    async def start_pools(self, pool_ids):
        self._record("start_pools", tuple(pool_ids))


def _args(**overrides):
    return SimpleNamespace(use_fault_tolerance=False, indep_dp=False, trainer_controller_addrs=None, **overrides)


@pytest.fixture
def events(monkeypatch) -> list:
    events: list = []
    pools = ["trainer-controller-actor", "trainer-engine-actor"]
    monkeypatch.setattr(trainer_rebuild, "compute_trainer_configs", lambda args: [SimpleNamespace(trainer_id="actor")])
    monkeypatch.setattr(trainer_rebuild, "compute_trainer_controller_pool_id", lambda tid: f"trainer-controller-{tid}")
    monkeypatch.setattr(trainer_rebuild, "compute_trainer_pool_id", lambda tid: f"trainer-engine-{tid}")
    monkeypatch.setattr(
        trainer_rebuild,
        "compute_specs",
        lambda args: [SimpleNamespace(name=n) for n in [*pools, "inference-engine-0"]],
    )
    monkeypatch.setattr(trainer_rebuild.ScalingConfig, "slice_from", classmethod(lambda cls, args: "scaling"))
    monkeypatch.setattr(trainer_rebuild, "compute_backend_capability", lambda args: "capability")
    monkeypatch.setattr(trainer_rebuild, "create_trainer_handles", lambda args, **kw: {"actor": "new-handle"})

    async def _fake_create(args, rollout_executor, *, handles):
        events.append(("create_training_models", tuple(handles)))
        return "new-actor", None

    monkeypatch.setattr(trainer_rebuild, "create_training_models", _fake_create)
    return events


class TestRebuildTrainingModels:
    async def test_the_old_trainer_is_disposed_and_rebuilt_on_the_new_view(self, events):
        old = MagicMock(dispose=AsyncMock(side_effect=lambda: events.append(("dispose",))))
        pools = ("trainer-controller-actor", "trainer-engine-actor")

        result = await rebuild_training_models(
            _args(), "executor", old_handles={"actor": old}, worker_manager=_Manager(events), trainer_pg_view=_NEW_VIEW
        )

        assert result == ("new-actor", None)
        assert events == [
            ("dispose",),
            ("stop_pools", pools),
            ("set_pg_view", "actor", (0, 1, 3), pools),
            ("replace_pools", pools, "scaling"),
            ("start_pools", pools),
            ("create_training_models", ("actor",)),
        ]

    @pytest.mark.parametrize("stage", ["replace_pools", "start_pools"])
    async def test_a_failure_stops_the_pools_and_restores_the_view(self, events, stage):
        with pytest.raises(TrainerRebuildError) as info:
            await rebuild_training_models(
                _args(),
                "executor",
                old_handles={},
                worker_manager=_Manager(events, fail_at=stage),
                trainer_pg_view=_NEW_VIEW,
            )
        assert info.value.stage == stage and info.value.cleanup_error is None
        assert info.value.previous_view == _OLD_VIEW and info.value.view_restored
        assert events[-2][0] == "stop_pools" and events[-1][:3] == ("set_pg_view", "actor", (7, 8))

    async def test_a_cancel_is_cleaned_up_and_re_raised(self, events, monkeypatch):
        async def _cancelled(args, rollout_executor, *, handles):
            raise asyncio.CancelledError

        monkeypatch.setattr(trainer_rebuild, "create_training_models", _cancelled)
        with pytest.raises(asyncio.CancelledError):
            await rebuild_training_models(_args(), "executor", old_handles={}, worker_manager=_Manager(events))
        assert events[-1][0] == "stop_pools"

    @pytest.mark.parametrize(
        "overrides",
        [dict(use_fault_tolerance=True), dict(indep_dp=True), dict(trainer_controller_addrs=["actor=h:1"])],
    )
    async def test_paths_it_does_not_rebuild_are_refused(self, events, overrides):
        args = SimpleNamespace(**{**vars(_args()), **overrides})
        with pytest.raises(AssertionError):
            await rebuild_training_models(args, "executor", old_handles={}, worker_manager=_Manager(events))
        assert events == []
