from __future__ import annotations

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import PlacementGroupInfo, TrainerRebuildError, rebuild_training_models


class _Manager:
    def __init__(self, events: list) -> None:
        self.events = events

    async def stop_pools(self, pool_ids):
        self.events.append(("stop_pools", tuple(pool_ids)))

    async def set_pg_view(self, name, info, *, replacing_pools=()):
        self.events.append(("set_pg_view", name, tuple(info.pg_reordered_bundle_indices)))
        self.replacing_pools = tuple(replacing_pools)

    async def replace_pool_spec(self, spec):
        self.events.append(("replace_pool_spec", spec.name, spec.scheduling.num_workers_per_cell))
        return [f"{spec.name}-0"]

    async def start_pools(self, pool_ids):
        self.events.append(("start_pools", tuple(pool_ids)))

    async def get_pg_view(self, name):
        return PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[7, 8], pg_reordered_gpu_ids=[7, 8])


def _args(**overrides) -> Namespace:
    defaults = dict(actor_num_nodes=1, actor_num_gpus_per_node=3, use_critic=False, megatron_config=None)
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


@pytest.fixture
def events(monkeypatch) -> list:
    events: list = []

    async def _fake_create(args, rollout_executor):
        events.append(("create_training_models", args.actor_num_gpus_per_node))
        return "new-actor", None

    monkeypatch.setattr(placement_group_module, "create_training_models", _fake_create)
    return events


class TestRebuildTrainingModels:
    async def test_the_old_trainer_is_disposed_and_rebuilt_on_the_new_view(self, events):
        old = MagicMock(dispose=AsyncMock(side_effect=lambda: events.append(("dispose",))))
        view = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[0, 1, 3], pg_reordered_gpu_ids=[0, 1, 3])

        result = await rebuild_training_models(
            _args(), "executor", old_handles={"actor": old}, worker_manager=_Manager(events), trainer_pg_view=view
        )

        assert result == ("new-actor", None)
        names = [e for e in events if e[0] == "replace_pool_spec"]
        assert [e[0] for e in events] == [
            "dispose",
            "stop_pools",
            "set_pg_view",
            *["replace_pool_spec"] * len(names),
            "start_pools",
            "create_training_models",
        ]
        assert ("set_pg_view", "actor", (0, 1, 3)) in events
        trainer_engine = [e for e in names if e[1].startswith("trainer-engine")]
        assert [e[2] for e in trainer_engine] == [3]
        assert events[-1] == ("create_training_models", 3)

    async def test_a_failed_dispose_does_not_block_the_rebuild(self, events):
        old = MagicMock(dispose=AsyncMock(side_effect=RuntimeError("controller gone")))

        await rebuild_training_models(_args(), "executor", old_handles={"actor": old}, worker_manager=_Manager(events))

        assert not any(e[0] == "set_pg_view" for e in events)
        assert events[-1][0] == "create_training_models"

    async def test_ray_actor_handles_are_called_through_remote(self, events):
        class _Remote:
            def __init__(self, fn):
                self.remote = fn

        inner = _Manager(events)
        handle = MagicMock()
        for name in ("stop_pools", "set_pg_view", "replace_pool_spec", "start_pools"):
            setattr(handle, name, _Remote(getattr(inner, name)))

        await rebuild_training_models(_args(), "executor", old_handles={}, worker_manager=handle)

        assert [e[0] for e in events][0] == "stop_pools"

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(use_fault_tolerance=True), "not a fault-tolerance path"),
            (dict(indep_dp=True), "indep-DP"),
            (dict(trainer_controller_addrs="x"), "independently deployed"),
        ],
    )
    async def test_fault_tolerance_and_split_deployments_are_refused(self, events, overrides, match):
        with pytest.raises(AssertionError, match=match):
            await rebuild_training_models(
                _args(**overrides), "executor", old_handles={}, worker_manager=_Manager(events)
            )
        assert events == []


class TestRebuildFailure:
    _VIEW = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[0, 1], pg_reordered_gpu_ids=[0, 1])

    async def test_the_view_exempts_exactly_the_pools_being_replaced(self, events):
        manager = _Manager(events)
        await rebuild_training_models(
            _args(), "executor", old_handles={}, worker_manager=manager, trainer_pg_view=self._VIEW
        )
        replaced = tuple(e[1] for e in events if e[0] == "replace_pool_spec")
        assert manager.replacing_pools == replaced

    @pytest.mark.parametrize("stage", ["replace_pool_spec", "start_pools", "create_training_models"])
    async def test_a_failed_stage_stops_the_trainer_pools_and_names_the_stage(self, events, monkeypatch, stage):
        manager = _Manager(events)
        if stage == "create_training_models":

            async def _fail_create(args, rollout_executor):
                raise RuntimeError("boom")

            monkeypatch.setattr(placement_group_module, "create_training_models", _fail_create)
        else:

            async def _fail(*_args, **_kwargs):
                raise RuntimeError("boom")

            setattr(manager, stage, _fail)

        with pytest.raises(TrainerRebuildError, match=f"failed at {stage}") as info:
            await rebuild_training_models(
                _args(), "executor", old_handles={}, worker_manager=manager, trainer_pg_view=self._VIEW
            )

        assert info.value.stage == stage and info.value.cleanup_error is None
        assert "boom" in repr(info.value.__cause__)
        # cleanup ran after the failure: pools stopped, then the previous actor view restored
        assert [e[0] for e in events][-2:] == ["stop_pools", "set_pg_view"]

    async def test_a_failed_cleanup_is_reported(self, events):
        manager = _Manager(events)
        calls = 0

        async def _stop(pool_ids):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("stop failed")

        async def _fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        manager.stop_pools, manager.start_pools = _stop, _fail
        with pytest.raises(TrainerRebuildError, match="stopping trainer pools failed") as info:
            await rebuild_training_models(_args(), "executor", old_handles={}, worker_manager=manager)
        assert info.value.stage == "start_pools"


class TestRebuildFailureRestoresTheView:
    _VIEW = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[0, 1], pg_reordered_gpu_ids=[0, 1])

    async def test_the_previous_actor_view_is_put_back_and_reported(self, events):
        manager = _Manager(events)

        async def _fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        manager.start_pools = _fail
        with pytest.raises(TrainerRebuildError) as info:
            await rebuild_training_models(
                _args(), "executor", old_handles={}, worker_manager=manager, trainer_pg_view=self._VIEW
            )
        assert info.value.view_restored and info.value.previous_view.pg_reordered_bundle_indices == [7, 8]
        assert events[-1] == ("set_pg_view", "actor", (7, 8))

    async def test_a_cancellation_is_cleaned_up_and_reraised_unchanged(self, events):
        import asyncio

        manager = _Manager(events)

        async def _cancel(*_args, **_kwargs):
            raise asyncio.CancelledError()

        manager.replace_pool_spec = _cancel
        with pytest.raises(asyncio.CancelledError):
            await rebuild_training_models(
                _args(), "executor", old_handles={}, worker_manager=manager, trainer_pg_view=self._VIEW
            )
        assert [e[0] for e in events][-2:] == ["stop_pools", "set_pg_view"]
