from __future__ import annotations

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers.naming import compute_cell_id

# One startup PG of 6 bundles split like an M1 map: trainer [0, 1], rollout [2, 3], standby [4, 5].
_BUNDLES = [5, 3, 1, 0, 2, 4]  # reordered bundle index per logical position
_GPUS = [10, 11, 12, 13, 14, 15]


def _view(positions: list[int]) -> PlacementGroupInfo:
    return PlacementGroupInfo(
        pg="fake-pg",
        pg_reordered_bundle_indices=[_BUNDLES[p] for p in positions],
        pg_reordered_gpu_ids=[_GPUS[p] for p in positions],
    )


def _pgs() -> dict[str, PlacementGroupInfo]:
    return {"actor": _view([0, 1]), "rollout": _view([2, 3]), "standby": _view([4, 5])}


def _engine_spec(**overrides):
    return _make_spec(
        "engine",
        num_cells=2,
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=1,
        pg_name="rollout",
        **overrides,
    )


def _bundle_of_last_create(cluster: FakeRayCluster) -> int:
    return cluster.handles[-1].options["scheduling_strategy"].placement_group_bundle_index


class TestRebindCell:
    async def test_a_stopped_cell_restarts_on_the_standby_bundle(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec()], _pgs())
        cell_id = compute_cell_id(pool_id="engine", cell_index=1)
        assert manager.get_cell_bundles(cell_id) == [_BUNDLES[3]]

        await manager.stop_cells([cell_id])
        await manager.rebind_cell(cell_id, pg_name="standby", pg_slot_offset=1)
        await manager.start_cells([cell_id])

        assert _bundle_of_last_create(fake_ray_cluster) == _BUNDLES[5]
        assert manager.get_cell_bundles(cell_id) == [_BUNDLES[5]]
        assert manager.get_worker_infos(cell_id)[0].gpu_ids == [_GPUS[5]]

    async def test_a_running_cell_cannot_be_rebound(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec()], _pgs())
        with pytest.raises(AssertionError, match="stop it before rebinding"):
            await manager.rebind_cell(
                compute_cell_id(pool_id="engine", cell_index=0), pg_name="standby", pg_slot_offset=0
            )

    async def test_a_bundle_used_by_a_running_cell_is_refused(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec()], _pgs())
        cell_1 = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([cell_1])

        with pytest.raises(AssertionError, match="in use by running cell engine"):
            await manager.rebind_cell(cell_1, pg_name="rollout", pg_slot_offset=0)

    async def test_targets_outside_the_startup_views_are_refused(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec()], _pgs())
        cell_1 = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([cell_1])

        with pytest.raises(AssertionError, match="outside 'standby'"):
            await manager.rebind_cell(cell_1, pg_name="standby", pg_slot_offset=2)
        with pytest.raises(AssertionError, match="unknown placement group view"):
            await manager.rebind_cell(cell_1, pg_name="nowhere", pg_slot_offset=0)

    async def test_a_gpu_less_cell_cannot_be_bound(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_make_spec("router")], _pgs())
        cell = compute_cell_id(pool_id="router", cell_index=0)
        await manager.stop_cells([cell])
        with pytest.raises(AssertionError, match="owns no GPU slots"):
            await manager.rebind_cell(cell, pg_name="standby", pg_slot_offset=0)


class TestReplacePoolAndPgView:
    def _trainer_spec(self, num_workers: int):
        return _make_spec(
            "trainer",
            num_cells=1,
            num_workers_per_cell=num_workers,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            pg_name="actor",
        )

    async def test_the_trainer_grows_onto_freed_rollout_bundles(self, fake_ray_cluster: FakeRayCluster):
        """P2 -> P3: stop the trainer and one engine, widen the actor view, restart with three workers."""
        manager = await _launch([self._trainer_spec(2), _engine_spec()], _pgs())
        engine_1 = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([engine_1])
        await manager.stop_pools(["trainer"])

        await manager.set_pg_view("actor", _view([0, 1, 3]))
        [cell_id] = await manager.replace_pool_spec(self._trainer_spec(3))
        await manager.start_pools(["trainer"])

        assert manager.get_cell_bundles(cell_id) == sorted(_BUNDLES[p] for p in (0, 1, 3))
        assert [a.gpu_ids for a in manager._find_cell(cell_id).actors] == [[10], [11], [13]]
        assert manager.get_cell_infos(pool_ids=["trainer"])[cell_id].workers_hash == "pseudo-hash-2"

    async def test_a_running_pool_cannot_be_replaced_or_repointed(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([self._trainer_spec(2)], _pgs())
        with pytest.raises(AssertionError, match="stop the pool first"):
            await manager.replace_pool_spec(self._trainer_spec(1))
        with pytest.raises(AssertionError, match="stop them before re-pointing"):
            await manager.set_pg_view("actor", _view([0]))

    async def test_a_view_outside_the_startup_pg_is_refused(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([self._trainer_spec(2)], _pgs())
        await manager.stop_pools(["trainer"])
        with pytest.raises(AssertionError, match="outside the placement group created at startup"):
            await manager.set_pg_view(
                "actor", PlacementGroupInfo(pg="fake-pg", pg_reordered_bundle_indices=[9], pg_reordered_gpu_ids=[0])
            )

    async def test_a_spec_needing_more_slots_than_the_view_is_refused(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([self._trainer_spec(2)], _pgs())
        await manager.stop_pools(["trainer"])
        with pytest.raises(AssertionError, match="outside 'actor'"):
            await manager.replace_pool_spec(self._trainer_spec(3))
