from __future__ import annotations

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import BundleInUseError

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

        with pytest.raises(BundleInUseError, match="in use by running cell engine"):
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
        assert manager.get_cell_infos(pool_ids=["trainer"])[cell_id].workers_hash == "pseudo-hash-3"

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


class TestPgViewOccupancy:
    async def test_a_view_over_bundles_of_a_running_cell_is_refused(self, fake_ray_cluster: FakeRayCluster):
        """Pointing the trainer at a running engine's bundle would double-book that GPU on the next start."""
        spec = _make_spec(
            "trainer", num_workers_per_cell=2, num_gpus_per_worker=0.4, num_gpu_slots_per_worker=1, pg_name="actor"
        )
        manager = await _launch([spec, _engine_spec()], _pgs())
        await manager.stop_pools(["trainer"])

        with pytest.raises(BundleInUseError, match="in use by running cell engine"):
            await manager.set_pg_view("actor", _view([0, 1, 2]))

        await manager.set_pg_view("actor", _view([0, 1, 4]))
        assert manager.pgs["actor"].pg_reordered_gpu_ids == [10, 11, 14]

    async def test_a_replaced_pool_starts_at_a_new_generation(self, fake_ray_cluster: FakeRayCluster):
        spec = _make_spec(
            "trainer", num_workers_per_cell=1, num_gpus_per_worker=0.4, num_gpu_slots_per_worker=1, pg_name="actor"
        )
        manager = await _launch([spec], _pgs())
        cell_id = compute_cell_id(pool_id="trainer", cell_index=0)
        await manager.stop_pools(["trainer"])

        await manager.replace_pool_spec(spec)

        assert manager.get_cell_infos(pool_ids=["trainer"])[cell_id].workers_hash == "pseudo-hash-2"


class TestPgViewStoppedCellsFit:
    def _trainer_spec(self, num_workers: int):
        return _make_spec(
            "trainer",
            num_cells=1,
            num_workers_per_cell=num_workers,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            pg_name="actor",
        )

    async def test_a_view_too_small_for_a_stopped_cell_is_refused(self, fake_ray_cluster: FakeRayCluster):
        """Shrinking the view under a stopped 2-worker trainer would make its next start reach outside the view."""
        manager = await _launch([self._trainer_spec(2)], _pgs())
        await manager.stop_pools(["trainer"])

        with pytest.raises(AssertionError, match=r"beyond the 1 bundles of the new 'actor' view"):
            await manager.set_pg_view("actor", _view([0]))
        assert manager.pgs["actor"].pg_reordered_gpu_ids == [10, 11]

    async def test_a_pool_being_replaced_is_exempt(self, fake_ray_cluster: FakeRayCluster):
        """P2 -> P1: the rebuild re-points the view first, then replaces the trainer spec with one worker."""
        manager = await _launch([self._trainer_spec(2)], _pgs())
        await manager.stop_pools(["trainer"])

        await manager.set_pg_view("actor", _view([0]), replacing_pools=["trainer"])
        [cell_id] = await manager.replace_pool_spec(self._trainer_spec(1))
        await manager.start_pools(["trainer"])

        assert manager.get_cell_bundles(cell_id) == [_BUNDLES[0]]


class TestStartRollback:
    async def test_a_failed_rollback_is_reported_as_such(self, fake_ray_cluster: FakeRayCluster, monkeypatch):
        from miles.utils.workers import ray_worker_manager as rwm

        manager = await _launch([_engine_spec()], _pgs())
        cell_id = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([cell_id])

        async def _fail(self):
            raise RuntimeError("post_setup failed")

        original_stop = rwm._CellManager.stop

        async def _stop_fails_once(self):
            await original_stop(self)
            raise RuntimeError("stop failed")

        monkeypatch.setattr(rwm._CellManager, "post_setup", _fail)
        monkeypatch.setattr(rwm._CellManager, "stop", _stop_fails_once)
        with pytest.raises(rwm.StartRollbackFailedError, match="during the rollback failed too") as info:
            await manager.start_cells([cell_id])
        assert "post_setup failed" in repr(info.value.__cause__)

    async def test_a_clean_rollback_reraises_the_start_error(self, fake_ray_cluster: FakeRayCluster, monkeypatch):
        from miles.utils.workers import ray_worker_manager as rwm

        manager = await _launch([_engine_spec()], _pgs())
        cell_id = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([cell_id])

        async def _fail(self):
            raise RuntimeError("post_setup failed")

        monkeypatch.setattr(rwm._CellManager, "post_setup", _fail)
        with pytest.raises(RuntimeError, match="post_setup failed"):
            await manager.start_cells([cell_id])
        assert not manager._find_cell(cell_id).alive
