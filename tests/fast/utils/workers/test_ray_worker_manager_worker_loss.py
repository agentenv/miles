"""A cell whose workers die is described ``workers_lost``; nothing else goes down.

An engine killed under a member publish: the manager keeps running with its other cells, says which cell lost
which workers (so a caller's watchdog can tell "its engine died" from "it was stopped or never started"), accepts
a stop of that cell, and starts it again on request.
"""

from __future__ import annotations

import pytest
import ray
from tests.fast.utils.workers.fake_ray import EVENT_KILL, READINESS_METHOD, FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec

from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def _cell(index: int) -> str:
    return compute_cell_id(pool_id="engine", cell_index=index)


def _worker(cell_index: int, worker_in_cell_index: int = 0) -> str:
    return compute_worker_name(pool_id="engine", cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)


def _describe(manager: RayWorkerManager) -> dict[str, dict]:
    return manager.describe_cells(pool_ids=["engine"])


async def _scan_all_live_cells(manager: RayWorkerManager) -> None:
    for cell in manager._all_cells():
        if cell.alive:
            await cell._scan_liveness_once()


def _kill_worker_process(cluster: FakeRayCluster, *, handle_index: int) -> None:
    cluster.handles[handle_index].failing_methods[READINESS_METHOD] = ray.exceptions.RayActorError()


@pytest.fixture
async def manager_with_a_lost_cell(fake_ray_cluster: FakeRayCluster) -> tuple[RayWorkerManager, FakeRayCluster]:
    manager = await _launch([_make_spec("engine", num_cells=3)])
    _kill_worker_process(fake_ray_cluster, handle_index=1)
    await _scan_all_live_cells(manager)
    return manager, fake_ray_cluster


class TestWorkerLossIsLocalToTheCell:
    async def test_the_manager_and_the_other_cells_survive(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell

        described = _describe(manager)
        assert [described[_cell(i)]["state"] for i in range(3)] == ["running", "workers_lost", "running"]
        assert [h.killed for h in cluster.handles] == [False, True, False]
        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert [infos[_cell(i)].alive for i in range(3)] == [True, False, True]
        assert manager.get_worker_infos(_cell(1)) == []
        assert len(manager.get_worker_infos(_cell(0))) == 1

    async def test_the_lost_workers_are_named(self, manager_with_a_lost_cell):
        manager, _ = manager_with_a_lost_cell

        described = _describe(manager)

        assert described[_cell(1)]["state"] == "workers_lost"
        assert described[_cell(1)]["lost_workers"] == [_worker(1)]
        assert described[_cell(0)]["lost_workers"] is None

    async def test_every_dead_worker_of_a_cell_is_named(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _scan_all_live_cells(manager)

        assert _describe(manager)[_cell(0)]["lost_workers"] == [_worker(0, 0), _worker(0, 1)]

    async def test_a_cell_with_live_workers_is_not_lost(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_make_spec("engine")])
        await _scan_all_live_cells(manager)
        described = _describe(manager)[_cell(0)]
        assert described["state"] == "running" and described["lost_workers"] is None


class TestStopOfALostCellSucceeds:
    async def test_stop_cells_acknowledges_the_loss_and_kills_nothing_more(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell
        kills_before = cluster.events.count(EVENT_KILL)

        await manager.stop_cells([_cell(1)])

        described = _describe(manager)[_cell(1)]
        assert described["state"] == "stopped" and described["lost_workers"] is None
        assert cluster.events.count(EVENT_KILL) == kills_before

    async def test_stop_cells_twice_is_fine(self, manager_with_a_lost_cell):
        manager, _ = manager_with_a_lost_cell
        await manager.stop_cells([_cell(1)])
        await manager.stop_cells([_cell(1)])
        assert _describe(manager)[_cell(1)]["state"] == "stopped"

    async def test_a_mixed_stop_of_lost_and_running_cells_succeeds(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell

        await manager.stop_cells([_cell(1), _cell(2)])

        described = _describe(manager)
        assert [described[_cell(i)]["state"] for i in range(3)] == ["running", "stopped", "stopped"]
        assert [h.killed for h in cluster.handles] == [False, True, True]

    async def test_shutdown_still_stops_everything(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell
        await manager.shutdown()
        assert all(d["state"] == "stopped" for d in _describe(manager).values())
        assert all(h.killed for h in cluster.handles)


class TestALostCellCanBeStartedAgain:
    async def test_start_cells_clears_the_loss(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell

        await manager.start_cells([_cell(1)])

        described = _describe(manager)[_cell(1)]
        assert described["state"] == "running" and described["lost_workers"] is None
        assert described["generation"] == 2
        assert len(cluster.handles) == 4

    async def test_an_explicitly_stopped_cell_is_never_reported_lost(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells([_cell(0)])
        described = _describe(manager)[_cell(0)]
        assert described["state"] == "stopped" and described["lost_workers"] is None
