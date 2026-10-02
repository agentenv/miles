"""A27 (GPU 1r5 d2), without Ray: an engine-side failure during update_weights must not cost the trainer cell.

The trainer's connect to a SIGKILLed engine raised; ``TrainerCell._execute_raw`` treated it as a worker failure,
killed all four healthy trainer ranks, and the next train step found no cell ("Cannot recover when all cells are
dead"), taking the healthy old engines down with the island. The Ray-backed versions of these checks live in
tests/fast/ray/train (test_cell.py::TestExternalFailure, test_group.py::TestUpdateWeightsExternalFailure).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from miles.ray.train.cell import TrainerCell
from miles.ray.train.cell_state import StateAllocatedAlive
from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import NonRetryableError
from miles.utils.workers.worker_handle import BaseWorkerHandle, ExternalFailureError

pytestmark = pytest.mark.asyncio


class _EngineDied(ExternalFailureError):
    pass


class _RankHandle(BaseWorkerHandle):
    """One trainer rank: records calls, fails update_weights while ``engine_dead`` is set, never dies on its own."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.engine_dead = True
        self.killed = False

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        return None

    async def wait_dead(self, *, timeout: float) -> None:
        if not self.killed:
            raise AssertionError("wait_dead on a rank that was never killed")

    async def kill_self(self) -> None:
        self.killed = True

    async def probe_is_dead(self) -> bool:
        return self.killed

    async def update_weights(self, **kwargs) -> int:
        self.calls.append("update_weights")
        if self.engine_dead:
            raise _EngineDied("engine 0 failed to join weight update group miles-pp_0")
        return 7

    async def train(self, **kwargs) -> str:
        self.calls.append("train")
        return "trained"


def _alive_cell(handles: list[_RankHandle]) -> TrainerCell:
    cell = object.__new__(TrainerCell)
    cell.cell_id = "trainer-engine-actor-00000"
    cell.cell_index = 0
    cell._state = StateAllocatedAlive(
        worker_handles=handles,
        indep_dp_info=IndepDPInfo(
            cell_index=0, num_cells=1, alive_rank=0, alive_size=1, quorum_id=0, alive_cell_indices=[0]
        ),
    )
    cell.health_checker = MagicMock()
    return cell


def _controller(cell: TrainerCell) -> TrainerController:
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell}
    return group


class TestTheCellSurvivesAnEngineSideFailure:
    async def test_the_cell_stays_alive_and_no_rank_is_killed(self):
        handles = [_RankHandle() for _ in range(4)]
        cell = _alive_cell(handles)

        with pytest.raises(ExternalFailureError, match="engine 0 failed to join"):
            await cell.execute("update_weights", info=None)

        assert cell.is_alive and not cell.is_errored
        assert not any(h.killed for h in handles)
        assert await cell.execute("train", rollout_id=1) == ["trained"] * 4

    async def test_a_trainer_side_failure_still_errors_the_cell_and_kills_every_rank(self):
        handles = [_RankHandle() for _ in range(2)]
        for h in handles:
            h.engine_dead = False

        async def broken(**kwargs):
            raise RuntimeError("NCCL init failed")

        handles[1].update_weights = broken  # type: ignore[method-assign]
        cell = _alive_cell(handles)

        with pytest.raises(RuntimeError, match="NCCL init failed"):
            await cell.execute("update_weights", info=None)

        assert cell.is_errored
        assert all(h.killed for h in handles)


class TestTheControllerDoesNotRetryAnEngineSideFailure:
    async def test_the_failure_is_non_retryable_and_the_next_publication_runs_on_the_same_cell(self):
        handles = [_RankHandle() for _ in range(4)]
        cell = _alive_cell(handles)
        group = _controller(cell)

        started = asyncio.get_running_loop().time()
        with pytest.raises(NonRetryableError, match="rollout engine side.*engine 0 failed to join"):
            await group.update_weights(info=None, rollout_id=1)
        elapsed = asyncio.get_running_loop().time() - started

        assert all(h.calls == ["update_weights"] for h in handles), "the dead engines were not retried"
        assert elapsed < 0.5, f"a retry backoff ({elapsed:.1f}s) was taken"
        assert cell.is_alive and not any(h.killed for h in handles)

        for h in handles:  # REBUILT_OLD: the old members are the publication target again
            h.engine_dead = False
        assert await group.update_weights(info=None, rollout_id=2) == 7
        assert cell.is_alive
