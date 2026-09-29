from types import SimpleNamespace

import pytest

from miles.ray.train.group import TrainerController
from miles.ray.train_actor import TrainRayActor
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs

pytestmark = pytest.mark.asyncio

_PLUGIN_PATH = f"{__name__}._plugin_scale_rank"


def _plugin_scale_rank(actor, *, scale: int = 1):
    return actor.rank * scale


async def _async_plugin(actor):
    return actor.rank


class _FakeCell:
    def __init__(self, *, cell_index: int, ranks: list[int], is_alive: bool = True) -> None:
        self.cell_index = cell_index
        self.cell_id = f"trainer-engine-actor-{cell_index}"
        self.is_alive = is_alive
        self.actors = [SimpleNamespace(rank=r) for r in ranks]
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, fn_name: str, **kwargs) -> list:
        self.calls.append((fn_name, kwargs))
        return [getattr(TrainRayActor, fn_name)(actor, **kwargs) for actor in self.actors]


def _make_controller(cells: list[_FakeCell]) -> TrainerController:
    controller = object.__new__(TrainerController)
    controller._cells_by_id = {cell.cell_id: cell for cell in cells}
    return controller


class TestTrainActorRunPlugin:
    def test_the_plugin_receives_the_actor_and_its_kwargs(self):
        """Out-of-tree code needs the worker itself, e.g. to reach its model and optimizer."""
        actor = SimpleNamespace(rank=3)

        assert TrainRayActor.run_plugin(actor, _PLUGIN_PATH, kwargs={"scale": 2}) == 6
        assert TrainRayActor.run_plugin(actor, _PLUGIN_PATH) == 3

    def test_an_async_plugin_is_refused(self):
        """The worker runs the plugin synchronously; a coroutine would be returned unawaited."""
        with pytest.raises(ValueError, match="async"):
            TrainRayActor.run_plugin(SimpleNamespace(rank=0), f"{__name__}._async_plugin")

    def test_the_entry_is_callable_over_rpc(self):
        """Under --worker-comm-backend rpc an unannotated public method makes the pool unreachable."""
        spec = collect_rpc_method_specs(TrainRayActor)["run_plugin"]
        query = dict(fn_path=_PLUGIN_PATH, kwargs={"scale": 2})

        assert spec.serializer.decode_query(spec.serializer.encode_query(query)) == query


class TestTrainerControllerRunPlugin:
    async def test_every_worker_runs_the_plugin_and_results_come_back_in_cell_order(self):
        """Per-rank results are positional; a reordered list would misattribute state to ranks."""
        cells = [_FakeCell(cell_index=1, ranks=[2, 3]), _FakeCell(cell_index=0, ranks=[0, 1])]
        controller = _make_controller(cells)

        assert await controller.run_plugin(_PLUGIN_PATH, kwargs={"scale": 10}) == [0, 10, 20, 30]
        assert all(cell.calls == [("run_plugin", dict(fn_path=_PLUGIN_PATH, kwargs={"scale": 10}))] for cell in cells)

    async def test_a_controller_with_a_dead_cell_refuses_to_run_anything(self):
        """A plugin applied to only part of the fleet leaves the ranks with diverged state."""
        cells = [_FakeCell(cell_index=0, ranks=[0]), _FakeCell(cell_index=1, ranks=[1], is_alive=False)]
        controller = _make_controller(cells)

        with pytest.raises(AssertionError, match="alive"):
            await controller.run_plugin(_PLUGIN_PATH)

        assert [cell.calls for cell in cells] == [[], []]

    def test_the_entry_is_callable_over_rpc(self):
        """A split run reaches this pool only over rpc."""
        assert "run_plugin" in collect_rpc_method_specs(TrainerController)
