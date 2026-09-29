from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.test_inference_controller import _make_controller, _RecordingServer

from miles.ray.rollout.inference_controller import UpdatableEngines


class _Cell:
    def __init__(self, cell_id: str, *, gpu_offset: int, workers_hash: str = "h", ready: bool = True) -> None:
        self.meta = SimpleNamespace(
            cell_id=cell_id, gpu_offset=gpu_offset, num_gpus_per_engine=1, workers_hash=workers_hash
        )
        self.api_client = f"client-{cell_id}"
        self.is_pending_weights = True
        self.is_pending_weights_or_serving = ready
        self.is_uninitialized = False
        self.marked_ready = 0

    async def mark_weights_ready(self) -> None:
        self.marked_ready += 1
        self.is_pending_weights = False


def _setup():
    cells = {
        "e0": _Cell("e0", gpu_offset=0, workers_hash="h0"),
        "e1": _Cell("e1", gpu_offset=1, workers_hash="h1"),
        "e2": _Cell("e2", gpu_offset=2, workers_hash="h2", ready=False),
    }
    srv = _RecordingServer(dict(cells), model_name="actor", update_weights=True)
    srv.api_clients = ["all-clients"]
    return _make_controller({"actor": srv}), srv, cells


class TestMemberFilteredUpdateWeights:
    async def test_only_the_named_members_are_returned_in_gpu_offset_order(self):
        """A not-ready cell outside the member set must not block the publish either."""
        controller, _, cells = _setup()

        info = await controller.start_update_weights(members=["e1", "e0"])
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert info == UpdatableEngines(
            rollout_engines=["client-e0", "client-e1"],
            engine_gpu_counts=[1, 1],
            engine_gpu_offsets=[0, 1],
            snapshot_cell_id_to_hashes={"e0": "h0", "e1": "h1"},
        )

    async def test_end_marks_only_members_ready_and_leaves_the_rest_untouched(self):
        controller, _, cells = _setup()
        cells["e2"].is_pending_weights_or_serving = True

        info = await controller.start_update_weights(members=["e2"])
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert {cid: c.marked_ready for cid, c in cells.items()} == {"e0": 0, "e1": 0, "e2": 1}

    async def test_an_unknown_member_is_refused_and_releases_the_lock(self):
        controller, _, _ = _setup()

        with pytest.raises(KeyError, match="e9"):
            await controller.start_update_weights(members=["e0", "e9"])

        assert not controller.context_lock._lock.locked()

    async def test_without_members_every_cell_is_used_as_before(self, monkeypatch):
        controller, srv, _ = _setup()
        waited: list = []

        async def _ready(model_id=None, cell_ids=None):
            waited.append(cell_ids)

        monkeypatch.setattr(controller, "_ensure_cells_ready", _ready)
        info = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert waited == [None]
        assert info.rollout_engines == ["all-clients"]
        assert set(info.snapshot_cell_id_to_hashes) == {"e0", "e1", "e2"}


class TestUpdateWeightsForwardsMembers:
    async def test_members_reach_the_controller_only_when_given(self, monkeypatch):
        from argparse import Namespace

        from miles.ray import placement_group as placement_group_module
        from miles.ray.placement_group import update_weights

        monkeypatch.setattr(
            placement_group_module,
            "FTTestActionOrchestrationExecutor",
            MagicMock(from_args=MagicMock(return_value=MagicMock(run_after_step=AsyncMock()))),
        )
        args = Namespace(debug_train_only=True, debug_rollout_only=False)
        actor = MagicMock(update_weights=AsyncMock(return_value=None))
        controller = MagicMock(start_update_weights=AsyncMock(), end_update_weights=AsyncMock())

        await update_weights(args, actor, MagicMock(), controller)
        await update_weights(args, actor, MagicMock(), controller, members=["e1"])

        assert controller.start_update_weights.await_args_list[0].kwargs == {"model_id": None}
        assert controller.start_update_weights.await_args_list[1].kwargs == {"model_id": None, "members": ["e1"]}
