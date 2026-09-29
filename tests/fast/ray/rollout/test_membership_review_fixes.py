"""Review fixes on the elastic membership API: journal restore, start rollback, tracked cells, cordoned
admission, weight-version commit, and drains that race a publish."""

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.test_inference_controller import _make_controller, _RecordingServer
from tests.fast.ray.rollout.test_inference_controller_membership import _controller, _MembershipProvider

from miles.ray import placement_group as placement_group_module
from miles.ray.rollout.inference_controller import (
    CellsNotTrackedError,
    MembershipEpochMismatchError,
    MembershipIncompleteError,
)
from miles.utils.workers.ray_worker_manager import StartRollbackFailedError


class _Cell:
    """A cell with the state flags, router registration and weight version the controller looks at."""

    def __init__(self, cell_id: str, *, serving: bool = False, version: str = "3", gpu_offset: int = 0) -> None:
        self.meta = SimpleNamespace(
            cell_id=cell_id, gpu_offset=gpu_offset, num_gpus_per_engine=1, workers_hash=f"h-{cell_id}"
        )
        self.is_serving = serving
        self.is_pending_weights = not serving
        self.is_pending_weights_or_serving = True
        self.is_uninitialized = False
        self.cordoned = False
        self.registered_cordoned: bool | None = None
        self.inflight = [0]
        self.version = version
        self.api_client = SimpleNamespace(
            get_weight_version=self._get_version, update_weight_version=self._set_version
        )

    async def _get_version(self):
        return self.version

    async def _set_version(self, version: str):
        self.version = version

    async def mark_weights_ready(self, *, cordoned: bool = False) -> None:
        assert self.is_pending_weights
        self.registered_cordoned = cordoned
        self.cordoned = cordoned
        self.is_pending_weights, self.is_serving = False, True

    async def cordon(self) -> None:
        assert self.is_serving
        self.cordoned = True

    async def uncordon(self) -> None:
        assert self.is_serving
        self.cordoned = False

    async def get_inflight(self) -> int:
        return self.inflight.pop(0) if len(self.inflight) > 1 else self.inflight[0]


def _publish_setup(*cells: _Cell, miles_router: bool = True):
    srv = _RecordingServer({c.meta.cell_id: c for c in cells}, model_name="actor", update_weights=True)
    controller = _make_controller({"actor": srv})
    controller.args.use_miles_router = miles_router
    return controller, srv


# ----------------------------------------------------------------------------- B3


class TestRestoreMembershipFromTheJournal:
    async def test_a_restarted_controller_takes_the_epoch_and_incomplete_state(self):
        controller, provider, _ = _controller()
        status = await controller.restore_membership_state(
            epoch=7, incomplete=["stop", ["engine-1"]], expected_current_epoch=0
        )
        assert status == dict(epoch=7, incomplete=["stop", ["engine-1"]])

        with pytest.raises(MembershipIncompleteError):
            await controller.start_cells(["engine-2"], expected_epoch=7)
        assert await controller.stop_cells(["engine-1"], expected_epoch=7) == 8
        assert provider.calls == [("stop", ["engine-1"])]
        assert (await controller.get_membership_status())["incomplete"] is None

    async def test_restore_is_a_compare_and_set(self):
        controller, _, _ = _controller()
        await controller.start_cells(["engine-0"], expected_epoch=0)
        with pytest.raises(MembershipEpochMismatchError, match="restore expected membership epoch 0, but it is 1"):
            await controller.restore_membership_state(epoch=5, expected_current_epoch=0)
        assert await controller.get_membership_epoch() == 1

    async def test_last_op_restores_the_idempotent_retry(self):
        controller, provider, _ = _controller()
        await controller.restore_membership_state(
            epoch=4, last_op=["start", ["engine-1", "engine-0"]], expected_current_epoch=0
        )
        assert await controller.start_cells(["engine-0", "engine-1"], expected_epoch=3) == 4
        assert provider.calls == []

    @pytest.mark.parametrize("bad", [dict(incomplete=["drop", ["x"]]), dict(last_op=["start", []]), dict(epoch=-1)])
    async def test_malformed_state_is_refused(self, bad):
        controller, _, _ = _controller()
        with pytest.raises(ValueError):
            await controller.restore_membership_state(**{"epoch": 1, **bad}, expected_current_epoch=0)
        assert await controller.get_membership_status() == dict(epoch=0, incomplete=None)


class _RayTaskErrorLike(RuntimeError):
    def __init__(self, cause):
        super().__init__(str(cause))
        self.cause = cause


class TestStartRollbackFailure:
    @pytest.mark.parametrize("wrap", [False, True])
    async def test_a_failed_rollback_marks_the_cells_to_be_stopped(self, wrap):
        controller, provider, _ = _controller()
        error = StartRollbackFailedError("start and rollback failed")
        provider.fail_next = _RayTaskErrorLike(error) if wrap else error

        with pytest.raises(RuntimeError):
            await controller.start_cells(["engine-1"], expected_epoch=0)

        assert await controller.get_membership_status() == dict(epoch=0, incomplete=["stop", ["engine-1"]])
        with pytest.raises(MembershipIncompleteError):
            await controller.start_cells(["engine-1"], expected_epoch=0)
        assert await controller.stop_cells(["engine-1"], expected_epoch=0) == 1

    async def test_a_clean_rollback_leaves_membership_complete(self):
        controller, provider, _ = _controller()
        provider.fail_next = RuntimeError("start failed, rolled back")
        with pytest.raises(RuntimeError):
            await controller.start_cells(["engine-1"], expected_epoch=0)
        assert await controller.get_membership_status() == dict(epoch=0, incomplete=None)


# ----------------------------------------------------------------------------- B4


class _TrackingProvider(_MembershipProvider):
    """start_cells makes the cell show up in the server after ``delay_polls`` sleeps of the controller."""

    def __init__(self, srv: _RecordingServer, *, track: bool) -> None:
        super().__init__()
        self.srv, self.track = srv, track

    async def start_cells(self, *, cell_ids):
        await super().start_cells(cell_ids=cell_ids)
        if self.track:

            async def _later():
                await asyncio.sleep(0.01)
                for cell_id in cell_ids:
                    self.srv.server_cells[cell_id] = _Cell(cell_id)

            asyncio.get_running_loop().create_task(_later())


class TestStartThenPublish:
    async def test_a_publish_to_an_untracked_member_names_the_fix(self):
        controller, srv = _publish_setup(_Cell("engine-0", serving=True))
        with pytest.raises(CellsNotTrackedError, match="wait_cells_tracked"):
            await controller.start_update_weights(members=["engine-1"], expected_epoch=0)
        assert srv.health_checker_activeness.get().active  # refused before pausing anything
        info = await controller.start_update_weights(members=["engine-0"], expected_epoch=0)  # lock was released
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    async def test_start_cells_can_wait_until_the_cells_are_tracked(self):
        srv = _RecordingServer({}, model_name="actor", update_weights=True)
        provider = _TrackingProvider(srv, track=True)
        controller = _make_controller({"actor": srv}, engine_provider=provider)

        epoch = await controller.start_cells(
            ["engine-1"], expected_epoch=0, wait_tracked_timeout_seconds=5, poll_interval_seconds=0.005
        )

        assert epoch == 1 and "engine-1" in srv.server_cells
        info = await controller.start_update_weights(members=["engine-1"], expected_epoch=1)
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    async def test_a_wait_timeout_leaves_the_epoch_committed(self):
        srv = _RecordingServer({}, model_name="actor", update_weights=True)
        controller = _make_controller({"actor": srv}, engine_provider=_TrackingProvider(srv, track=False))
        with pytest.raises(TimeoutError):
            await controller.start_cells(
                ["engine-1"], expected_epoch=0, wait_tracked_timeout_seconds=0, poll_interval_seconds=0
            )
        assert await controller.get_membership_epoch() == 1


# ----------------------------------------------------------------------------- B5


class TestValidateBeforePause:
    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(members=["engine-0"]), ValueError),
            (dict(members=["engine-0"], expected_epoch=3), MembershipEpochMismatchError),
        ],
    )
    async def test_a_refused_member_publish_does_not_pause_health_monitoring(self, kwargs, error):
        controller, srv = _publish_setup(_Cell("engine-0"))
        with pytest.raises(error):
            await controller.start_update_weights(**kwargs)
        assert srv.health_checker_activeness.get().active

    async def test_an_incomplete_membership_refuses_before_pausing(self):
        controller, srv = _publish_setup(_Cell("engine-0"))
        await controller.restore_membership_state(epoch=0, incomplete=["stop", ["x"]], expected_current_epoch=0)
        with pytest.raises(MembershipIncompleteError):
            await controller.start_update_weights(members=["engine-0"], expected_epoch=0)
        assert srv.health_checker_activeness.get().active

    async def test_an_accepted_publish_still_pauses(self):
        controller, srv = _publish_setup(_Cell("engine-0"))
        info = await controller.start_update_weights(members=["engine-0"], expected_epoch=0)
        assert not srv.health_checker_activeness.get().active
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)


class TestCordonedAdmission:
    async def test_new_members_join_cordoned_and_take_traffic_only_after_admit(self):
        new, old = _Cell("engine-1", gpu_offset=1), _Cell("engine-0", serving=True)
        controller, _ = _publish_setup(old, new)

        info = await controller.start_update_weights(members=["engine-1"], expected_epoch=0, admit_cordoned=True)
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert new.is_serving and new.registered_cordoned is True and new.cordoned
        assert not old.cordoned
        # the caller reads the weights back here (check_weights), then admits
        await controller.admit_cells(["engine-1"], expected_epoch=0)
        assert not new.cordoned

    async def test_the_flag_applies_to_one_publish_only(self):
        a, b = _Cell("engine-0"), _Cell("engine-1", gpu_offset=1)
        controller, _ = _publish_setup(a, b)
        info = await controller.start_update_weights(members=["engine-0"], expected_epoch=0, admit_cordoned=True)
        await controller.abort_update_weights()
        info = await controller.start_update_weights(members=["engine-1"], expected_epoch=0)
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        assert b.registered_cordoned is False

    async def test_admit_checks_epoch_and_serving(self):
        pending, serving = _Cell("engine-1"), _Cell("engine-0", serving=True)
        controller, _ = _publish_setup(pending, serving)
        with pytest.raises(MembershipEpochMismatchError):
            await controller.admit_cells(["engine-0"], expected_epoch=1)
        with pytest.raises(RuntimeError, match="not serving"):
            await controller.admit_cells(["engine-1"], expected_epoch=0)

    async def test_cordoned_admission_needs_the_miles_router(self):
        controller, srv = _publish_setup(_Cell("engine-0"), miles_router=False)
        with pytest.raises(ValueError, match="Miles router"):
            await controller.start_update_weights(members=["engine-0"], expected_epoch=0, admit_cordoned=True)
        assert srv.health_checker_activeness.get().active

    async def test_the_default_publish_registers_uncordoned(self):
        cell = _Cell("engine-0")
        controller, _ = _publish_setup(cell)
        info = await controller.start_update_weights(members=["engine-0"], expected_epoch=0)
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        assert cell.registered_cordoned is False


class _Executor:
    def __init__(self):
        self.versions = []

    async def set_weight_version(self, version, trainer_model_id=None):
        self.versions.append((version, trainer_model_id))


class TestWeightVersionCommit:
    async def test_members_first_then_restamp_then_commit(self):
        """Member publish (v4 to the new cell), re-stamp the old cells carrying the same policy, then commit."""
        old, new = _Cell("engine-0", serving=True, version="3"), _Cell("engine-1", serving=True, version="4")
        controller, _ = _publish_setup(old, new)
        executor = _Executor()

        with pytest.raises(RuntimeError, match=r"do not report weight version 4"):
            await placement_group_module.commit_weight_version(
                executor, controller, weight_version=4, expected_epoch=0, trainer_model_id="actor"
            )
        assert executor.versions == []

        await controller.set_cells_weight_version(["engine-0"], weight_version=4, expected_epoch=0)
        await placement_group_module.commit_weight_version(
            executor, controller, weight_version=4, expected_epoch=0, trainer_model_id="actor"
        )
        assert executor.versions == [(4, "actor")]
        assert await controller.get_cells_weight_versions() == {"engine-0": "4", "engine-1": "4"}

    async def test_pending_cells_are_not_counted_and_epochs_are_checked(self):
        serving, pending = _Cell("engine-0", serving=True, version="5"), _Cell("engine-1", version="default")
        controller, _ = _publish_setup(serving, pending)
        executor = _Executor()

        await placement_group_module.commit_weight_version(executor, controller, weight_version=5, expected_epoch=0)
        with pytest.raises(RuntimeError, match="expected membership epoch 2"):
            await placement_group_module.commit_weight_version(
                executor, controller, weight_version=5, expected_epoch=2
            )
        with pytest.raises(MembershipEpochMismatchError):
            await controller.set_cells_weight_version(["engine-0"], weight_version=6, expected_epoch=1)
        with pytest.raises(KeyError, match="not serving"):
            await controller.set_cells_weight_version(["engine-1"], weight_version=6, expected_epoch=0)
        assert executor.versions == [(5, None)]


# ----------------------------------------------------------------------------- B6


class TestDrainRacesAPublish:
    async def test_a_cell_published_during_the_drain_joins_cordoned_and_is_counted(self):
        busy = _Cell("engine-0", serving=True)
        busy.inflight = [1, 1, 1, 0]
        pending = _Cell("engine-1", gpu_offset=1)
        pending.inflight = [2, 0]
        controller, _ = _publish_setup(busy, pending)

        drain = asyncio.create_task(
            controller.drain_cells(["engine-0", "engine-1"], timeout_seconds=10, poll_interval_seconds=0.01)
        )
        await asyncio.sleep(0.005)  # the drain waits with the lock released
        info = await controller.start_update_weights(members=["engine-1"], expected_epoch=0)
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        assert pending.registered_cordoned is True  # never selectable, not even for one request

        assert await drain is True
        assert busy.cordoned and pending.cordoned
        assert pending.inflight == [0]  # its in-flight count was read, i.e. it was waited for
        assert controller._draining_cell_ids == frozenset()

    async def test_a_cell_that_turns_serving_between_polls_is_cordoned_before_counting(self):
        busy = _Cell("engine-0", serving=True)
        busy.inflight = [1, 0]
        late = _Cell("engine-1")
        controller, _ = _publish_setup(busy, late)
        original = busy.get_inflight

        async def _flip_then_count():
            late.is_serving, late.is_pending_weights = True, False  # e.g. registered by another path
            return await original()

        busy.get_inflight = _flip_then_count
        assert await controller.drain_cells(["engine-0", "engine-1"], timeout_seconds=10, poll_interval_seconds=0)
        assert late.cordoned
