from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.test_inference_controller import _FakeWorkerProvider, _make_controller, _RecordingServer

from miles.ray.rollout.inference_controller import MembershipEpochMismatchError, MembershipIncompleteError

_DECLARED = ["engine-0", "engine-1", "engine-2"]


class _MembershipProvider(_FakeWorkerProvider):
    def __init__(self, declared: list[str] = _DECLARED) -> None:
        super().__init__([])
        self.declared = list(declared)
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_next: Exception | None = None

    async def list_declared_cell_ids(self) -> list[str]:
        return list(self.declared)

    async def start_cells(self, *, cell_ids: list[str]) -> None:
        self._maybe_fail()
        self.calls.append(("start", cell_ids))

    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        self._maybe_fail()
        self.calls.append(("stop", cell_ids))

    def _maybe_fail(self) -> None:
        if (error := self.fail_next) is not None:
            self.fail_next = None
            raise error


def _controller(provider=None, servers=None):
    provider = provider or _MembershipProvider()
    servers = servers if servers is not None else {"default": _RecordingServer()}
    return _make_controller(servers, engine_provider=provider), provider, servers


class TestExplicitCellMembership:
    async def test_a_fresh_controller_starts_at_epoch_zero(self):
        controller, _, _ = _controller()
        assert await controller.get_membership_epoch() == 0

    async def test_start_reaches_the_provider_and_bumps_the_epoch(self):
        controller, provider, _ = _controller()

        epoch = await controller.start_cells(["engine-2", "engine-1"], expected_epoch=0)

        assert epoch == 1
        assert provider.calls == [("start", ["engine-1", "engine-2"])]
        assert await controller.get_membership_epoch() == 1

    async def test_stop_deregisters_a_tracked_cell_before_stopping_it(self):
        srv = _RecordingServer(server_cells={"engine-0": SimpleNamespace(), "engine-1": SimpleNamespace()})
        controller, provider, _ = _controller(servers={"default": srv})

        epoch = await controller.stop_cells(["engine-0"], expected_epoch=0)

        assert epoch == 1
        assert srv.calls == [("remove", "engine-0")]
        assert sorted(srv.server_cells) == ["engine-1"]
        assert provider.calls == [("stop", ["engine-0"])]

    async def test_a_stale_epoch_is_refused_without_acting(self):
        controller, provider, _ = _controller()
        await controller.start_cells(["engine-0"], expected_epoch=0)

        with pytest.raises(MembershipEpochMismatchError, match="expected membership epoch 0, but it is 1"):
            await controller.stop_cells(["engine-0"], expected_epoch=0)
        with pytest.raises(MembershipEpochMismatchError):
            await controller.start_cells(["engine-1"], expected_epoch=5)

        assert provider.calls == [("start", ["engine-0"])]

    async def test_repeating_the_committed_call_is_idempotent(self):
        """A retried request after a lost reply must not start the cells a second time or move the epoch."""
        controller, provider, _ = _controller()
        first = await controller.start_cells(["engine-0", "engine-1"], expected_epoch=0)

        again = await controller.start_cells(["engine-1", "engine-0", "engine-0"], expected_epoch=0)

        assert first == again == 1
        assert provider.calls == [("start", ["engine-0", "engine-1"])]

    async def test_only_the_last_committed_call_is_replayed(self):
        controller, provider, _ = _controller()
        await controller.start_cells(["engine-0"], expected_epoch=0)
        await controller.stop_cells(["engine-0"], expected_epoch=1)

        with pytest.raises(MembershipEpochMismatchError):
            await controller.start_cells(["engine-0"], expected_epoch=0)
        assert await controller.stop_cells(["engine-0"], expected_epoch=1) == 2
        assert len(provider.calls) == 2

    async def test_undeclared_cells_are_refused(self):
        controller, provider, _ = _controller()

        with pytest.raises(KeyError, match="engine-9"):
            await controller.start_cells(["engine-0", "engine-9"], expected_epoch=0)

        assert provider.calls == []
        assert await controller.get_membership_epoch() == 0

    async def test_a_failed_provider_call_leaves_the_epoch_for_a_retry(self):
        controller, provider, _ = _controller()
        provider.fail_next = RuntimeError("launch failed")

        with pytest.raises(RuntimeError, match="launch failed"):
            await controller.start_cells(["engine-0"], expected_epoch=0)
        assert await controller.get_membership_epoch() == 0

        assert await controller.start_cells(["engine-0"], expected_epoch=0) == 1

    async def test_a_provider_without_on_demand_cells_is_refused(self):
        controller, _, _ = _controller(provider=_FakeWorkerProvider([]))

        with pytest.raises(NotImplementedError, match="cannot start or stop cells"):
            await controller.start_cells(["engine-0"], expected_epoch=0)

    async def test_an_empty_request_is_refused(self):
        controller, _, _ = _controller()
        with pytest.raises(AssertionError, match="at least one cell"):
            await controller.stop_cells([], expected_epoch=0)


class TestHalfFailedStop:
    async def test_a_failed_stop_is_recorded_and_only_its_retry_is_accepted(self):
        srv = _RecordingServer(server_cells={"engine-0": SimpleNamespace(), "engine-1": SimpleNamespace()})
        controller, provider, _ = _controller(servers={"default": srv})
        provider.fail_next = RuntimeError("kill failed")

        with pytest.raises(RuntimeError, match="kill failed"):
            await controller.stop_cells(["engine-0"], expected_epoch=0)

        assert "engine-0" not in srv.server_cells
        assert await controller.get_membership_status() == dict(
            epoch=0,
            incomplete=["stop", ["engine-0"]],
            incomplete_reason="stop_failed",
            retry_advances_epoch=True,
        )
        with pytest.raises(MembershipIncompleteError, match="retry it before"):
            await controller.start_cells(["engine-2"], expected_epoch=0)
        with pytest.raises(MembershipIncompleteError):
            await controller.start_update_weights(members=["engine-1"], expected_epoch=0)

        assert await controller.stop_cells(["engine-0"], expected_epoch=0) == 1
        assert await controller.get_membership_status() == dict(
            epoch=1,
            incomplete=None,
            incomplete_reason=None,
            retry_advances_epoch=False,
        )


class _TrackedCell:
    def __init__(self, ready: bool) -> None:
        self.is_pending_weights_or_serving = ready


class TestWaitCellsTracked:
    async def test_it_returns_once_the_watch_has_added_the_started_cells(self):
        srv = _RecordingServer(server_cells={})
        controller, _, _ = _controller(servers={"default": srv})
        polls = 0

        def _watch_step() -> None:
            nonlocal polls
            polls += 1
            if polls == 1:
                srv.server_cells["engine-1"] = _TrackedCell(ready=False)
            else:
                srv.server_cells["engine-1"].is_pending_weights_or_serving = True

        original_find = controller._find_cell_or_none

        def _find_after_a_watch_step(cell_id):
            _watch_step()
            return original_find(cell_id)

        controller._find_cell_or_none = _find_after_a_watch_step

        await controller.wait_cells_tracked(["engine-1"], timeout_seconds=10, poll_interval_seconds=0)
        assert polls == 2

    async def test_it_times_out_with_the_missing_cells(self):
        controller, _, _ = _controller(servers={"default": _RecordingServer(server_cells={})})

        with pytest.raises(TimeoutError, match="engine-7"):
            await controller.wait_cells_tracked(["engine-7"], timeout_seconds=0, poll_interval_seconds=0)
