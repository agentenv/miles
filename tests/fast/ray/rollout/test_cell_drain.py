from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.test_inference_controller import _make_controller
from tests.fast.ray.rollout.test_server_cell_dispose import _make_cell

from miles.ray.rollout.cell_state import CellAddrInfo, StatePendingWeights, StateServing

_URL = "http://10.0.0.2:30000"


def _router_client(inflight_sequence: list[dict[str, int]]) -> MagicMock:
    client = MagicMock()
    client.cordon_worker = AsyncMock()
    client.uncordon_worker = AsyncMock()
    client.remove_worker = AsyncMock()
    client.get_worker_inflight = AsyncMock(side_effect=list(inflight_sequence))
    return client


def _serving_cell(client: MagicMock, *, use_miles_router: bool = True, url: str = _URL):
    cell = _make_cell(router_api_client=client, use_miles_router=use_miles_router)
    cell._state = StateServing(addr_info=CellAddrInfo(server_url=url, bootstrap_port=None, gate_url=None))
    return cell


class TestServerCellCordon:
    async def test_cordon_and_uncordon_name_the_cells_url(self):
        client = _router_client([])
        cell = _serving_cell(client)
        try:
            await cell.cordon()
            await cell.uncordon()
        finally:
            await cell.dispose()

        client.cordon_worker.assert_awaited_once_with(worker_url=_URL)
        client.uncordon_worker.assert_awaited_once_with(worker_url=_URL)

    async def test_cordon_does_not_deregister_the_cell(self):
        client = _router_client([])
        cell = _serving_cell(client)
        await cell.cordon()

        client.remove_worker.assert_not_awaited()
        assert cell.is_serving
        await cell.dispose()

    async def test_a_cell_that_is_not_registered_cannot_be_cordoned(self):
        client = _router_client([])
        cell = _make_cell(router_api_client=client, use_miles_router=True)
        cell._state = StatePendingWeights(addr_info=CellAddrInfo(server_url=_URL, bootstrap_port=None, gate_url=None))

        with pytest.raises(AssertionError, match="only a serving cell"):
            await cell.cordon()
        await cell.dispose()

    async def test_cordon_needs_the_miles_router(self):
        cell = _serving_cell(_router_client([]), use_miles_router=False)
        with pytest.raises(AssertionError, match="--use-miles-router"):
            await cell.cordon()
        await cell.dispose()

    async def test_wait_drained_returns_true_once_nothing_is_in_flight(self):
        client = _router_client([{_URL: 2}, {_URL: 1, "other": 9}, {"other": 9}])
        cell = _serving_cell(client)

        assert await cell.wait_drained(timeout_seconds=10, poll_interval_seconds=0) is True
        assert client.get_worker_inflight.await_count == 3
        await cell.dispose()

    async def test_wait_drained_gives_up_at_the_deadline_without_aborting(self):
        client = _router_client([{_URL: 1}] * 3)
        client.abort_all_requests = AsyncMock()
        cell = _serving_cell(client)

        assert await cell.wait_drained(timeout_seconds=0, poll_interval_seconds=0) is False
        client.abort_all_requests.assert_not_awaited()
        await cell.dispose()


class _FakeDrainCell:
    def __init__(self, cell_id: str, inflight: list[int]) -> None:
        self.meta = SimpleNamespace(cell_id=cell_id)
        self.is_serving = True
        self._inflight = list(inflight)
        self.cordoned = False
        self.aborted = False

    async def cordon(self) -> None:
        self.cordoned = True

    async def uncordon(self) -> None:
        self.cordoned = False

    async def get_inflight(self) -> int:
        return self._inflight.pop(0) if len(self._inflight) > 1 else self._inflight[0]

    async def abort_all(self) -> None:
        self.aborted = True


def _controller_with(*cells: _FakeDrainCell):
    srv = SimpleNamespace(server_cells={cell.meta.cell_id: cell for cell in cells})
    return _make_controller({"default": srv}), srv


class TestInferenceControllerDrain:
    async def test_drain_cordons_then_waits_for_zero_in_flight(self):
        a, b = _FakeDrainCell("a", [2, 1, 0]), _FakeDrainCell("b", [0])
        controller, _ = _controller_with(a, b, _FakeDrainCell("c", [5]))

        assert await controller.drain_cells(["a", "b"], timeout_seconds=10, poll_interval_seconds=0) is True
        assert a.cordoned and b.cordoned

    async def test_drain_times_out_without_aborting_and_leaves_the_cells_cordoned(self):
        a = _FakeDrainCell("a", [3])
        controller, _ = _controller_with(a)

        assert await controller.drain_cells(["a"], timeout_seconds=0, poll_interval_seconds=0) is False
        assert a.cordoned and not a.aborted

        await controller.uncordon_cells(["a"])
        assert not a.cordoned

    async def test_a_cell_removed_while_waiting_counts_as_drained(self):
        a = _FakeDrainCell("a", [1, 1, 1])
        controller, srv = _controller_with(a)
        polls = 0
        original = a.get_inflight

        async def _get_inflight_then_vanish() -> int:
            nonlocal polls
            polls += 1
            if polls == 2:
                del srv.server_cells["a"]
            return await original()

        a.get_inflight = _get_inflight_then_vanish

        assert await controller.drain_cells(["a"], timeout_seconds=10, poll_interval_seconds=0) is True

    async def test_unknown_cells_are_refused_before_cordoning_anything(self):
        a = _FakeDrainCell("a", [0])
        controller, _ = _controller_with(a)

        with pytest.raises(KeyError, match="zz"):
            await controller.drain_cells(["a", "zz"], timeout_seconds=1)
        with pytest.raises(KeyError):
            await controller.cordon_cells(["zz"])
        assert not a.cordoned
