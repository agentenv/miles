from __future__ import annotations

from typing import Any

import pytest

from miles.utils.workers.worker_provider.ray import RayWorkerProvider


class _Recorder:
    def __init__(self, calls: list, name: str, answer: Any = None) -> None:
        self._calls, self._name, self._answer = calls, name, answer

    def remote(self, *args, **kwargs):
        self._calls.append((self._name, args, kwargs))

        async def _resolved():
            return self._answer

        return _resolved()


class _FakeManagerHandle:
    def __init__(self) -> None:
        self.calls: list = []
        self.start_cells = _Recorder(self.calls, "start_cells")
        self.stop_cells = _Recorder(self.calls, "stop_cells")
        self.get_cell_infos = _Recorder(
            self.calls, "get_cell_infos", answer={"pool-a-1": object(), "pool-a-0": object()}
        )
        self.describe_cells = _Recorder(self.calls, "describe_cells", answer={"pool-a-0": {"state": "unbound"}})


class TestRayWorkerProviderMembership:
    async def test_start_and_stop_forward_the_cell_ids_to_the_manager(self):
        handle = _FakeManagerHandle()
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["pool-a"])

        await provider.start_cells(cell_ids=["pool-a-1"])
        await provider.stop_cells(cell_ids=["pool-a-0"])

        assert handle.calls == [("start_cells", (["pool-a-1"],), {}), ("stop_cells", (["pool-a-0"],), {})]

    async def test_declared_cells_are_every_cell_of_the_watched_pools_alive_or_not(self):
        handle = _FakeManagerHandle()
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["pool-a"])

        assert await provider.list_declared_cell_ids() == ["pool-a-0", "pool-a-1"]
        assert handle.calls == [("get_cell_infos", (), {"pool_ids": ["pool-a"]})]

    async def test_declared_cells_need_the_watched_pools(self):
        provider = RayWorkerProvider(worker_manager_handle=_FakeManagerHandle())
        with pytest.raises(AssertionError, match="pool_ids"):
            await provider.list_declared_cell_ids()

    async def test_describe_forwards_the_watched_pools(self):
        handle = _FakeManagerHandle()
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["pool-a"])

        assert await provider.describe_declared_cells() == {"pool-a-0": {"state": "unbound"}}
        assert handle.calls == [("describe_cells", (), {"pool_ids": ["pool-a"]})]
