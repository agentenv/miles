from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    pass


class WorkerStillBusyError(Exception):
    pass


class ExternalFailureError(Exception):
    """The worker raising it is healthy: the failure belongs to an external party it talked to.

    A cell treats an exception from a worker call as a worker failure (mark errored, kill every rank). Raise a
    subclass of this when the worker itself is fine and must stay alive, e.g. a rollout engine that died while
    the trainer connected to it for a weight update (yeto A27: killing the trainer there left the island with no
    trainer and brought the healthy old engines down with it). Raise it on *every* rank of the worker so that
    no rank is left behind in a collective.
    """


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None: ...

    async def wait_idle(self, *, timeout: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot tell whether the worker is running a call")

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot submit a call it will never get an answer to")

    async def wait_dead(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if await self.probe_is_dead():
                return
            if time.monotonic() >= deadline:
                logger.error("Timed out after %.0fs waiting for %r to die; proceeding anyway", timeout, self)
                return
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    @abc.abstractmethod
    async def probe_is_dead(self) -> bool: ...
