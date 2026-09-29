import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin, InferenceControllerEvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils import async_utils
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import SimpleTicker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0
DRAIN_POLL_INTERVAL_SECONDS = 0.5


class MembershipEpochMismatchError(RuntimeError):
    pass


class MembershipIncompleteError(RuntimeError):
    """A membership change failed half way; only a retry of that same change is accepted until it succeeds."""


class CellsNotTrackedError(KeyError):
    """Cells were started but no server has taken them in yet; call ``wait_cells_tracked`` before publishing."""


def _is_start_rollback_failure(error: BaseException) -> bool:
    """Whether a start failed *and* its rollback failed (``RayWorkerManager.StartRollbackFailedError``).

    Matched by class name, on the error or on the ``cause`` a Ray actor error carries, so this module does not
    import the worker manager.
    """
    for candidate in (error, getattr(error, "cause", None)):
        names = {cls.__name__ for cls in type(candidate).__mro__} if candidate is not None else set()
        if "StartRollbackFailedError" in names:
            return True
    return False


@enforce_lock_discipline
class InferenceController:
    # class-level defaults so a controller built without __init__ (tests) still has a membership epoch
    _membership_epoch: int = 0
    _last_membership_op: tuple[str, tuple[str, ...]] | None = None
    _membership_incomplete: tuple[str, tuple[str, ...]] | None = None
    # set by start_update_weights(admit_cordoned=True) for the end_update_weights of that same publish
    _admit_cordoned: bool = False
    # cell id -> cell registered cordoned by an admit_cordoned publish and not admitted yet (only admit_cells opens it)
    _awaiting_admission: dict[str, Any] = {}
    # why membership is incomplete: "stop_failed", "start_rollback_failed" or "restored"
    _membership_incomplete_reason: str | None = None
    # cells a drain_cells call is waiting on; if one becomes Serving meanwhile it is registered cordoned
    _draining_cell_ids: frozenset[str] = frozenset()

    @lock_exempt
    def __init__(
        self,
        args,
        *,
        engine_provider: BaseWorkerProvider,
        router_providers: Sequence[BaseWorkerProvider],
    ) -> None:
        self._init_once = InitOnce(type(self).__name__)
        self.args = args
        self._engine_provider = engine_provider
        self._router_providers = router_providers
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self._eval_fleet: InferenceControllerEvalFleet | None = None
        self._watcher_disposers: list[StopWatchFn] = []
        self._ticker: SimpleTicker | None = None

    @lock_exempt
    @init_once
    async def init(self) -> None:
        configure_logger(self.args, source=SimpleProcessIdentity(component="inference_controller"))

        if not self.args.starts_inference_engines:
            return

        await self._engine_provider.init()
        router_addrs = await resolve_router_addrs(self.args, router_providers=self._router_providers)
        self.servers = await create_rollout_servers(
            self.args,
            context_lock=self.context_lock,
            engine_provider=self._engine_provider,
            router_addrs=router_addrs,
        )
        if self.args.eval_num_gpus > 0:
            self._eval_fleet = InferenceControllerEvalFleet(self.args, srv=self.servers["eval"])

        self._watcher_disposers.append(await self._engine_provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)

        await self.wait_expected_num_cells()

    # -------------------------- explicit membership -----------------------------

    @lock_exempt
    async def get_membership_epoch(self) -> int:
        return self._membership_epoch

    @lock_exempt
    async def get_membership_status(self) -> dict[str, Any]:
        """The epoch plus the change that failed half way (``[op, cell_ids]``) and must be retried, if any.

        ``incomplete_reason`` says why (``stop_failed``, ``start_rollback_failed`` or ``restored``). A successful retry
        of the incomplete change is a normal membership change and advances the epoch by one -- also after
        ``start_rollback_failed``, where the retry is a stop of cells that never joined; ``retry_advances_epoch``
        states this explicitly for the caller's journal.
        """
        incomplete = self._membership_incomplete
        return dict(
            epoch=self._membership_epoch,
            incomplete=None if incomplete is None else [incomplete[0], list(incomplete[1])],
            incomplete_reason=None if incomplete is None else self._membership_incomplete_reason,
            retry_advances_epoch=incomplete is not None,
        )

    @with_lock
    async def restore_membership_state(
        self,
        *,
        epoch: int,
        incomplete: list | None = None,
        last_op: list | None = None,
        expected_current_epoch: int,
    ) -> dict[str, Any]:
        """Set the membership epoch and half-failed state from the caller's durable journal.

        The fork keeps membership state only in memory; the caller (yeto, whose journal holds the authoritative
        pool epoch) calls this when it (re)connects, e.g. after this controller restarted at epoch 0. It is a
        compare-and-set on ``expected_current_epoch`` (read it with ``get_membership_status``), so a membership
        change that raced in is not overwritten. ``incomplete`` is ``[op, cell_ids]`` of a change that must be
        retried; ``last_op`` is ``[op, cell_ids]`` of the change that produced ``epoch``, which re-enables the
        idempotent retry of that change. Returns the new status.
        """
        if expected_current_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"restore expected membership epoch {expected_current_epoch}, but it is {self._membership_epoch}"
            )
        if epoch < 0:
            raise ValueError(f"membership epoch must be >= 0, got {epoch}")

        def _op(value, what):
            if value is None:
                return None
            op, cell_ids = value
            if op not in ("start", "stop") or not cell_ids:
                raise ValueError(f"{what} must be [\"start\"|\"stop\", non-empty cell ids], got {value!r}")
            return op, tuple(sorted(set(cell_ids)))

        self._membership_incomplete = _op(incomplete, "incomplete")
        self._membership_incomplete_reason = None if incomplete is None else "restored"
        self._last_membership_op = _op(last_op, "last_op")
        self._membership_epoch = epoch
        logger.info(
            f"membership restored by the caller: epoch={epoch} incomplete={self._membership_incomplete} "
            f"last_op={self._last_membership_op}"
        )
        return await self.get_membership_status()

    @with_lock
    async def wait_cells_tracked(
        self,
        cell_ids: list[str],
        *,
        timeout_seconds: float = CELLS_READY_TIMEOUT_SECONDS,
        poll_interval_seconds: float = CELLS_READY_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Wait until every cell was taken in by a server and is PendingWeights or Serving.

        start_cells returns once the workers run; the cells join the servers through the provider watch later, so a
        member publish right after start_cells must wait here first (or use ``start_cells(wait_tracked_timeout_
        seconds=...)``); a publish to an untracked member raises ``CellsNotTrackedError``.
        """
        await self._wait_cells_tracked(
            cell_ids, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds
        )

    @requires_lock
    async def _wait_cells_tracked(
        self, cell_ids: list[str], *, timeout_seconds: float, poll_interval_seconds: float
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            missing = [
                cell_id
                for cell_id in cell_ids
                if (cell := self._find_cell_or_none(cell_id)) is None or not cell.is_pending_weights_or_serving
            ]
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"cells {missing} were not ready to take weights after {timeout_seconds}s")
            async with self.context_lock.with_released():
                await asyncio.sleep(poll_interval_seconds)

    @with_lock
    async def start_cells(
        self,
        cell_ids: list[str],
        *,
        expected_epoch: int,
        wait_tracked_timeout_seconds: float | None = None,
        poll_interval_seconds: float = CELLS_READY_POLL_INTERVAL_SECONDS,
    ) -> int:
        """Start pre-declared engine cells and return the new membership epoch.

        The cells join through the provider watch (reconcile -> PendingWeights) and take no traffic until their
        weights are published. They are *not* tracked by a server when this returns unless
        ``wait_tracked_timeout_seconds`` is given: a member publish to them must first call ``wait_cells_tracked``
        (otherwise it raises ``CellsNotTrackedError``). With a timeout, this waits after committing the epoch; a
        TimeoutError then leaves the epoch committed and the caller may wait again. A repeated call with the same
        cells right after it committed returns the same epoch without acting again; any other epoch mismatch is
        refused. If the start fails and the worker manager's rollback fails too, the cells may be half running:
        membership is marked incomplete as ``["stop", cell_ids]`` (reason ``start_rollback_failed``) and only a stop
        of those cells is accepted next; that stop, when it succeeds, advances the epoch by one like any stop.
        """
        epoch = await self._change_membership("start", cell_ids, expected_epoch=expected_epoch)
        if wait_tracked_timeout_seconds is not None:
            await self._wait_cells_tracked(
                cell_ids, timeout_seconds=wait_tracked_timeout_seconds, poll_interval_seconds=poll_interval_seconds
            )
        return epoch

    @with_lock
    async def stop_cells(self, cell_ids: list[str], *, expected_epoch: int) -> int:
        """Remove pre-declared engine cells from their servers (router deregistration) and stop them."""
        return await self._change_membership("stop", cell_ids, expected_epoch=expected_epoch)

    @requires_lock
    async def _change_membership(self, op: str, cell_ids: list[str], *, expected_epoch: int) -> int:
        ids = tuple(sorted(set(cell_ids)))
        assert ids, f"{op}_cells needs at least one cell id"
        current = self._membership_epoch
        if self._last_membership_op == (op, ids) and expected_epoch == current - 1:
            logger.info(f"{op}_cells {list(ids)} already committed as epoch {current}; not acting again")
            return current
        if (incomplete := self._membership_incomplete) is not None and incomplete != (op, ids):
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way at epoch {current}; retry it before "
                f"asking for {op}_cells {list(ids)}"
            )
        if expected_epoch != current:
            raise MembershipEpochMismatchError(
                f"{op}_cells {list(ids)} expected membership epoch {expected_epoch}, but it is {current}"
            )

        provider = self._engine_provider
        if not all(hasattr(provider, name) for name in ("start_cells", "stop_cells", "list_declared_cell_ids")):
            raise NotImplementedError(f"{type(provider).__name__} cannot start or stop cells on demand")
        declared = set(await provider.list_declared_cell_ids())
        unknown = sorted(set(ids) - declared)
        if unknown:
            raise KeyError(f"cells {unknown} were not declared at startup; declared cells are {sorted(declared)}")

        if op == "stop":
            # Deregistering first keeps traffic off cells that are going away. If stopping then fails, the cells are
            # out of routing but may still run (the watch re-adds them as PendingWeights, not routed); the epoch
            # stays and only this same stop is accepted until it succeeds.
            try:
                for srv in self.servers.values():
                    for cell_id in ids:
                        if cell_id in srv.server_cells:
                            await srv.remove_cell(cell_id)
                await provider.stop_cells(cell_ids=list(ids))
            except BaseException:
                self._membership_incomplete = (op, ids)
                self._membership_incomplete_reason = "stop_failed"
                logger.error(f"stop_cells {list(ids)} failed half way at epoch {current}; retry the same stop")
                raise
        else:
            # the worker manager rolls a failed start back, so a failed start leaves membership unchanged -- unless
            # that rollback failed as well, in which case the cells may be half running and must be stopped
            try:
                await provider.start_cells(cell_ids=list(ids))
            except BaseException as error:
                if _is_start_rollback_failure(error):
                    self._membership_incomplete = ("stop", ids)
                    self._membership_incomplete_reason = "start_rollback_failed"
                    logger.error(
                        f"start_cells {list(ids)} failed and its rollback failed at epoch {current}; "
                        f"stop these cells before any other membership change"
                    )
                raise

        if op == "stop":
            self._awaiting_admission = {k: v for k, v in self._awaiting_admission.items() if k not in ids}
        self._membership_incomplete = None
        self._membership_incomplete_reason = None
        self._membership_epoch = current + 1
        self._last_membership_op = (op, ids)
        logger.info(f"{op}_cells {list(ids)} committed membership epoch {self._membership_epoch}")
        return self._membership_epoch

    # -------------------------- cordon / drain -----------------------------

    @with_lock
    async def cordon_cells(self, cell_ids: list[str]) -> None:
        await asyncio.gather(*[cell.cordon() for cell in self._find_cells(cell_ids)])

    @with_lock
    async def uncordon_cells(self, cell_ids: list[str]) -> None:
        """Undo a cordon (e.g. cancel a drain). Cells waiting for ``admit_cells`` are refused: only an admission,
        which checks their weight version, may open them."""
        cells = self._find_cells(cell_ids)
        waiting = [cell.meta.cell_id for cell in cells if self._is_awaiting_admission(cell)]
        if waiting:
            raise RuntimeError(f"cells {waiting} await admission after a cordoned publish; use admit_cells")
        await asyncio.gather(*[cell.uncordon() for cell in cells])

    @requires_lock
    def _is_awaiting_admission(self, cell) -> bool:
        return self._awaiting_admission.get(cell.meta.cell_id) is cell

    @with_lock
    async def admit_cells(
        self, cell_ids: list[str], *, expected_epoch: int, expected_weight_version: str | int
    ) -> None:
        """Let cells published with ``admit_cordoned=True`` take traffic, after the caller verified their weights.

        The cells must be awaiting admission (registered cordoned by that publish's ``end_update_weights``), the
        membership epoch must still be the one the publish was planned for, and every cell must report
        ``expected_weight_version`` (the version the caller's read-back verified); otherwise nothing is admitted.
        """
        if expected_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"admit {cell_ids} expected membership epoch {expected_epoch}, but it is {self._membership_epoch}"
            )
        if (incomplete := self._membership_incomplete) is not None:
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way; retry it first"
            )
        cells = self._find_cells(cell_ids)
        not_serving = [cell.meta.cell_id for cell in cells if not cell.is_serving]
        if not_serving:
            raise RuntimeError(f"cells {not_serving} are not serving (weights not published yet); nothing to admit")
        not_waiting = [cell.meta.cell_id for cell in cells if not self._is_awaiting_admission(cell)]
        if not_waiting:
            raise RuntimeError(f"cells {not_waiting} were not published with admit_cordoned; nothing to admit")
        versions = await asyncio.gather(*[cell.api_client.get_weight_version() for cell in cells])
        wrong = {
            cell.meta.cell_id: str(v)
            for cell, v in zip(cells, versions, strict=True)
            if str(v) != str(expected_weight_version)
        }
        if wrong:
            raise RuntimeError(f"cells {wrong} do not report weight version {expected_weight_version}; not admitted")
        await asyncio.gather(*[cell.uncordon() for cell in cells])
        admitted = {cell.meta.cell_id for cell in cells}
        self._awaiting_admission = {k: v for k, v in self._awaiting_admission.items() if k not in admitted}

    @with_lock
    async def drain_cells(
        self,
        cell_ids: list[str],
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = DRAIN_POLL_INTERVAL_SECONDS,
    ) -> bool:
        """Cordon the cells, then wait until the router counts no in-flight request on any of them.

        Returns False when the deadline passes first; nothing is aborted and the cells stay cordoned (uncordon them
        to cancel). The controller lock is released while waiting. A cell that disappears meanwhile has left the
        router (or was replaced by a cell that is not serving yet), so it holds no in-flight request any more. A
        target cell that becomes Serving while this waits (a publish ran in between) is registered cordoned by
        ``end_update_weights`` and is cordoned here as well before it is counted, so it never takes new traffic.

        In-flight counts are per router registration: if a cell's URL was deregistered and registered again, requests
        still running from the earlier registration are not counted (see ``MilesRouter._finish_url``).
        """
        cells = self._find_cells(cell_ids)
        cordoned: set[int] = set()

        async def _cordon_serving(targets) -> None:
            fresh = [cell for cell in targets if cell.is_serving and id(cell) not in cordoned]
            await asyncio.gather(*[cell.cordon() for cell in fresh])
            cordoned.update(id(cell) for cell in fresh)

        # a cell that is not serving is not registered with the router, so it has nothing to cordon or drain
        await _cordon_serving(cells)
        deadline = time.monotonic() + timeout_seconds
        self._draining_cell_ids = self._draining_cell_ids | frozenset(cell_ids)
        try:
            while True:
                remaining = [
                    cell for cell in map(self._find_cell_or_none, cell_ids) if cell is not None and cell.is_serving
                ]
                await _cordon_serving(remaining)
                counts = await asyncio.gather(*[cell.get_inflight() for cell in remaining])
                busy = {cell.meta.cell_id: n for cell, n in zip(remaining, counts, strict=True) if n > 0}
                if not busy:
                    return True
                if time.monotonic() >= deadline:
                    logger.warning(
                        f"drain of {cell_ids} timed out after {timeout_seconds}s; still in flight: {busy}"
                    )
                    return False
                async with self.context_lock.with_released():
                    await asyncio.sleep(poll_interval_seconds)
        finally:
            self._draining_cell_ids = self._draining_cell_ids - frozenset(cell_ids)

    @requires_lock
    def _find_cell_or_none(self, cell_id: str) -> ServerCell | None:
        for srv in self.servers.values():
            if (cell := srv.server_cells.get(cell_id)) is not None:
                return cell
        return None

    @requires_lock
    def _find_cells(self, cell_ids: list[str]) -> list[ServerCell]:
        cells = [self._find_cell_or_none(cell_id) for cell_id in cell_ids]
        unknown = [cell_id for cell_id, cell in zip(cell_ids, cells, strict=True) if cell is None]
        if unknown:
            raise KeyError(f"cells {unknown} are not tracked by this controller")
        return cells

    # TEMPORARY: exists only so a suspend can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        await self._engine_provider.stop_cells(cell_ids=[cell_id])

    # TEMPORARY: exists only so fault injection can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def inject_fault_between_weight_updates(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # TEMPORARY: colocate cannot kill rollout workers while trainer ranks own the shared GPUs
        server = next((srv for srv in self.servers.values() if cell_id in srv.server_cells), None)
        if server is None:
            raise KeyError(f"Unknown rollout cell {cell_id!r}")
        if not server.health_checker_activeness.get().active:
            raise RuntimeError(f"Rollout cell {cell_id!r} is offloaded; refusing fault injection")

        await self._engine_provider._worker_manager_handle.inject_fault.remote(
            cell_id,
            mode=mode.value,
            worker_in_cell_index=sub_index,
        )

    # -------------------------- take over -----------------------------

    @lock_exempt
    async def is_initialized(self) -> bool:
        return self._init_once.is_initialized()

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = CELLS_READY_TIMEOUT_SECONDS) -> None:
        await asyncio.gather(*[srv.wait_init_expected_num_cells(timeout=timeout) for srv in self.servers.values()])

    @with_lock
    async def abort_all(self) -> None:
        await async_utils.gather_and_raise_first([srv.abort_all() for srv in self.servers.values()])

    # -------------------------- registration -----------------------------

    @lock_exempt
    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        assert self.servers, (
            f"this inference controller is not ready: reporter {snapshot.reporter_id} announced its cells before "
            f"the orchestration script built this controller's servers, so it does not know yet which models this "
            f"run serves; the reporter announces them again, and they are taken in once the controller is up"
        )
        assert isinstance(provider := self._engine_provider, RegistrationHub), (
            f"run {self.args.run_uuid} deploys its own engines and takes no registration "
            f"(reporter {snapshot.reporter_id})"
        )
        for cell in snapshot.cells:
            assert (
                model_id := cell.info.meta.get("model_id")
            ) in self.servers, (
                f"cell {cell.info.cell_id} serves model {model_id!r}, this run serves {sorted(self.servers)}"
            )
        await provider.ingest(snapshot)

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id: int, model_id: str | None = None) -> None:
        await self._health_monitoring_resume(model_id)
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def prepare_eval(self, model_id: str | None = None) -> None:
        await self._health_monitoring_resume(model_id)

    @with_lock
    async def dispose(self) -> None:
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await ticker.dispose()

        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

        for srv in self.servers.values():
            await srv.dispose()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None) -> None:
        await self._offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None) -> None:
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self) -> None:
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @with_lock
    async def offload_kv(self) -> None:
        tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
        if "kv_cache" in self.args.offload_rollout_level:
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
        await self._offload(tags=tags)

    @with_lock
    async def offload_weights(self) -> None:
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @requires_lock
    async def _offload(self, tags: list[str] | None):
        await self._health_monitoring_pause(None)
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @acquires_lock
    async def start_update_weights(
        self,
        model_id: str | None = None,
        members: list[str] | None = None,
        expected_epoch: int | None = None,
        admit_cordoned: bool = False,
    ) -> "UpdatableEngines":
        """Return engines eligible for weight updates.

        ``members`` restricts the update to exactly these cell ids of the updatable server: only they are waited for,
        returned, and snapshotted, so ``end_update_weights`` marks only them ready and every other cell keeps its
        state. ``None`` (the default) means every cell. A member publish must name the membership ``expected_epoch``
        it was planned for; the returned engines carry that epoch, and since membership changes take this lock the
        epoch cannot move before ``end_update_weights``. A member publish is validated before the health monitor is
        paused, so a refused publish changes nothing.

        ``admit_cordoned`` makes ``end_update_weights`` register the cells it marks ready as cordoned: they take no
        traffic until ``admit_cells`` after the caller verified their weights (e.g. ``check_weights`` read-back).
        """
        self._admit_cordoned = False
        if admit_cordoned and not self.args.use_miles_router:
            raise ValueError("admit_cordoned needs the Miles router (--use-miles-router)")
        if members is not None:
            srv = self._validate_member_publish(model_id=model_id, members=members, expected_epoch=expected_epoch)
            await self._health_monitoring_pause(model_id)
            self._admit_cordoned = admit_cordoned
            return await self._start_member_update_weights(
                srv, model_id=model_id, members=members, expected_epoch=expected_epoch
            )
        await self._health_monitoring_pause(model_id)
        self._admit_cordoned = admit_cordoned
        await self._ensure_cells_ready(model_id=model_id)

        srv = self._get_updatable_server(model_id=model_id)
        if not srv:
            return UpdatableEngines(
                rollout_engines=[],
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )

        return UpdatableEngines(
            rollout_engines=srv.api_clients,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell_id: cell.meta.workers_hash for cell_id, cell in srv.server_cells.items()},
        )

    @requires_lock
    def _validate_member_publish(
        self, *, model_id: str | None, members: list[str], expected_epoch: int | None
    ) -> RolloutServer:
        if expected_epoch is None:
            raise ValueError("a member publish needs the membership expected_epoch it was planned for")
        if expected_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"publish to {members} expected membership epoch {expected_epoch}, but it is {self._membership_epoch}"
            )
        if (incomplete := self._membership_incomplete) is not None:
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way; retry it first"
            )
        srv = self._get_updatable_server(model_id=model_id)
        assert srv is not None, f"no updatable server to publish members {members} to"
        unknown = sorted(set(members) - set(srv.server_cells))
        if unknown:
            raise CellsNotTrackedError(
                f"members {unknown} are not cells of {srv.model_name} (yet); its cells are "
                f"{sorted(srv.server_cells)}. start_cells returns before the cells are tracked: call "
                f"wait_cells_tracked (or start_cells(wait_tracked_timeout_seconds=...)) before publishing to them"
            )
        return srv

    @requires_lock
    async def _start_member_update_weights(
        self, srv: RolloutServer, *, model_id: str | None, members: list[str], expected_epoch: int
    ) -> "UpdatableEngines":
        wanted = set(members)
        await self._ensure_cells_ready(model_id=model_id, cell_ids=wanted)
        cells = sorted(
            (cell for cell_id, cell in srv.server_cells.items() if cell_id in wanted),
            key=lambda cell: cell.meta.gpu_offset,
        )
        return UpdatableEngines(
            rollout_engines=[cell.api_client for cell in cells],
            engine_gpu_counts=[cell.meta.num_gpus_per_engine for cell in cells],
            engine_gpu_offsets=[cell.meta.gpu_offset for cell in cells],
            snapshot_cell_id_to_hashes={cell.meta.cell_id: cell.meta.workers_hash for cell in cells},
            membership_epoch=expected_epoch,
        )

    @releases_lock
    async def abort_update_weights(self) -> None:
        self._admit_cordoned = False

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        admit_cordoned, self._admit_cordoned = self._admit_cordoned, False
        draining = self._draining_cell_ids
        ready = [
            (cell_id, cell)
            for srv in self.servers.values()
            for cell_id, cell in srv.server_cells.items()
            if cell_id in snapshot_cell_id_to_hashes
            and snapshot_cell_id_to_hashes[cell_id] == cell.meta.workers_hash
            and cell.is_pending_weights
        ]
        await asyncio.gather(
            *[
                (
                    cell.mark_weights_ready(cordoned=True)
                    if admit_cordoned or cell_id in draining
                    else cell.mark_weights_ready()
                )
                for cell_id, cell in ready
            ]
        )
        if admit_cordoned:
            self._awaiting_admission = {**self._awaiting_admission, **{cell_id: cell for cell_id, cell in ready}}

    # -------------------------- weight versions -----------------------------

    @acquires_lock
    async def start_commit_weight_version(
        self, *, weight_version: int, expected_epoch: int, model_id: str | None = None
    ) -> dict[str, str]:
        """Check, under the controller lock, that the executor may take ``weight_version``; keeps the lock held.

        The caller sets the executor's version and then calls ``end_commit_weight_version`` (always, also on
        failure). While the lock is held no membership change, publish or re-stamp can run, so the epoch and the
        versions checked here are still true when the executor's version is set. Returns cell id -> version.
        """
        if expected_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"weight version commit expected membership epoch {expected_epoch}, "
                f"but it is {self._membership_epoch}"
            )
        if (incomplete := self._membership_incomplete) is not None:
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way; retry it first"
            )
        versions = await self._cells_weight_versions(model_id)
        if not versions:
            raise RuntimeError("no serving engine reports a weight version; nothing to commit")
        stale = {cell_id: v for cell_id, v in versions.items() if v != str(weight_version)}
        if stale:
            raise RuntimeError(f"serving engines {stale} do not report weight version {weight_version} yet")
        return versions

    @releases_lock
    async def end_commit_weight_version(self) -> None:
        pass

    @with_lock
    async def get_cells_weight_versions(self, model_id: str | None = None) -> dict[str, str]:
        """The weight version each Serving cell of the updatable server reports (cell id -> version string)."""
        return await self._cells_weight_versions(model_id)

    @requires_lock
    async def _cells_weight_versions(self, model_id: str | None) -> dict[str, str]:
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return {}
        serving = [(cell_id, cell) for cell_id, cell in srv.server_cells.items() if cell.is_serving]
        versions = await asyncio.gather(*[cell.api_client.get_weight_version() for _, cell in serving])
        return {cell_id: str(version) for (cell_id, _), version in zip(serving, versions, strict=True)}

    @with_lock
    async def set_cells_weight_version(
        self, cell_ids: list[str], *, weight_version: int, expected_epoch: int, model_id: str | None = None
    ) -> None:
        """Re-stamp the weight version of Serving cells without touching their weights.

        For a member publish that carried the policy the other cells already serve: the trainer's version counter
        advanced for the members only, so the caller re-stamps the others with the same number. Only valid when the
        weights are identical; the caller (which knows the policy each version holds) is responsible for that.
        """
        if expected_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"re-stamping {cell_ids} expected membership epoch {expected_epoch}, "
                f"but it is {self._membership_epoch}"
            )
        if (incomplete := self._membership_incomplete) is not None:
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way; retry it first"
            )
        srv = self._get_updatable_server(model_id=model_id)
        assert srv is not None, "no updatable server"
        cells = [srv.server_cells.get(cell_id) for cell_id in cell_ids]
        bad = [cid for cid, cell in zip(cell_ids, cells, strict=True) if cell is None or not cell.is_serving]
        if bad:
            raise KeyError(f"cells {bad} are not serving cells of {srv.model_name}")
        await asyncio.gather(*[cell.api_client.update_weight_version(str(weight_version)) for cell in cells])

    @requires_lock
    async def _ensure_cells_ready(self, model_id: str | None = None, cell_ids: set[str] | None = None) -> None:
        deadline = time.monotonic() + CELLS_READY_TIMEOUT_SECONDS
        while True:
            cells = [
                cell
                for srv in self._get_servers_of_model_id(model_id)
                for cell_id, cell in srv.server_cells.items()
                if cell_ids is None or cell_id in cell_ids
            ]
            if self.args.colocate:
                await asyncio.gather(*[cell.init() for cell in cells if cell.is_uninitialized])
            pending = [cell for cell in cells if not cell.is_pending_weights_or_serving]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {CELLS_READY_TIMEOUT_SECONDS}s waiting for "
                    f"{len(pending)}/{len(cells)} cells to become ready"
                )
            logger.info(f"Waiting for {len(pending)}/{len(cells)} cells to become ready...")
            async with self.context_lock.with_released():
                await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    @requires_lock
    def _get_servers_of_model_id(self, model_id: str | None) -> list[RolloutServer]:
        if model_id is None:
            return list(self.servers.values())
        srv = self.servers.get(model_id)
        assert srv is not None, f"No server for model_id {model_id!r}, known ids: {sorted(self.servers)}"
        return [srv]

    @requires_lock
    def _get_updatable_server(self, model_id: str | None = None) -> RolloutServer | None:
        if model_id is not None:
            [srv] = self._get_servers_of_model_id(model_id)
            assert srv.update_weights, f"Server for model_id {model_id!r} is frozen (update_weights=False)"
            return srv

        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Pass model_id to update exactly one of them."
                )

    # -------------------------- eval fleet -----------------------------

    @lock_exempt
    async def get_eval_fleet_info(self) -> EvalFleetInfo | None:
        return self._eval_fleet.info if self._eval_fleet is not None else None

    @lock_exempt
    async def pin_eval_fleet(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        assert (
            self._eval_fleet is not None
        ), "this run deploys no eval fleet, so nothing asked this controller to pin one"
        return await self._eval_fleet.pin(checkpoint_dir=checkpoint_dir, weight_version=weight_version)

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {
            cell_id: cell.cell_status()
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
    async def check_weights(
        self,
        action: str,
        allow_quant_error: bool = False,
        selector: str = "all",
        skip_list: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[Any]:
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

    @with_lock
    async def _tick_cells(self) -> None:
        cells = [cell for srv in list(self.servers.values()) for cell in list(srv.server_cells.values())]
        results = await asyncio.gather(
            *[asyncio.wait_for(cell.tick(), timeout=CELL_TICK_TIMEOUT_SECONDS) for cell in cells],
            return_exceptions=True,
        )
        for cell, result in zip(cells, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Ticking cell {cell.meta.cell_id} failed", exc_info=result)

    # -------------------------- reconcile -----------------------------

    @with_lock
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        async def _add(_cell_id: str, observed_info: CellInfo) -> None:
            observed_cell_meta = _compute_server_cell_meta_from_info(observed_info)
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)

        async def _remove(remove_cell_id: str) -> None:
            await actual_srv.remove_cell(remove_cell_id)

        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual_cell.meta.workers_hash if actual_cell is not None else None,
            add=_add,
            remove=_remove,
        )

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self, model_id: str | None) -> None:
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(False)

    @requires_lock
    async def _health_monitoring_resume(self, model_id: str | None) -> None:
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(True)


@dataclass(frozen=True)
class UpdatableEngines:
    rollout_engines: list[SGLangApiClient]
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]
    # set only for a member publish: the membership epoch the member set belongs to
    membership_epoch: int | None = None


# TODO may move and generalize later
def _compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )
