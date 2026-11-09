"""Explicit, epoch-guarded membership of the inference engine cells (elastic rollout).

``InferenceMembershipMixin`` is a base of ``InferenceController``: the caller starts and stops cells the worker
manager declared at startup, drains them through the router, admits cells it published cordoned, and commits the
executor's weight version once every serving cell carries it. Every change holds the controller lock and commits a
new membership epoch, so a member publish planned for an epoch cannot race a membership change.

Membership state lives in this process only; a caller with a durable journal restores it after a restart with
``restore_membership_state``.
"""

import asyncio
import logging
import time
from typing import Any, Protocol, runtime_checkable

from miles.ray.rollout.server_cell import ServerCell
from miles.utils.context_lock import (
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)

logger = logging.getLogger(__name__)

DRAIN_POLL_INTERVAL_SECONDS = 0.5
TRACK_POLL_INTERVAL_SECONDS = 2.0
TRACK_TIMEOUT_SECONDS = 3600.0
_MEMBERSHIP_OPS = ("start", "stop")


class MembershipEpochMismatchError(RuntimeError):
    pass


class MembershipIncompleteError(RuntimeError):
    """A membership change failed half way; only a retry of that same change is accepted until it succeeds."""


class CellsNotTrackedError(KeyError):
    """Cells were started but no server has taken them in yet; call ``wait_cells_tracked`` before publishing."""


@runtime_checkable
class MembershipWorkerProvider(Protocol):
    """An engine provider that starts and stops the cells it declared at startup on demand."""

    async def start_cells(self, *, cell_ids: list[str]) -> None: ...

    async def stop_cells(self, *, cell_ids: list[str]) -> None: ...

    async def describe_declared_cells(self) -> dict[str, dict[str, Any]]: ...


def _is_start_rollback_failure(error: BaseException) -> bool:
    """Whether a start failed *and* its rollback failed (``ray_worker_manager.StartRollbackFailedError``).

    Matched by class name on the error and its ``__cause__``: a Ray task error is raised as a subclass of the remote
    error type, and an RPC failure keeps the remote type (#3857), so this module does not import the worker manager.
    """
    candidates = [error, *([error.__cause__] if error.__cause__ is not None else [])]
    return any("StartRollbackFailedError" in {cls.__name__ for cls in type(c).__mro__} for c in candidates)


@enforce_lock_discipline
class InferenceMembershipMixin:
    # class-level defaults so a controller built without __init__ (tests) still has a membership state
    _membership_epoch: int = 0
    _last_membership_op: tuple[str, tuple[str, ...]] | None = None
    _membership_incomplete: tuple[str, tuple[str, ...]] | None = None
    # why membership is incomplete: "stop_failed", "start_rollback_failed" or "restored"
    _membership_incomplete_reason: str | None = None
    # set by start_update_weights(admit_cordoned=True) for the end_update_weights of that same publish
    _admit_cordoned: bool = False
    # cell id -> cell registered cordoned by an admit_cordoned publish and not admitted yet
    _awaiting_admission: dict[str, ServerCell] = {}
    # cells a drain_cells call waits on; one that becomes Serving meanwhile is registered cordoned
    _draining_cell_ids: frozenset[str] = frozenset()

    # -------------------------- status -----------------------------

    @lock_exempt
    async def get_membership_status(self) -> dict[str, Any]:
        """The epoch plus the change that failed half way (``[op, cell_ids]``) and must be retried, if any."""
        incomplete = self._membership_incomplete
        return dict(
            epoch=self._membership_epoch,
            incomplete=None if incomplete is None else [incomplete[0], list(incomplete[1])],
            incomplete_reason=None if incomplete is None else self._membership_incomplete_reason,
            retry_advances_epoch=incomplete is not None,
        )

    @with_lock
    async def describe_cells(self) -> dict[str, dict[str, Any]]:
        """Every declared engine cell with its worker-manager state and binding, plus how this controller sees it."""
        cells = await self._membership_provider().describe_declared_cells()
        for cell_id, desc in cells.items():
            tracked = self._find_cell_or_none(cell_id)
            desc["tracked"] = tracked is not None
            desc["serving"] = tracked is not None and tracked.is_serving
            desc["awaiting_admission"] = tracked is not None and self._is_awaiting_admission(tracked)
        return cells

    @with_lock
    async def restore_membership_state(
        self,
        *,
        epoch: int,
        incomplete: list | None = None,
        last_op: list | None = None,
        expected_current_epoch: int,
    ) -> dict[str, Any]:
        """Compare-and-set the membership epoch and half-failed state from the caller's durable journal."""
        if expected_current_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"restore expected membership epoch {expected_current_epoch}, but it is {self._membership_epoch}"
            )
        if epoch < 0:
            raise ValueError(f"membership epoch must be >= 0, got {epoch}")
        self._membership_incomplete = _parse_op(incomplete, "incomplete")
        self._membership_incomplete_reason = None if incomplete is None else "restored"
        self._last_membership_op = _parse_op(last_op, "last_op")
        self._membership_epoch = epoch
        logger.info(
            f"membership restored by the caller: epoch={epoch} incomplete={self._membership_incomplete} "
            f"last_op={self._last_membership_op}"
        )
        return await self.get_membership_status()

    # -------------------------- start / stop -----------------------------

    @with_lock
    async def wait_cells_tracked(
        self,
        cell_ids: list[str],
        *,
        timeout_seconds: float = TRACK_TIMEOUT_SECONDS,
        poll_interval_seconds: float = TRACK_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Wait until every cell was taken in by a server and is PendingWeights or Serving."""
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
    async def start_cells(self, cell_ids: list[str], *, expected_epoch: int) -> int:
        """Start pre-declared engine cells and return the new membership epoch.

        The cells join through the provider watch (reconcile -> PendingWeights) and take no traffic until weights are
        published to them; call ``wait_cells_tracked`` before a member publish. Repeating the call that produced the
        current epoch returns it without acting again; any other epoch mismatch is refused. If the start fails and
        the worker manager's rollback fails too, only a stop of those cells is accepted next.
        """
        return await self._change_membership("start", cell_ids, expected_epoch=expected_epoch)

    @with_lock
    async def stop_cells(self, cell_ids: list[str], *, expected_epoch: int) -> int:
        """Remove pre-declared engine cells from their servers (router deregistration first) and stop them."""
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
        provider = self._membership_provider()
        declared = set(await provider.describe_declared_cells())
        if unknown := sorted(set(ids) - declared):
            raise KeyError(f"cells {unknown} were not declared at startup; declared cells are {sorted(declared)}")

        if op == "stop":
            # Deregistering first keeps traffic off cells that are going away. If stopping then fails, the cells are
            # out of routing but may still run; the epoch stays and only this same stop is accepted until it succeeds.
            try:
                for srv in self.servers.values():
                    for cell_id in ids:
                        if cell_id in srv.all_server_cells:
                            await srv.remove_cell(cell_id)
                await provider.stop_cells(cell_ids=list(ids))
            except BaseException:
                self._membership_incomplete, self._membership_incomplete_reason = (op, ids), "stop_failed"
                logger.error(f"stop_cells {list(ids)} failed half way at epoch {current}; retry the same stop")
                raise
            self._awaiting_admission = {k: v for k, v in self._awaiting_admission.items() if k not in ids}
        else:
            try:
                await provider.start_cells(cell_ids=list(ids))
            except BaseException as error:
                if _is_start_rollback_failure(error):
                    self._membership_incomplete = ("stop", ids)
                    self._membership_incomplete_reason = "start_rollback_failed"
                    logger.error(f"start_cells {list(ids)} failed and its rollback failed; stop these cells first")
                raise

        self._membership_incomplete = self._membership_incomplete_reason = None
        self._membership_epoch = current + 1
        self._last_membership_op = (op, ids)
        logger.info(f"{op}_cells {list(ids)} committed membership epoch {self._membership_epoch}")
        return self._membership_epoch

    # -------------------------- cordon / drain / admit -----------------------------

    @with_lock
    async def uncordon_cells(self, cell_ids: list[str]) -> None:
        """Undo a cordon (e.g. cancel a drain). Cells awaiting admission are refused: only ``admit_cells`` opens them."""
        cells = self._find_cells(cell_ids)
        if waiting := [cell.meta.cell_id for cell in cells if self._is_awaiting_admission(cell)]:
            raise RuntimeError(f"cells {waiting} await admission after a cordoned publish; use admit_cells")
        await asyncio.gather(*[cell.uncordon() for cell in cells])

    @with_lock
    async def admit_cells(
        self, cell_ids: list[str], *, expected_epoch: int, expected_weight_version: str | int
    ) -> None:
        """Let cells published with ``admit_cordoned=True`` take traffic, after the caller verified their weights."""
        self._check_membership_epoch(expected_epoch, what=f"admit {cell_ids}")
        cells = self._find_cells(cell_ids)
        if not_serving := [cell.meta.cell_id for cell in cells if not cell.is_serving]:
            raise RuntimeError(f"cells {not_serving} are not serving (weights not published yet); nothing to admit")
        if not_waiting := [cell.meta.cell_id for cell in cells if not self._is_awaiting_admission(cell)]:
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

        False when the deadline passes first; nothing is aborted and the cells stay cordoned (``uncordon_cells`` to
        cancel). The lock is released while waiting. A target cell that becomes Serving meanwhile is registered
        cordoned by ``end_update_weights`` and cordoned here before it is counted.
        """
        cells = self._find_cells(cell_ids)
        cordoned: set[int] = set()

        async def _cordon_serving(targets: list[ServerCell]) -> None:
            fresh = [cell for cell in targets if cell.is_serving and id(cell) not in cordoned]
            await asyncio.gather(*[cell.cordon() for cell in fresh])
            cordoned.update(id(cell) for cell in fresh)

        await _cordon_serving(cells)
        deadline = time.monotonic() + timeout_seconds
        self._draining_cell_ids = self._draining_cell_ids | frozenset(cell_ids)
        try:
            while True:
                remaining = [c for c in map(self._find_cell_or_none, cell_ids) if c is not None and c.is_serving]
                await _cordon_serving(remaining)
                counts = await asyncio.gather(*[cell.get_inflight() for cell in remaining])
                busy = {cell.meta.cell_id: n for cell, n in zip(remaining, counts, strict=True) if n > 0}
                if not busy:
                    return True
                if time.monotonic() >= deadline:
                    logger.warning(f"drain of {cell_ids} timed out after {timeout_seconds}s; still in flight: {busy}")
                    return False
                async with self.context_lock.with_released():
                    await asyncio.sleep(poll_interval_seconds)
        finally:
            self._draining_cell_ids = self._draining_cell_ids - frozenset(cell_ids)

    # -------------------------- weight version commit -----------------------------

    @acquires_lock
    async def start_commit_weight_version(
        self, *, weight_version: int, expected_epoch: int, model_id: str | None = None
    ) -> dict[str, str]:
        """Check, holding the lock until ``end_commit_weight_version``, that every Serving cell reports the version.

        While the lock is held no membership change, publish or admission can run, so what is checked here still
        holds when the caller sets the executor's weight version. Returns cell id -> version.
        """
        self._check_membership_epoch(expected_epoch, what="weight version commit")
        versions = await self._cells_weight_versions(model_id)
        if not versions:
            raise RuntimeError("no serving engine reports a weight version; nothing to commit")
        if stale := {cell_id: v for cell_id, v in versions.items() if v != str(weight_version)}:
            raise RuntimeError(f"serving engines {stale} do not report weight version {weight_version} yet")
        return versions

    @releases_lock
    async def end_commit_weight_version(self) -> None:
        pass

    # -------------------------- helpers used by the publish path -----------------------------

    @requires_lock
    def _validate_member_publish(self, *, srv, members: list[str], expected_epoch: int | None) -> None:
        if expected_epoch is None:
            raise ValueError("a member publish needs the membership expected_epoch it was planned for")
        self._check_membership_epoch(expected_epoch, what=f"publish to {members}")
        if unknown := sorted(set(members) - set(srv.normal_server_cells)):
            raise CellsNotTrackedError(
                f"members {unknown} are not serviceable cells of {srv.model_name} (yet); its cells are "
                f"{sorted(srv.normal_server_cells)}. Call wait_cells_tracked after start_cells before publishing"
            )

    @requires_lock
    def _registers_cordoned(self, cell_id: str) -> bool:
        return self._admit_cordoned or cell_id in self._draining_cell_ids

    @requires_lock
    def _check_membership_epoch(self, expected_epoch: int, *, what: str) -> None:
        if expected_epoch != self._membership_epoch:
            raise MembershipEpochMismatchError(
                f"{what} expected membership epoch {expected_epoch}, but it is {self._membership_epoch}"
            )
        if (incomplete := self._membership_incomplete) is not None:
            raise MembershipIncompleteError(
                f"{incomplete[0]}_cells {list(incomplete[1])} failed half way; retry it first"
            )

    @requires_lock
    async def _cells_weight_versions(self, model_id: str | None) -> dict[str, str]:
        if (srv := self._get_updatable_server(model_id=model_id)) is None:
            return {}
        serving = [(cell_id, cell) for cell_id, cell in srv.all_server_cells.items() if cell.is_serving]
        versions = await asyncio.gather(*[cell.api_client.get_weight_version() for _, cell in serving])
        return {cell_id: str(version) for (cell_id, _), version in zip(serving, versions, strict=True)}

    @requires_lock
    def _is_awaiting_admission(self, cell: ServerCell) -> bool:
        return self._awaiting_admission.get(cell.meta.cell_id) is cell

    @requires_lock
    def _find_cell_or_none(self, cell_id: str) -> ServerCell | None:
        for srv in self.servers.values():
            if (cell := srv.all_server_cells.get(cell_id)) is not None:
                return cell
        return None

    @requires_lock
    def _find_cells(self, cell_ids: list[str]) -> list[ServerCell]:
        cells = [self._find_cell_or_none(cell_id) for cell_id in cell_ids]
        if unknown := [cell_id for cell_id, cell in zip(cell_ids, cells, strict=True) if cell is None]:
            raise KeyError(f"cells {unknown} are not tracked by this controller")
        return cells

    @lock_exempt
    def _membership_provider(self) -> MembershipWorkerProvider:
        provider = self._engine_provider
        if not isinstance(provider, MembershipWorkerProvider):
            raise NotImplementedError(f"{type(provider).__name__} cannot start or stop declared cells on demand")
        return provider


def _parse_op(value, what: str) -> tuple[str, tuple[str, ...]] | None:
    if value is None:
        return None
    op, cell_ids = value
    if op not in _MEMBERSHIP_OPS or not cell_ids:
        raise ValueError(f'{what} must be ["start"|"stop", non-empty cell ids], got {value!r}')
    return op, tuple(sorted(set(cell_ids)))
