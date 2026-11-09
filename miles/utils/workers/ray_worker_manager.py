from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.http_utils import wrap_ipv6
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.cell_operations.base import StaleFaultTargetError
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseCommandSpec,
    BaseServeSpec,
    BaseSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    WorkerCtorContext,
    WorkerLaunchContext,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"

_LIVENESS_SCAN_INTERVAL_SECONDS = 10.0


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(args, specs: list[BaseSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(args, specs, pgs, comm_backend=comm_backend))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(
        self, args, specs: list[BaseSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        configure_logger(args, source=SimpleProcessIdentity(component="worker_manager"))

        self.scaling = ScalingConfig.slice_from(args)
        self.port_allocator = PortAllocator(range_start=args.worker_dynamic_port_start)
        self.comm_backend = comm_backend
        self.pgs = pgs
        # views of the placement group created at startup; a re-pointed view must stay inside them
        self._startup_pgs = dict(pgs)
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)
        self._membership_lock = asyncio.Lock()

        self._validate_initial_bindings()
        # deferred cells are declared stopped: they wait for an explicit start_cells (after rebind_cell if unbound)
        await self.start_cells([c.cell_id for c in self._all_cells() if not c.deferred])

    async def start_cells(self, cell_ids: list[str]) -> None:
        async with self._membership_lock:
            cells = [cell for cell_id in cell_ids if (cell := self._find_cell(cell_id)).actors is None]
            unbound = [c.cell_id for c in cells if c.unbound]
            if unbound:
                raise CellUnboundError(f"deferred cells {unbound} are bound to no bundle; rebind_cell them first")
            for cell in cells:
                # bindings were checked free when made; check again, a view may have been re-pointed onto them since
                self._assert_bundles_free(cell.bundles(), ignore=cell)
            try:
                await _gather_or_raise([c.launch_actors() for c in cells])
                await _gather_or_raise([c.alloc_ports() for c in cells])
                await _gather_or_raise([c.post_setup() for c in cells])
            except Exception as error:
                logger.error(f"Starting cells {[c.cell_id for c in cells]} failed, rolling back", exc_info=True)
                results = await asyncio.gather(*[c.stop() for c in cells], return_exceptions=True)
                failed = {c.cell_id: r for c, r in zip(cells, results, strict=True) if isinstance(r, BaseException)}
                if failed:
                    raise StartRollbackFailedError(
                        f"starting cells {[c.cell_id for c in cells]} failed ({error!r}) and stopping "
                        f"{sorted(failed)} during the rollback failed too ({list(failed.values())!r}); they may "
                        f"still hold workers, stop them again"
                    ) from error
                raise

    async def stop_cells(self, cell_ids: list[str]) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[self._find_cell(cell_id).stop() for cell_id in cell_ids])

    async def shutdown(self) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for cell in self._all_cells()])

    # -------------------------- rebinding (elastic placement) -----------------------------
    # A stopped cell may move onto other bundles of the placement group created at startup; nothing here creates
    # bundles, so every target stays inside the startup placement.

    async def rebind_cell(self, cell_id: str, *, pg_name: str, pg_slot_offset: int) -> None:
        """Bind a stopped cell's workers to slots ``pg_slot_offset..`` of ``pgs[pg_name]`` for its next start."""
        async with self._membership_lock:
            cell = self._find_cell(cell_id)
            assert not cell.alive, f"cell {cell_id} is running; stop it before rebinding it"
            binding = _CellBinding(pg_name=pg_name, pg_slot_offset=pg_slot_offset)
            bundles = self._validate_binding(cell, binding)
            self._assert_bundles_free(bundles, ignore=cell)
            cell.binding = binding
            logger.info(f"Cell {cell_id} rebound to {pg_name}[{pg_slot_offset}:] (bundles {sorted(bundles)})")

    async def unbind_cell(self, cell_id: str) -> None:
        """Return a stopped deferred cell to the unbound state, so it holds no bundle (e.g. before a trainer grows)."""
        async with self._membership_lock:
            cell = self._find_cell(cell_id)
            assert cell.deferred, f"cell {cell_id} was not declared deferred; only deferred cells can be unbound"
            assert not cell.alive, f"cell {cell_id} is running; stop it before unbinding it"
            cell.binding = None
            logger.info(f"Cell {cell_id} unbound")

    async def set_pg_view(
        self, pg_name: str, info: PlacementGroupInfo, *, replacing_pools: list[str] | tuple[str, ...] = ()
    ) -> None:
        """Point ``pg_name`` at another slice of the startup placement group (e.g. GPUs a trainer released).

        Refused while a running cell uses the view, when a bundle lies outside the startup placement group or is
        held by a running cell, and when a stopped cell still bound to the view would reach outside it.
        """
        async with self._membership_lock:
            users = [c.cell_id for c in self._all_cells() if c.alive and c.pg_name == pg_name]
            assert not users, f"cells {users} run on {pg_name!r}; stop them before re-pointing it"
            known = {
                (pg_key(pgi.pg), b) for pgi in self._startup_pgs.values() for b in pgi.pg_reordered_bundle_indices
            }
            wanted = {(pg_key(info.pg), b) for b in info.pg_reordered_bundle_indices}
            outside = sorted(b for _, b in wanted - known)
            assert not outside, f"bundles {outside} are outside the placement group created at startup"
            assert len(wanted) == len(info.pg_reordered_bundle_indices), f"{pg_name!r} view repeats bundles"
            self._assert_bundles_free(wanted, ignore=None)
            num_slots = len(info.pg_reordered_bundle_indices)
            overflowing = {
                c.cell_id: slots
                for pool_id, pool in self._pools.items()
                if pool_id not in replacing_pools
                for c in pool.cells
                if c.pg_name == pg_name and (slots := [slot for slot in c.all_slots() if slot >= num_slots])
            }
            assert not overflowing, (
                f"stopped cells {overflowing} are bound to slots beyond the {num_slots} bundles of the new "
                f"{pg_name!r} view; rebind them or replace their pools (replacing_pools) first"
            )
            self.pgs = {**self.pgs, pg_name: info}

    def get_pg_view(self, pg_name: str) -> PlacementGroupInfo:
        return self.pgs[pg_name]

    async def stop_pools(self, pool_ids: list[str]) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for pool_id in pool_ids for cell in self._pools[pool_id].cells])

    async def start_pools(self, pool_ids: list[str]) -> None:
        """Start the stopped cells of the pools, except deferred cells, which only start through ``start_cells``."""
        await self.start_cells(
            [c.cell_id for pool_id in pool_ids for c in self._pools[pool_id].cells if not c.deferred]
        )

    async def replace_pool_spec(self, spec: BaseSpec) -> list[str]:
        """Swap the spec of a fully stopped pool (e.g. a trainer with a new GPU count); returns its new cell ids."""
        return (await self.replace_pools([spec]))[spec.name]

    async def replace_pools(
        self, specs: list[BaseSpec], *, scaling: ScalingConfig | None = None
    ) -> dict[str, list[str]]:
        """Swap the specs of fully stopped pools and optionally adopt a new ``scaling``; returns the new cell ids.

        A trainer resize changes both: its TrainerConfig (the spec) and the GPU counts its scheduling reads (the
        scaling). Pools not named here must schedule exactly as before under the new scaling. The new cells start one
        generation above the old ones, so no workers_hash repeats.
        """
        async with self._membership_lock:
            names = [spec.name for spec in specs]
            if unknown := sorted(set(names) - set(self._pools)):
                raise AssertionError(f"unknown pool {unknown}; known: {sorted(self._pools)}")
            if running := [c.cell_id for name in names for c in self._pools[name].cells if c.alive]:
                raise AssertionError(f"cells {running} of {names} are running; stop the pool first")
            new_scaling = scaling if scaling is not None else self.scaling
            changed = [
                name
                for name, pool in self._pools.items()
                if name not in names and pool.spec.scheduling(new_scaling) != pool.spec.scheduling(self.scaling)
            ]
            assert not changed, f"the new scaling also changes pools {changed}, which are not replaced"
            old_scaling, self.scaling = self.scaling, new_scaling
            try:
                new_pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
                for name, pool in new_pools.items():
                    generation = max((c.generation for c in self._pools[name].cells), default=0) + 1
                    for cell in pool.cells:
                        # a new spec is a new generation even before launch, so nothing mistakes it for the old one
                        cell.generation = generation
                        if cell.pg_name is not None and (slots := cell.all_slots()):
                            num_slots = len(self.pgs[cell.pg_name].pg_reordered_bundle_indices)
                            bad = [slot for slot in slots if not 0 <= slot < num_slots]
                            assert not bad, f"slots {bad} are outside {cell.pg_name!r}, which has {num_slots} bundles"
            except BaseException:
                self.scaling = old_scaling
                raise
            self._pools.update(new_pools)
            return {name: [c.cell_id for c in pool.cells] for name, pool in new_pools.items()}

    def get_cell_bundles(self, cell_id: str) -> list[int]:
        return sorted(b for _, b in self._find_cell(cell_id).bundles())

    def describe_cells(
        self, *, pool_ids: list[str] | None = None, category: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Every cell of the pools ``get_cell_infos`` would list, with its state and binding, read-only.

        ``alias`` is the caller's name for the cell (placement map ``rollout_cells``) or None; ``state`` is
        ``unbound`` (a deferred cell with no binding), ``stopped`` or ``running``; ``bundles`` / ``gpu_ids`` are the
        bundles of the startup placement group the cell uses or would use at its next start.
        """
        cell_ids = set(self.get_cell_infos(pool_ids=pool_ids, category=category))
        return {c.cell_id: c.describe() for c in self._all_cells() if c.cell_id in cell_ids}

    def _validate_initial_bindings(self) -> None:
        owner: dict[tuple[Any, int], str] = {}
        for cell in self._all_cells():
            if cell.binding is None:
                continue
            for bundle in self._validate_binding(cell, cell.binding):
                assert bundle not in owner, f"cells {owner[bundle]} and {cell.cell_id} are declared on bundle {bundle}"
                owner[bundle] = cell.cell_id

    def _validate_binding(self, cell: _CellManager, binding: _CellBinding) -> set[tuple[Any, int]]:
        scheduling = cell.spec.scheduling(self.scaling)
        assert scheduling.num_gpu_slots_per_worker > 0, f"{cell.spec.name!r} owns no GPU slots to rebind"
        assert (
            binding.pg_name in self.pgs
        ), f"unknown placement group view {binding.pg_name!r}; known: {sorted(self.pgs)}"
        probe = dataclasses.replace(cell, binding=binding, actors=None)
        num_slots = len(self.pgs[binding.pg_name].pg_reordered_bundle_indices)
        bad = [slot for slot in probe.all_slots() if not 0 <= slot < num_slots]
        assert not bad, f"slots {bad} are outside {binding.pg_name!r}, which has {num_slots} bundles"
        return probe.bundles()

    def _assert_bundles_free(self, bundles: set[tuple[Any, int]], *, ignore: _CellManager | None) -> None:
        for other in self._all_cells():
            if other is ignore or not other.alive:
                continue
            if overlap := bundles & other.bundles():
                raise BundleInUseError(
                    f"bundles {sorted(b for _, b in overlap)} are in use by running cell {other.cell_id}"
                )

    def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        cell = self._find_cell(cell_id)
        if not cell.alive or not 0 <= rank < len(cell.actors):
            raise StaleFaultTargetError(f"Cell {cell_id} has no live worker at index {rank}")
        return ObservedFaultHookTarget(cell_id=cell_id, rank=rank, workers_hash=cell.get_info().workers_hash)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        target = command.request.target
        assert isinstance(target, ObservedFaultHookTarget), "A fault hook sent to a cell names the worker it observed"
        if self.observe_fault_target(cell_id=target.cell_id, rank=target.rank) != target:
            raise StaleFaultTargetError("Fault hook target no longer matches the observed worker")
        actor = self._find_cell(target.cell_id).actors[target.rank].actor_handle
        return await actor.control_fault_hook.remote(command=command)

    def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._find_actor(worker_name).self_addrs
        assert addrs is not None, (
            f"{worker_name} has not been given its ports yet; a caller reading them now would take the "
            f"endpoints it cannot find for endpoints the worker does not have"
        )
        return addrs

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        return {
            name: [a.self_addrs or {} for c in g.cells if c.alive for a in c.actors] for name, g in self._pools.items()
        }

    def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        cell = self._find_cell(cell_id)
        return [self._compute_worker_info(actor) for actor in (cell.actors if cell.actors is not None else [])]

    def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None = None) -> dict[str, CellInfo]:
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        if pool_ids is None:
            pool_ids = [
                name
                for name, pool in self._pools.items()
                if pool.spec.scheduling(self.scaling).declares_dynamic_pool()
            ]
        unknown = set(pool_ids) - set(self._pools)
        assert not unknown, f"{unknown=} {sorted(self._pools)=}"
        infos = [
            c.get_info()
            for name in pool_ids
            if category is None or self._pools[name].spec.category == category
            for c in self._pools[name].cells
        ]
        return {info.cell_id: info for info in infos}

    def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> ray.actor.ActorHandle:
        actor = self._find_actor(worker_name)
        assert actor.generation == expected_generation, (
            f"{worker_name} is now generation {actor.generation}, not the {expected_generation} it was described as; "
            f"ask for its worker infos again"
        )
        return actor.actor_handle

    def _compute_worker_info(self, actor: _BaseActorManager) -> WorkerInfo:
        served_over_rpc = isinstance(actor.spec, BaseServeSpec) and self.comm_backend == WorkerCommBackend.RPC
        return WorkerInfo(
            name=actor.name,
            generation=actor.generation,
            self_addrs=actor.self_addrs or {},
            gpu_ids=actor.gpu_ids,
            worker_class=actor.spec.worker_class if served_over_rpc else None,
        )

    def _find_actor(self, worker_name: str) -> _BaseActorManager:
        matches = [a for c in self._all_cells() if c.alive for a in c.actors if a.name == worker_name]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    def _find_cell(self, cell_id: str) -> _CellManager:
        matches = [c for c in self._all_cells() if c.cell_id == cell_id]
        assert len(matches) == 1, f"{cell_id=} {matches=}"
        return matches[0]

    def _all_cells(self) -> list[_CellManager]:
        return [c for g in self._pools.values() for c in g.cells]


@dataclass(kw_only=True)
class _PoolManager:
    spec: BaseSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseSpec, manager: RayWorkerManager) -> _PoolManager:
        scheduling = spec.scheduling(manager.scaling)
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                    deferred=cell_index >= scheduling.num_cells,
                    binding=(
                        _CellBinding(*b)
                        if cell_index < len(scheduling.initial_bindings)
                        and (b := scheduling.initial_bindings[cell_index]) is not None
                        else None
                    ),
                )
                for cell_index in range(scheduling.num_cells + scheduling.num_deferred_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseSpec)


class BundleInUseError(RuntimeError):
    pass


class CellUnboundError(RuntimeError):
    """A deferred cell was asked to start before ``rebind_cell`` bound it to bundles."""


class StartRollbackFailedError(RuntimeError):
    """A start failed and the rollback that should have stopped its cells failed as well."""


def pg_key(pg: Any) -> Any:
    # key bundles by the placement group id, so two handles of the same group compare equal
    return pg.id if isinstance(pg, PlacementGroup) else pg


@dataclass(frozen=True)
class _CellBinding:
    pg_name: str
    pg_slot_offset: int


def _actor_manager_cls(spec: BaseSpec, *, comm_backend: WorkerCommBackend) -> type[_BaseActorManager]:
    match spec, comm_backend:
        case BaseCommandSpec(), _:
            return _CommandActorManager
        case BaseServeSpec(), WorkerCommBackend.RPC:
            return _ServeActorRpcCommManager
        case BaseServeSpec(), WorkerCommBackend.RAY:
            return _ServeActorRayCommManager
    raise AssertionError(f"{spec.name} is neither served nor launched as a command")


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    manager: RayWorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None
    generation: int = 0
    liveness_scan_task: asyncio.Task | None = None
    # set by RayWorkerManager.rebind_cell or the placement map; None keeps the spec's own slot layout
    binding: _CellBinding | None = None
    # declared beyond the scheduling's num_cells: not started at init, and only a binding makes it startable
    deferred: bool = False

    @property
    def alias(self) -> str | None:
        aliases = self.spec.scheduling(self.manager.scaling).cell_aliases
        return aliases[self.cell_index] if self.cell_index < len(aliases) else None

    @property
    def unbound(self) -> bool:
        return self.deferred and self.binding is None

    @property
    def pg_name(self) -> str | None:
        if self.unbound:
            return None
        return self.binding.pg_name if self.binding is not None else self.spec.scheduling(self.manager.scaling).pg_name

    def slot_of(self, worker_in_cell_index: int) -> int | None:
        scheduling = self.spec.scheduling(self.manager.scaling)
        if self.pg_name is None:
            return None
        if (binding := self.binding) is not None:
            return binding.pg_slot_offset + worker_in_cell_index * scheduling.num_gpu_slots_per_worker
        return (
            scheduling.pg_slot_offset
            + (self.cell_index * scheduling.num_workers_per_cell + worker_in_cell_index)
            * scheduling.num_gpu_slots_per_worker
        )

    def all_slots(self) -> list[int]:
        scheduling = self.spec.scheduling(self.manager.scaling)
        if self.pg_name is None or scheduling.num_gpu_slots_per_worker == 0:
            return []
        return [
            self.slot_of(w) + k
            for w in range(scheduling.num_workers_per_cell)
            for k in range(scheduling.num_gpu_slots_per_worker)
        ]

    def bundles(self) -> set[tuple[Any, int]]:
        if not (slots := self.all_slots()):
            return set()
        pg = self.manager.pgs[self.pg_name]
        return {(pg_key(pg.pg), pg.pg_reordered_bundle_indices[slot]) for slot in slots}

    def describe(self) -> dict[str, Any]:
        bundles: list[int] | None = []
        gpu_ids: list[int] | None = []
        if slots := self.all_slots():
            pg = self.manager.pgs[self.pg_name]
            if all(0 <= slot < len(pg.pg_reordered_bundle_indices) for slot in slots):
                bundles = [pg.pg_reordered_bundle_indices[slot] for slot in slots]
                gpu_ids = [pg.pg_reordered_gpu_ids[slot] for slot in slots]
            else:  # a stopped cell left outside a re-pointed view
                bundles = gpu_ids = None
        return dict(
            cell_id=self.cell_id,
            alias=self.alias,
            pool_id=self.spec.name,
            deferred=self.deferred,
            state="unbound" if self.unbound else ("running" if self.alive else "stopped"),
            generation=self.generation,
            pg_name=self.pg_name,
            pg_slot_offset=self.binding.pg_slot_offset if self.binding is not None else None,
            bundles=bundles,
            gpu_ids=gpu_ids,
        )

    async def launch_actors(self):
        assert self.actors is None
        self.generation += 1
        scheduling = self.spec.scheduling(self.manager.scaling)
        actor_manager_cls = _actor_manager_cls(self.spec, comm_backend=self.manager.comm_backend)
        self.actors = [
            actor_manager_cls(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
                gpu_slot_index=self.slot_of(worker_in_cell_index),
            )
            for worker_in_cell_index in range(scheduling.num_workers_per_cell)
        ]
        await self._for_all_actors(lambda a: a.launch_actor())
        self.liveness_scan_task = asyncio.create_task(self._scan_liveness_forever(self.generation))

    async def alloc_ports(self) -> None:
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        await self._for_all_actors(lambda a: a.post_setup())

    async def stop(self) -> None:
        if self.actors is None:
            return
        await self._for_all_actors(lambda a: a.stop())
        self.actors = None

    async def _scan_liveness_forever(self, generation: int) -> None:
        while self.generation == generation and self.actors is not None:
            await asyncio.sleep(_LIVENESS_SCAN_INTERVAL_SECONDS)
            try:
                await self._scan_liveness_once()
            except Exception:
                logger.error(f"Scanning liveness of cell {self.cell_id} failed, will scan again", exc_info=True)

    async def _scan_liveness_once(self) -> None:
        generation = self.generation
        dead_worker_names = await self._find_dead_worker_names()
        if not dead_worker_names:
            return

        async with self.manager._membership_lock:
            if self.actors is None or self.generation != generation:
                return
            logger.error(
                f"Cell {self.cell_id} lost workers {dead_worker_names} without being stopped, "
                f"so the whole cell is torn down and reported as not alive"
            )
            await self.stop()

    async def _find_dead_worker_names(self) -> list[str]:
        if (actors := self.actors) is None:
            return []
        probes = await asyncio.gather(*[a.probe_is_dead() for a in actors])
        return [actor.name for actor, is_dead in zip(actors, probes, strict=True) if is_dead]

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            pool_id=self.spec.name,
            alive=self.alive and self._all_workers_have_addrs,
            worker_names=[a.name for a in self.actors] if self.actors is not None else [],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=self.spec.compute_static_meta(self.manager.scaling).resolve(cell_index=self.cell_index),
        )

    @property
    def cell_id(self) -> str:
        return compute_cell_id(pool_id=self.spec.name, cell_index=self.cell_index)

    @property
    def alive(self) -> bool:
        return self.actors is not None

    @property
    def _all_workers_have_addrs(self) -> bool:
        return all(a.self_addrs is not None for a in self.actors or [])


_SHUTDOWN_TIMEOUT = 30


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    local_gpu_ids: list[int] | None = None
    gpu_slot_index: int | None

    async def launch_actor(self) -> None:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        allocated: NamedHostAndPorts = {}

        node_ip, external_ip, self.local_gpu_ids = await asyncio.gather(
            self.actor_handle._get_node_ip.remote(),
            self.actor_handle._get_node_external_ip.remote(),
            self.actor_handle._to_local_gpu_ids.remote(gpu_ids=self.gpu_ids),
        )
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            if port_info.allow_dynamic:
                # The master port is bound much later (the cell's init_process_group);
                # the worker holds it until then so nothing else on the node takes it.
                port = await self.manager.port_allocator.alloc(
                    self.actor_handle,
                    node_ip=node_ip,
                    consecutive=port_info.num_consecutive,
                    hold=port_info.mode == "master",
                )
            else:
                port = port_info.static_port + (self.parent.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            allocated[port_info.name] = HostAndPort(
                host=wrap_ipv6(node_ip),
                port=port,
                external_host=wrap_ipv6(external_ip) if external_ip else None,
            )

        self.self_addrs = allocated

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        free = await self.actor_handle._is_port_available.remote(port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    @property
    def launch_context(self) -> WorkerLaunchContext:
        return WorkerLaunchContext(
            args=self.spec.args,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            num_workers_per_cell=self.spec.scheduling(self.manager.scaling).num_workers_per_cell,
            gpu_ids=self.gpu_ids,
        )

    def _compute_remote_options(self) -> dict:
        return {}

    def _create_actor(self, actor_class: type, **ctor_kwargs) -> ray.actor.ActorHandle:
        scheduling = self.spec.scheduling(self.manager.scaling)
        scheduling_strategy = None
        if (pg_name := self.parent.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        remote_options = self._compute_remote_options()
        remote_class = ray.remote(**remote_options)(actor_class) if remote_options else ray.remote(actor_class)

        return remote_class.options(
            num_cpus=scheduling.num_cpus_per_worker,
            num_gpus=scheduling.num_gpus_per_worker,
            **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
            runtime_env={"env_vars": self.spec.env_var(self.launch_context)},
            **(compute_ray_pin_head_options() if scheduling.pin_to_head else {}),
        ).remote(**ctor_kwargs)

    async def probe_is_dead(self) -> bool:
        if self.actor_handle is None:
            return False
        return await RayWorkerHandle(self.actor_handle).probe_is_dead()

    async def stop(self) -> None:
        if self.actor_handle is None:
            return

        await self._shutdown_gracefully()

        try:
            ray.kill(self.actor_handle)
            logger.info(f"Killed actor at {self=}")
        except Exception as e:
            logger.warning(f"Failed to kill actor at {self=} ({e})")

    async def _shutdown_gracefully(self) -> None:
        pass

    @property
    def name(self) -> str:
        return compute_worker_name(
            pool_id=self.spec.name,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def generation(self) -> int:
        return self.parent.generation

    @property
    def gpu_ids(self) -> list[int]:
        if (pg_name := self.parent.pg_name) is None:
            return []
        pg = self.manager.pgs[pg_name]
        base_gpu_id = int(pg.pg_reordered_gpu_ids[self.gpu_slot_index])
        scheduling = self.spec.scheduling(self.manager.scaling)
        return list(range(base_gpu_id, base_gpu_id + scheduling.num_gpu_slots_per_worker))

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[BaseCommandSpec]):
    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(CommandActor)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            **dict(self.launch_context),
            scaling=self.manager.scaling,
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            pool_addrs=self.manager.get_addrs(),
            local_gpu_ids=self.local_gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        # exec lets the command replace /bin/sh, so parent death and Ray's child cleanup kill the
        # command itself rather than a shell that would leave it orphaned with its ports.
        self.actor_handle.run.remote(cmd=f"exec {launch_cmd}", envs={})

    async def _shutdown_gracefully(self) -> None:
        try:
            await asyncio.wait_for(self.actor_handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")


@dataclass
class _ServeActorRayCommManager(_BaseActorManager[BaseServeSpec]):
    def _compute_remote_options(self) -> dict:
        groups = self.spec.concurrency_groups
        return {} if groups is None else dict(concurrency_groups=groups)

    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(
            self._compute_actor_class(),
            ctor_kwargs=self.spec.ctor_kwargs,
            context=self.launch_context,
        )

    def _compute_actor_class(self) -> type:
        actor_class = bootstrapped_worker_class(self.spec.worker_class)
        method_groups = self._compute_method_concurrency_groups(actor_class)
        if not method_groups:
            return actor_class
        return type(
            f"{actor_class.__name__}WithConcurrencyGroups",
            (actor_class,),
            {
                name: _route_method_to_concurrency_group(
                    getattr(actor_class, name), group=group
                )  # config-access-exempt: attribute selected at runtime from name
                for name, group in method_groups.items()
            },
        )

    def _compute_method_concurrency_groups(self, actor_class: type) -> dict[str, str]:
        if (groups := self.spec.concurrency_groups) is None:
            return {}

        method_groups = declared_concurrency_groups(actor_class)
        assert method_groups, (
            f"Worker {self.spec.name!r} declares concurrency groups {sorted(groups)} but no method of "
            f"{actor_class.__name__} is annotated with @rpc(concurrency_group=...): threading the actor "
            f"while every method stays in the default group buys nothing"
        )
        undeclared = sorted(set(method_groups.values()) - set(groups))
        assert not undeclared, f"Worker {self.spec.name!r} routes methods to undeclared groups: {undeclared}"
        return method_groups

    async def post_setup(self) -> None:
        pass


def _route_method_to_concurrency_group(method: Callable, *, group: str) -> Callable:
    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    routed.__signature__ = inspect.signature(method)
    del routed.__wrapped__
    return ray.method(concurrency_group=group)(routed)


@dataclass
class _ServeActorRpcCommManager(_BaseActorManager[BaseServeSpec]):
    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(
            ServeActor,
            build_worker=partial(
                _build_serve_worker,
                worker_class_path=self.spec.worker_class,
                ctor_kwargs=self.spec.ctor_kwargs,
                context=self.launch_context,
            ),
        )

    async def post_setup(self) -> None:
        await self.actor_handle.start_rpc_server.remote(port=self.self_addrs[RPC_PORT_NAME].port)


def _build_serve_worker(
    *, worker_class_path: str, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
) -> Any:
    return bootstrapped_worker_class(worker_class_path)(ctor_kwargs=ctor_kwargs, context=context)


@functools.cache
def bootstrapped_worker_class(worker_class_path: str) -> type:
    worker_class = load_function(worker_class_path)

    class BootstrappedWorker(worker_class, NodeProbeMixin):
        def __init__(
            self, *, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
        ) -> None:
            super().__init__(**ctor_kwargs(_ctor_context(context)))

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def _ctor_context(launch_context: WorkerLaunchContext) -> WorkerCtorContext:
    return WorkerCtorContext(
        args=launch_context.args,
        cell_index=launch_context.cell_index,
        worker_in_cell_index=launch_context.worker_in_cell_index,
        num_workers_per_cell=launch_context.num_workers_per_cell,
        gpu_ids=launch_context.gpu_ids,
        capability=DeferredBackendCapability(create=_create_ray_backend_capability),
    )


def _create_ray_backend_capability() -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


async def _gather_or_raise(coros: list[Coroutine[Any, Any, None]]) -> None:
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
