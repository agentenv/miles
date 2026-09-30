import asyncio
import json
import logging
import socket
from collections.abc import Sequence
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig, compute_trainer_args
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_router_providers,
    create_inference_controller_handle,
)
from miles.ray.specs.rollout import create_rollout_executor_handle
from miles.ray.specs.train import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    compute_trainer_configs,
    create_trainer_controller_handle,
    external_trainer_controller_addrs,
    specs_trainer,
    specs_trainer_controller,
)
from miles.ray.wiring import get_backend_capability
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.hot_restart import (
    init_or_reset_inference_controller,
    trainer_init_or_load_state,
    wait_trainers_idle,
    wait_until_worker_not_initialized,
)
from miles.utils.test_utils.ft_test_actions import FTTestActionOrchestrationExecutor
from miles.utils.workers.types import DeployComponent, DeploymentIdentity
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.static import wait_static_addrs_ready

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


class PlacementGroupInfo(NamedTuple):
    pg: PlacementGroup
    pg_reordered_bundle_indices: list[int]
    pg_reordered_gpu_ids: list[int]


def _create_placement_group(num_gpus) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return PlacementGroupInfo(pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids)


def _get_placement_group_layout(args) -> tuple[int, int]:
    selector = DeployComponent(args.deploy_component)
    trainer_num_gpus = _compute_trainer_num_gpus(args) if selector.selects(DeployComponent.TRAINER) else 0
    if args.debug_train_only:
        eval_num_gpus = args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
        return trainer_num_gpus + eval_num_gpus, trainer_num_gpus

    rollout_num_gpus = (
        (args.rollout_num_gpus or 0) + args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
    )
    if args.rollout_external:
        return (0, 0) if args.debug_rollout_only else (trainer_num_gpus, trainer_num_gpus)
    if args.debug_rollout_only:
        return rollout_num_gpus, 0
    if args.colocate:
        return max(trainer_num_gpus, rollout_num_gpus), 0
    return trainer_num_gpus + rollout_num_gpus, trainer_num_gpus


def _compute_trainer_num_gpus(args) -> int:
    num_policies = len([config for config in compute_trainer_configs(args) if config.role == ACTOR_ROLE])
    return args.actor_num_nodes * args.actor_num_gpus_per_node * num_policies


STANDBY_PG_NAME = "standby"
_PLACEMENT_MAP_ROLES = ("trainer", "rollout", "standby")
# Not a role: an optional explicit list of the rollout engine cells, see ``RolloutCellDecl``.
ROLLOUT_CELLS_KEY = "rollout_cells"
_ROLE_VIEW_NAMES = {"rollout": "rollout", STANDBY_PG_NAME: STANDBY_PG_NAME}


class RolloutCellDecl(NamedTuple):
    """One rollout engine cell declared by the caller (yeto) in the placement map.

    ``name`` is the caller's id for the cell; the fork keeps its own cell id (``compute_cell_id(pool, index)`` with
    ``index`` the position in ``rollout_cells``) and reports both (``RayWorkerManager.describe_cells``).
    ``bundles`` are logical bundle positions, a consecutive run of the ``rollout`` or ``standby`` role, or empty for
    an unbound cell. ``start`` False declares the cell stopped: it is not started at startup, takes no GPU, no
    traffic and no weights until the caller starts it (after ``rebind_cell`` if it is unbound).
    """

    name: str
    bundles: tuple[int, ...]
    start: bool = True


class PlacementMap(NamedTuple):
    """Explicit role -> logical bundle positions (indices into the (node, gpu) sorted bundle order)."""

    trainer: tuple[int, ...]
    rollout: tuple[int, ...]
    standby: tuple[int, ...] = ()
    # explicit rollout engine cells (empty: the engine pool lays its cells out over the rollout role as before)
    rollout_cells: tuple[RolloutCellDecl, ...] = ()

    @property
    def num_bundles(self) -> int:
        return len(self.trainer) + len(self.rollout) + len(self.standby)


def parse_placement_map(raw: "str | dict | PlacementMap | None") -> PlacementMap | None:
    if raw is None or isinstance(raw, PlacementMap):
        return raw
    data = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(data, dict), f"a placement map is a JSON object keyed by role, got {data!r}"
    unknown = set(data) - set(_PLACEMENT_MAP_ROLES) - {ROLLOUT_CELLS_KEY}
    assert not unknown, f"placement map names unknown roles {sorted(unknown)}; known: {[*_PLACEMENT_MAP_ROLES, ROLLOUT_CELLS_KEY]}"
    fields = {}
    for role in _PLACEMENT_MAP_ROLES:
        indices = data.get(role, [])
        assert isinstance(indices, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in indices
        ), f"placement map role {role!r} must be a list of ints, got {indices!r}"
        fields[role] = tuple(indices)
    return PlacementMap(**fields, rollout_cells=_parse_rollout_cells(data.get(ROLLOUT_CELLS_KEY, [])))


def _parse_rollout_cells(raw) -> tuple[RolloutCellDecl, ...]:
    assert isinstance(raw, list), f"placement map {ROLLOUT_CELLS_KEY!r} must be a list, got {raw!r}"
    cells = []
    for entry in raw:
        assert isinstance(entry, dict) and not (
            unknown := set(entry) - {"name", "bundles", "start"}
        ), f"a rollout cell is {{'name', 'bundles', 'start'}}, got {entry!r}"
        name, bundles, start = entry.get("name"), entry.get("bundles", []), entry.get("start", True)
        assert isinstance(name, str) and name, f"rollout cell name must be a non-empty string, got {entry!r}"
        assert isinstance(bundles, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in bundles
        ), f"rollout cell {name!r} bundles must be a list of ints, got {bundles!r}"
        assert isinstance(start, bool), f"rollout cell {name!r} start must be a bool, got {start!r}"
        cells.append(RolloutCellDecl(name=name, bundles=tuple(bundles), start=start))
    return tuple(cells)


def rollout_cell_binding(pm: PlacementMap, cell: RolloutCellDecl) -> tuple[str, int] | None:
    """``(view name, slot offset)`` of a declared cell's bundles, or None for an unbound cell."""
    if not cell.bundles:
        return None
    for role, view in _ROLE_VIEW_NAMES.items():
        positions = getattr(pm, role)
        if cell.bundles[0] in positions:
            offset = positions.index(cell.bundles[0])
            assert tuple(positions[offset : offset + len(cell.bundles)]) == cell.bundles, (
                f"rollout cell {cell.name!r} bundles {list(cell.bundles)} are not a consecutive run of the {role!r} "
                f"role {list(positions)}"
            )
            return view, offset
    raise AssertionError(
        f"rollout cell {cell.name!r} bundles {list(cell.bundles)} are not in the rollout or standby role"
    )


def _validate_rollout_cells(pm: PlacementMap) -> None:
    cells = pm.rollout_cells
    names = [c.name for c in cells]
    dups = sorted({n for n in names if names.count(n) > 1})
    assert not dups, f"placement map rollout cells repeat names {dups}"
    starts = [c.start for c in cells]
    assert starts == sorted(starts, reverse=True), (
        f"placement map rollout cells must list the cells started at startup before the stopped ones, got "
        f"{[(c.name, c.start) for c in cells]}"
    )
    owner: dict[int, str] = {}
    for cell in cells:
        assert cell.bundles or not cell.start, f"rollout cell {cell.name!r} starts at startup but names no bundles"
        rollout_cell_binding(pm, cell)
        for b in cell.bundles:
            assert b not in owner, f"rollout cells {owner[b]!r} and {cell.name!r} share bundle {b}"
            owner[b] = cell.name


def validate_placement_map(pm: PlacementMap, *, trainer_num_gpus: int, rollout_num_gpus: int) -> None:
    """Reject duplicates, out-of-range or overlapping indices, gaps, and role sizes that disagree with the args."""
    n = pm.num_bundles
    for role in _PLACEMENT_MAP_ROLES:
        indices = getattr(pm, role)
        dups = sorted({i for i in indices if indices.count(i) > 1})
        assert not dups, f"placement map role {role!r} repeats bundles {dups}"
        out_of_range = sorted(i for i in indices if not 0 <= i < n)
        assert not out_of_range, f"placement map role {role!r} names bundles {out_of_range} outside 0..{n - 1}"
    for a_index, a in enumerate(_PLACEMENT_MAP_ROLES):
        for b in _PLACEMENT_MAP_ROLES[a_index + 1 :]:
            overlap = sorted(set(getattr(pm, a)) & set(getattr(pm, b)))
            assert not overlap, f"placement map roles {a!r} and {b!r} share bundles {overlap}"
    assert (
        len(pm.trainer) == trainer_num_gpus
    ), f"placement map gives the trainer {len(pm.trainer)} bundles but the args ask for {trainer_num_gpus}"
    assert (
        len(pm.rollout) == rollout_num_gpus
    ), f"placement map gives rollout {len(pm.rollout)} bundles but the args ask for {rollout_num_gpus}"
    _validate_rollout_cells(pm)


def slice_pg_info(info: PlacementGroupInfo, indices: Sequence[int]) -> PlacementGroupInfo:
    """The view of ``info`` made of its entries at ``indices`` (positions in ``info``, in the order given).

    Public so callers can build a view for ``RayWorkerManager.set_pg_view`` / ``rebind_cell`` from the startup views.
    """
    return PlacementGroupInfo(
        info.pg,
        [info.pg_reordered_bundle_indices[i] for i in indices],
        [info.pg_reordered_gpu_ids[i] for i in indices],
    )


# kept for callers of the former private name
_slice_pg_info = slice_pg_info


def create_placement_groups(
    args, *, placement_map: "PlacementMap | str | dict | None" = None
) -> dict[str, PlacementGroupInfo]:
    """Create placement groups for actor and rollout engines.

    With an explicit ``placement_map`` (or ``--yeto-placement-map``) every role gets exactly the logical bundles
    the map names, and a ``standby`` entry holds the reserved bundles no role starts on. Without one the layout is
    the trainer-then-rollout offset split.
    """
    num_gpus, rollout_offset = _get_placement_group_layout(args)

    if placement_map is None:
        placement_map = getattr(args, "yeto_placement_map", None)
    if (pm := parse_placement_map(placement_map)) is not None:
        return _create_placement_groups_from_map(args, pm, num_gpus=num_gpus, rollout_offset=rollout_offset)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    ans = {
        "actor": PlacementGroupInfo(pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": PlacementGroupInfo(pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    if args.use_critic:
        ans["critic"] = ans["actor"]
    return ans


def _create_placement_groups_from_map(
    args, pm: PlacementMap, *, num_gpus: int, rollout_offset: int
) -> dict[str, PlacementGroupInfo]:
    assert not args.colocate, "--yeto-placement-map splits GPUs between roles and cannot be combined with --colocate"
    validate_placement_map(pm, trainer_num_gpus=rollout_offset, rollout_num_gpus=num_gpus - rollout_offset)

    logger.info(f"Creating placement group with {pm.num_bundles} GPUs from explicit map {pm}...")
    full = PlacementGroupInfo(*_create_placement_group(pm.num_bundles))
    ans = {
        "actor": slice_pg_info(full, pm.trainer),
        "rollout": slice_pg_info(full, pm.rollout),
        STANDBY_PG_NAME: slice_pg_info(full, pm.standby),
    }
    if args.use_critic:
        ans["critic"] = ans["actor"]
    return ans


class TrainerInfo(NamedTuple):
    handle: BaseWorkerHandle
    restored_rollout_id: int
    start_rollout_id: int


# TODO: move (when reorganizing files)
def create_trainer_handles(args, *, trainer_configs: list[MegatronTrainerConfig]) -> dict[str, BaseWorkerHandle]:
    capability = get_backend_capability(args)
    return {
        config.trainer_id: create_trainer_controller_handle(args, capability=capability, trainer_id=config.trainer_id)
        for config in trainer_configs
    }


# TODO: move (when reorganizing files)
async def take_over_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> bool:
    await wait_external_trainers(args, handles=handles)
    resumed = await wait_trainers_idle(handles)

    if resumed and not _trainer_has_checkpoint(args):
        event_logger_checkpoint.discard(args)

    return resumed


def _trainer_has_checkpoint(args) -> bool:
    assert args.megatron_config is None, "a multi policy run's base --load holds no tracker to read"
    return read_checkpoint_tracker_iteration(args.requested_load) is not None


# TODO: move (when reorganizing files)
async def create_training_model(args, *, handle: BaseWorkerHandle, trainer_id: str, resumed: bool) -> TrainerInfo:
    restored_rollout_ids = await trainer_init_or_load_state(handle, args, trainer_id=trainer_id, resumed=resumed)
    assert len(set(restored_rollout_ids)) == 1, f"trainer {trainer_id!r} restored {restored_rollout_ids}"
    [restored_rollout_id] = set(restored_rollout_ids)

    if (x := args.start_rollout_id) is None:
        start_rollout_id = restored_rollout_id
    else:
        if x != restored_rollout_id:
            logger.info(
                f"trainer {trainer_id!r} restored rollout {restored_rollout_id}, and --start-rollout-id {x} was "
                f"asked for, so it starts at {x}"
            )
        start_rollout_id = x

    return TrainerInfo(handle=handle, restored_rollout_id=restored_rollout_id, start_rollout_id=start_rollout_id)


# TODO: move (when reorganizing files)
async def create_training_models(
    args, rollout_executor: BaseWorkerHandle
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    trainer_configs = compute_trainer_configs(args)
    handles = create_trainer_handles(args, trainer_configs=trainer_configs)
    resumed = await take_over_trainers(args, handles=handles)

    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    actor_info = await create_training_model(
        compute_trainer_args(args, actor_config),
        handle=handles[actor_config.trainer_id],
        trainer_id=actor_config.trainer_id,
        resumed=resumed,
    )

    critic_configs = [config for config in trainer_configs if config.role == CRITIC_ROLE]
    critic_info = None
    if args.use_critic:
        [critic_config] = critic_configs
        critic_info = await create_training_model(
            compute_trainer_args(args, critic_config),
            handle=handles[critic_config.trainer_id],
            trainer_id=critic_config.trainer_id,
            resumed=resumed,
        )
        assert critic_info.restored_rollout_id == actor_info.restored_rollout_id, (
            f"the actor restored to rollout {actor_info.restored_rollout_id} but its critic to "
            f"{critic_info.restored_rollout_id}"
        )
    else:
        assert (
            not critic_configs
        ), f"a run without --use-critic needs no critic, but the trainer configs are {trainer_configs}"

    args.start_rollout_id = actor_info.start_rollout_id

    await rollout_executor.set_train_parallel_config(await actor_info.handle.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_info.handle, critic_info.handle if critic_info is not None else None


async def rebuild_training_models(
    args,
    rollout_executor: BaseWorkerHandle,
    *,
    old_handles: dict[str, BaseWorkerHandle],
    worker_manager,
    trainer_pg_view: PlacementGroupInfo | None = None,
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    """Re-create the trainer on a (possibly different) bundle set with the sizes in ``args``, disposing the old one.

    This is a process rebuild for elastic placement: the old trainer controllers are disposed, every trainer pool
    (controllers and engines) is stopped, its specs are rebuilt from ``args`` (e.g. a new
    ``actor_num_gpus_per_node``), ``trainer_pg_view`` (bundles of the startup placement group) becomes the "actor"
    view when given, and ``create_training_models`` starts over; state comes back through the normal load path.
    A failure at any stage raises ``TrainerRebuildError`` (stage + cleanup outcome) after stopping the trainer
    pools again; there is no automatic rollback. It does not use the fault-tolerance cell refresh or indep-DP
    healing. ``worker_manager`` is the
    RayWorkerManager actor handle (or an object with the same async methods).
    """
    assert not args.use_fault_tolerance, "rebuild_training_models is not a fault-tolerance path"
    assert not args.indep_dp, "rebuild_training_models re-creates the trainer; indep-DP cells are not supported"
    assert args.trainer_controller_addrs is None, "an independently deployed trainer is not rebuilt from here"

    for trainer_id, handle in old_handles.items():
        try:
            await handle.dispose()
        except Exception:
            logger.warning(
                f"Disposing the old trainer controller {trainer_id!r} failed; stopping it anyway", exc_info=True
            )

    specs = [*specs_trainer_controller(args), *specs_trainer(args)]
    pool_ids = [spec.name for spec in specs]
    stage = "stop_pools"
    previous_view = None
    try:
        await _call_manager(worker_manager.stop_pools, pool_ids)
        if trainer_pg_view is not None:
            stage = "set_pg_view"
            if hasattr(worker_manager, "get_pg_view"):
                previous_view = await _call_manager(worker_manager.get_pg_view, "actor")
            # every trainer spec (actor and critic) schedules on the "actor" view; their pools are replaced next
            await _call_manager(worker_manager.set_pg_view, "actor", trainer_pg_view, replacing_pools=pool_ids)
        stage = "replace_pool_spec"
        for spec in specs:
            await _call_manager(worker_manager.replace_pool_spec, spec)
        stage = "start_pools"
        await _call_manager(worker_manager.start_pools, pool_ids)
        stage = "create_training_models"
        return await create_training_models(args, rollout_executor)
    except BaseException as error:
        # No rollback to the old trainer: its controllers are already disposed. Stop whatever was started so no
        # half-built trainer keeps bundles, point the "actor" view back at the old bundles, and report the stage;
        # the caller rebuilds again (e.g. with the previous args and view) and restores state from its cut.
        # A cancellation is cleaned up the same way and then re-raised unchanged.
        cleanup_error = None
        try:
            await _call_manager(worker_manager.stop_pools, pool_ids)
        except Exception as stop_error:  # noqa: BLE001
            cleanup_error = stop_error
        view_restored = False
        if previous_view is not None and cleanup_error is None:
            try:
                await _call_manager(worker_manager.set_pg_view, "actor", previous_view, replacing_pools=pool_ids)
                view_restored = True
            except Exception:  # noqa: BLE001
                logger.warning("Restoring the previous actor view after a failed rebuild failed", exc_info=True)
        if not isinstance(error, Exception):
            raise
        raise TrainerRebuildError(
            stage=stage,
            pool_ids=pool_ids,
            cleanup_error=cleanup_error,
            previous_view=previous_view,
            view_restored=view_restored,
        ) from error


class TrainerRebuildError(RuntimeError):
    """``rebuild_training_models`` failed at ``stage``; the old trainer is gone and no trainer is running.

    ``cleanup_error`` is set when stopping the trainer pools after the failure failed too; then workers of
    ``pool_ids`` may still hold bundles and must be stopped before the next rebuild. ``previous_view`` is the
    "actor" view before the rebuild (None when the view was not changed) and ``view_restored`` whether it was put
    back; if not, the caller passes ``previous_view`` to its next rebuild.
    """

    def __init__(
        self,
        *,
        stage: str,
        pool_ids: list[str],
        cleanup_error: BaseException | None,
        previous_view: PlacementGroupInfo | None = None,
        view_restored: bool = False,
    ) -> None:
        self.stage, self.pool_ids, self.cleanup_error = stage, pool_ids, cleanup_error
        self.previous_view, self.view_restored = previous_view, view_restored
        state = (
            "trainer pools stopped" if cleanup_error is None else f"stopping trainer pools failed: {cleanup_error!r}"
        )
        super().__init__(f"trainer rebuild failed at {stage}; the old trainer is disposed; {state}")


async def _call_manager(method, *args, **kwargs):
    if hasattr(method, "remote"):
        return await method.remote(*args, **kwargs)
    return await method(*args, **kwargs)


# TODO: move (when reorganizing files)
async def wait_external_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> None:
    """Wait for every independently deployed trainer controller, and refuse one that another run deployed."""
    if args.trainer_controller_addrs is None:
        return

    addrs = external_trainer_controller_addrs(args, trainer_ids=list(handles))
    logger.info(f"Waiting for the independently deployed trainer controllers at {addrs}")
    await wait_static_addrs_ready(addrs.values())

    identities = await asyncio.gather(*[handle.get_deployment_identity() for handle in handles.values()])
    for trainer_id, identity in zip(handles, identities, strict=True):
        _assert_external_trainer_in_run(identity, args=args, trainer_id=trainer_id)


def _assert_external_trainer_in_run(identity: DeploymentIdentity, *, args, trainer_id: str | None = None) -> None:
    assert identity.run_uuid == args.run_uuid, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, but this launch drives run {args.run_uuid}: every deployment a split run reaches has "
        f"to be a deployment of that same run, or its weight updates and its rollout samples belong to different runs"
    )
    assert identity.deploy_component == DeployComponent.TRAINER.value, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, and only a deployment that carries nothing but the trainer is reached by address: "
        f"an {DeployComponent.ALL.value} release of this run runs an orchestration script of its own, so both "
        f"scripts would drive the same trainer"
    )
    assert identity.deploy_instance_id is None or identity.deploy_instance_id == trainer_id, (
        f"trainer {trainer_id!r} answers as deployment {identity.deploy_instance_id!r}; "
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} entries are keyed by trainer id"
    )
    assert (
        identity.trainer_id == trainer_id
    ), f"{TRAINER_CONTROLLER_ADDRS_FLAG} keys trainer {identity.trainer_id!r} under {trainer_id!r}"


# TODO: move (when reorganizing files)
async def update_weights(
    args,
    actor_model,
    rollout_executor,
    inference_controller: BaseWorkerHandle,
    *,
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
    members: list[str] | None = None,
    expected_epoch: int | None = None,
    admit_cordoned: bool = False,
) -> int | None:
    """Publish the actor weights and return the weight version the trainer reported.

    ``members`` (cell ids, with the membership ``expected_epoch``) restricts the publish to those engines only. A
    member publish does not set the executor's weight version: non-members may still serve an older version, so the
    caller sets it with ``commit_weight_version`` once every serving engine carries the returned version.
    ``admit_cordoned`` registers the engines this publish readies cordoned; the caller verifies their weights and
    opens them with ``InferenceController.admit_cells(..., expected_weight_version=<returned version>)``.
    """
    orchestration_executor = FTTestActionOrchestrationExecutor.from_args(args, trainer_model_id=trainer_model_id)
    if rollout_id is not None:
        await orchestration_executor.run_after_step(rollout_id=rollout_id)

    info: UpdatableEngines = await inference_controller.start_update_weights(
        model_id=trainer_model_id,
        **({} if members is None else dict(members=members, expected_epoch=expected_epoch)),
        **({"admit_cordoned": True} if admit_cordoned else {}),
    )
    try:
        weight_version = await actor_model.update_weights(info=info, rollout_id=rollout_id)
    except BaseException:
        await inference_controller.abort_update_weights()
        raise
    await inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    await _maybe_log_inference_engine_weight_checksums(
        args, inference_controller=inference_controller, rollout_id=rollout_id, trainer_model_id=trainer_model_id
    )

    if weight_version is not None and members is None:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)
    return weight_version


async def commit_weight_version(
    rollout_executor,
    inference_controller: BaseWorkerHandle,
    *,
    weight_version: int,
    expected_epoch: int,
    trainer_model_id: str | None = None,
) -> None:
    """Set the executor's weight version after member publishes, once every Serving engine reports it.

    ``update_weights(members=...)`` leaves the executor's version alone because non-members may still serve an
    older one. The caller brings every Serving engine to ``weight_version`` (publishing to the rest, or, when the
    member publish carried the policy the others already serve, re-stamping them with
    ``InferenceController.set_cells_weight_version``) and then calls this. It refuses if the membership epoch moved
    or any Serving engine reports another version; the checks and the executor update happen under one hold of
    the controller lock.
    """
    # start_commit_weight_version checks epoch, incompleteness and every serving engine's version and keeps the
    # controller lock until end_commit_weight_version, so nothing can move them before the executor is set.
    await inference_controller.start_commit_weight_version(
        weight_version=weight_version, expected_epoch=expected_epoch, model_id=trainer_model_id
    )
    try:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)
    finally:
        await inference_controller.end_commit_weight_version()


async def _maybe_log_inference_engine_weight_checksums(
    args, *, inference_controller: BaseWorkerHandle, rollout_id: int | None, trainer_model_id: str | None
) -> None:
    if not is_event_logger_initialized():
        return
    if args.debug_train_only or args.debug_rollout_only:
        return

    check_weights_result = await inference_controller.check_weights(action="checksum", model_id=trainer_model_id)
    if not check_weights_result:
        return
    engine_checksums = flatten_inference_engine_checksums(check_weights_result)
    get_event_logger().log(
        InferenceEngineWeightChecksumEvent,
        dict(
            rollout_id=args.start_rollout_id - 1 if rollout_id is None else rollout_id,
            trainer_model_id=trainer_model_id,
            engine_checksums=engine_checksums,
        ),
    )


# TODO: move (when reorganizing files)
def maybe_start_api_server(
    args, *, trainer_models: dict[str, BaseWorkerHandle], inference_controller: BaseWorkerHandle
) -> None:
    if not args.api_server_port:
        return

    start_api_server(
        args=args,
        trainer_models=trainer_models,
        inference_controller=inference_controller,
        host=args.api_server_host,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=get_backend_capability(args).cell_operations(),
    )


class RolloutComponents(NamedTuple):
    inference_controller: BaseWorkerHandle
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


# TODO: move (when reorganizing files)
async def create_rollout_components(args) -> RolloutComponents:
    capability = get_backend_capability(args)

    if not args.debug_train_only or args.eval_num_gpus > 0:
        await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))

        session_server_provider = (
            capability.static_worker_provider(pool_id=SESSION_SERVER_POOL_ID) if args.use_session_server else None
        )
        await wait_session_server_ready(args, provider=session_server_provider)

    rollout_executor = create_rollout_executor_handle(capability=capability)
    await wait_until_worker_not_initialized(rollout_executor)

    inference_controller = create_inference_controller_handle(capability=capability)
    await init_or_reset_inference_controller(inference_controller, args=args)

    await rollout_executor.init()

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if (eval_fleet_info := await inference_controller.get_eval_fleet_info()) is not None:
        await rollout_executor.set_eval_fleet_info(eval_fleet_info)

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
