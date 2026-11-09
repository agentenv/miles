"""Rebuild the trainer processes on a (possibly different) bundle set of the startup placement group.

A process rebuild for elastic placement, not fault tolerance: the old trainer controllers are disposed, every
trainer pool (controllers and engines) is stopped, the "actor" view is optionally re-pointed, the worker manager
replaces the trainer pool specs and adopts the ScalingConfig of the given args (a trainer with another GPU
count), the pools start again and
``create_training_models`` restores state through the normal load path. There is no rollback to the old trainer;
a failure stops the trainer pools again, puts the previous view back and raises ``TrainerRebuildError``.
"""

import logging

from ray.actor import ActorMethod

from miles.ray.placement_group import PlacementGroupInfo, create_trainer_handles, create_training_models
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import compute_trainer_configs, compute_trainer_controller_pool_id, compute_trainer_pool_id
from miles.ray.wiring import compute_backend_capability
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


class TrainerRebuildError(RuntimeError):
    """``rebuild_training_models`` failed at ``stage``; the old trainer is gone and no trainer is running.

    ``cleanup_error`` is set when stopping the trainer pools after the failure failed too (their workers may still
    hold bundles). ``previous_view`` is the "actor" view before the rebuild (None when it was not changed) and
    ``view_restored`` whether it was put back.
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


async def rebuild_training_models(
    args,
    rollout_executor: BaseWorkerHandle,
    *,
    old_handles: dict[str, BaseWorkerHandle],
    worker_manager,
    trainer_pg_view: PlacementGroupInfo | None = None,
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    """Dispose the old trainer and create it again from ``args`` (sizes included) on ``trainer_pg_view`` if given.

    ``worker_manager`` is the RayWorkerManager actor handle. ``args`` may carry other trainer GPU counts than the
    run started with; the worker manager then lays the trainer pools out from ``ScalingConfig.slice_from(args)``.
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

    trainer_configs = compute_trainer_configs(args)
    pool_ids = [
        pool_id
        for config in trainer_configs
        for pool_id in (
            compute_trainer_controller_pool_id(config.trainer_id),
            compute_trainer_pool_id(config.trainer_id),
        )
    ]
    stage, previous_view = "stop_pools", None
    try:
        await _call(worker_manager.stop_pools, pool_ids)
        if trainer_pg_view is not None:
            stage = "set_pg_view"
            previous_view = await _call(worker_manager.get_pg_view, "actor")
            # every trainer pool (actor and critic) schedules on the "actor" view; their layout is replaced next
            await _call(worker_manager.set_pg_view, "actor", trainer_pg_view, replacing_pools=pool_ids)
        stage = "replace_pools"
        specs = [spec for spec in compute_specs(args) if spec.name in pool_ids]
        await _call(worker_manager.replace_pools, specs, scaling=ScalingConfig.slice_from(args))
        stage = "start_pools"
        await _call(worker_manager.start_pools, pool_ids)
        stage = "create_training_models"
        handles = create_trainer_handles(
            args, trainer_configs=trainer_configs, capability=compute_backend_capability(args)
        )
        return await create_training_models(args, rollout_executor, handles=handles)
    except BaseException as error:
        cleanup_error = None
        try:
            await _call(worker_manager.stop_pools, pool_ids)
        except Exception as stop_error:  # noqa: BLE001
            cleanup_error = stop_error
        view_restored = False
        if previous_view is not None and cleanup_error is None:
            try:
                await _call(worker_manager.set_pg_view, "actor", previous_view, replacing_pools=pool_ids)
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


async def _call(method, *args, **kwargs):
    # the RayWorkerManager actor handle, or an object with the same async methods (tests)
    if isinstance(method, ActorMethod):
        return await method.remote(*args, **kwargs)
    return await method(*args, **kwargs)
