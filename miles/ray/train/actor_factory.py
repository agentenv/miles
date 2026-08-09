import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus

TMS_TRAIN_DISK_BACKUP_DIR_ENV = "MILES_TMS_TRAIN_DISK_BACKUP_DIR"
TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV = "MILES_TMS_TRAIN_DISK_BACKUP_CHUNK_MB"
TMS_TRAIN_DISK_BACKUP_DEFAULT_CHUNK_MB = 256


def _configure_train_tms_env(env_vars: dict[str, str], dynlib_path: str) -> None:
    """Configure the trainer-only TMS region without affecting rollout actors."""

    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"

    # Do not allow train_env_vars to leave the mutually-exclusive backup modes
    # enabled together. The Miles-specific inputs below are the sole selector.
    for name in (
        "TMS_INIT_ENABLE_CPU_BACKUP",
        "TMS_INIT_ENABLE_DISK_BACKUP",
        "TMS_DISK_BACKUP_DIR",
        "TMS_DISK_BACKUP_CHUNK_MB",
    ):
        env_vars.pop(name, None)

    disk_backup_dir = os.environ.get(TMS_TRAIN_DISK_BACKUP_DIR_ENV)
    raw_chunk_mb = os.environ.get(TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV)
    if not disk_backup_dir:
        if raw_chunk_mb:
            raise ValueError(
                f"{TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV} requires "
                f"{TMS_TRAIN_DISK_BACKUP_DIR_ENV}"
            )
        env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
        return

    if not os.path.isabs(disk_backup_dir):
        raise ValueError(f"{TMS_TRAIN_DISK_BACKUP_DIR_ENV} must be absolute")
    chunk_mb = (
        TMS_TRAIN_DISK_BACKUP_DEFAULT_CHUNK_MB
        if raw_chunk_mb is None
        else int(raw_chunk_mb)
    )
    if chunk_mb <= 0:
        raise ValueError(f"{TMS_TRAIN_DISK_BACKUP_CHUNK_MB_ENV} must be positive")

    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
    env_vars["TMS_INIT_ENABLE_DISK_BACKUP"] = "1"
    env_vars["TMS_DISK_BACKUP_DIR"] = disk_backup_dir
    env_vars["TMS_DISK_BACKUP_CHUNK_MB"] = str(chunk_mb)


def allocate_gpus_for_actor(
    args,
    gpus_per_cell: int,
    pg: tuple[PlacementGroup, list[int], list[int]],
    num_gpus_per_actor: float,
    indep_dp_store_addr: str,
    role: str,
    cell_index: int,
):
    world_size = gpus_per_cell

    # Use placement group to lock resources for models of same type
    assert pg is not None
    pg, reordered_bundle_indices, _reordered_gpu_ids = pg

    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", default_fp8_block_scaling_fp32_scales()
        ),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if args.offload_train and args.train_backend == "megatron":
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
        _configure_train_tms_env(env_vars, dynlib_path)

    backend = args.train_backend
    if backend == "megatron":
        from miles.backends.megatron_utils.actor import MegatronTrainRayActor

        actor_impl = MegatronTrainRayActor

    else:
        from miles.backends.experimental.fsdp_utils import FSDPTrainRayActor

        actor_impl = FSDPTrainRayActor

    ft = args.use_fault_tolerance
    TrainRayActor = ray.remote(
        num_gpus=1,
        runtime_env={"env_vars": env_vars},
        **(dict(concurrency_groups={"heartbeat_status": 1, "default": 1, "fault_injector": 1}) if ft else {}),
    )(_with_ft_concurrency_groups(actor_impl) if ft else actor_impl)

    # Create worker actors
    actor_handles = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        actor = TrainRayActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        ).remote(
            args,
            world_size,
            rank,
            master_addr,
            master_port,
            indep_dp_store_addr=indep_dp_store_addr,
            role=role,
            cell_index=cell_index,
        )
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
        actor_handles.append(actor)

    return actor_handles


def _with_ft_concurrency_groups(actor_impl: type) -> type:
    class _FtTrainRayActor(actor_impl):
        @ray.method(concurrency_group="heartbeat_status")
        def get_heartbeat_status(self) -> HeartbeatStatus:
            return super().get_heartbeat_status()

        @ray.method(concurrency_group="fault_injector")
        def inject_fault(self, mode: str) -> None:
            super().inject_fault(mode)

    _FtTrainRayActor.__name__ = actor_impl.__name__
    _FtTrainRayActor.__qualname__ = actor_impl.__qualname__
    return _FtTrainRayActor
