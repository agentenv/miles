import gc
import hashlib
import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import (
    build_lora_sync_config,
    is_lora_weight_name,
    lora_base_sync_skipped,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import (
    _check_weight_sync_results,
    begin_weight_update,
    end_weight_update,
    format_published_weight_version,
    weight_update_selector,
)
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

_LORA_FLAT_BUCKET_MAX_BYTES = 256 * 1024 * 1024
_LORA_FLAT_BUCKET_MAX_TENSORS = 1024
_LORA_FLAT_BUCKET_MAX_THREADS = 16


def _colocated_engine_count(
    args: Namespace,
    engine_gpu_offsets: Sequence[int],
    engine_gpu_counts: Sequence[int],
) -> int:
    """Classify only true colocated engines; rollout offsets may be PG-relative."""

    if getattr(args, "bridge_distributed_weight_sync", False):
        if getattr(args, "colocate", False):
            raise RuntimeError("Bridge distributed weight sync cannot use colocated engines")
        return 0
    total_actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    count = 0
    for gpu_offset, gpu_count in zip(
        engine_gpu_offsets,
        engine_gpu_counts,
        strict=True,
    ):
        if gpu_offset + gpu_count > total_actor_gpus:
            break
        count += 1
    return count


def _partition_lora_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    max_bytes: int = _LORA_FLAT_BUCKET_MAX_BYTES,
    max_tensors: int = _LORA_FLAT_BUCKET_MAX_TENSORS,
) -> list[list[tuple[str, torch.Tensor]]]:
    """Split a full adapter before flattening it.

    torch.cat scales pathologically when handed tens of thousands of inputs.
    Bound both the copy size and input count while preserving the original
    tensor order. A single tensor larger than max_bytes is kept in its own
    bucket.
    """
    if max_bytes <= 0 or max_tensors <= 0:
        raise ValueError("LoRA flat-bucket limits must be positive")

    buckets: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    current_bytes = 0
    for item in named_tensors:
        tensor_bytes = item[1].numel() * item[1].element_size()
        if current and (len(current) >= max_tensors or current_bytes + tensor_bytes > max_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += tensor_bytes
    if current:
        buckets.append(current)
    return buckets


@contextmanager
def _bounded_lora_flatten_threads(enabled: bool):
    """Bound per-rank intra-op contention only while constructing flat buffers."""
    previous_threads = None
    if enabled:
        current_threads = torch.get_num_threads()
        if current_threads > _LORA_FLAT_BUCKET_MAX_THREADS:
            previous_threads = current_threads
            logger.info(
                "Limiting LoRA flatten intra-op threads from %d to %d",
                current_threads,
                _LORA_FLAT_BUCKET_MAX_THREADS,
            )
            torch.set_num_threads(_LORA_FLAT_BUCKET_MAX_THREADS)
    try:
        yield
    finally:
        if previous_threads is not None:
            torch.set_num_threads(previous_threads)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.is_lora = is_lora
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=self.is_lora,
        )
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False

        # Create IPC gather groups within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False

    def published_weight_version(self) -> str:
        """Return the exact version label attached to the current weights."""
        return format_published_weight_version(self.args, self.weight_version)

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self.all_rollout_engines = tuple(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock
        self._connection_stale = False

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # RolloutServer offsets are relative to the rollout placement-group
        # slice.  The explicit Bridge path is always non-colocated, so do not
        # compare those relative offsets to actor-world ranks.
        colocate_engine_nums = _colocated_engine_count(
            self.args,
            engine_gpu_offsets,
            engine_gpu_counts,
        )

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and get_parallel_state().pp.rank == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear ``update_weight_metrics``. Empty under colocate today; kept symmetric
        with the distributed updaters so the actor can drain unconditionally."""
        out = self.__dict__.pop("update_weight_metrics", {})
        return out

    @torch.no_grad()
    def prepare_weight_update(self) -> None:
        if self.is_lora:
            self._prepared_lora_weights = self._collect_lora_weights(self.weights_getter())

    def _collect_lora_weights(self, megatron_local_weights) -> list:
        accumulated_named_tensors: list = []
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
            megatron_local_weights, weight_type="lora"
        ):
            accumulated_named_tensors.extend(hf_named_tensors)
        return accumulated_named_tensors

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()

        # LoRA never mutates the base. With either path that retains it on the
        # rollout side (distributed keeps it on GPU; colocate either keeps a
        # host mirror or reloads it from the immutable checkpoint), we can skip
        # the base sync entirely
        # and the surrounding restore_weights_before_load / post_process_quantization
        # calls that would otherwise prep / re-quantize fresh base bytes.
        # TODO: implement lora weight checker
        skip_base_sync = (
            self.is_lora
            and (self.use_distribute or lora_base_sync_skipped(self.args))
            and not getattr(self.args, "check_weight_update_equal", False)
        )

        prepared_lora_weights = self.__dict__.pop("_prepared_lora_weights", None)

        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.all_rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.all_rollout_engines])
            if not skip_base_sync:
                selector = weight_update_selector(self.args)
                if selector == "all":
                    begin_weight_update(self.all_rollout_engines)
                else:
                    begin_weight_update(self.all_rollout_engines, selector)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = None
        if not skip_base_sync or (self.is_lora and prepared_lora_weights is None):
            megatron_local_weights = self.weights_getter()

        if not skip_base_sync:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                try:
                    refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                    results = ray.get(refs)
                    _check_weight_sync_results(results, is_lora=False)
                    del long_lived_tensors
                finally:
                    self._release_distributed_engine_lock()

        if self.is_lora:
            # SGLang's load_lora_adapter_from_tensors expects the full adapter in
            # one call; drain the bridge's chunker so --update-weight-buffer-size
            # only bounds the base path.
            accumulated_named_tensors = (
                prepared_lora_weights
                if prepared_lora_weights is not None
                else self._collect_lora_weights(megatron_local_weights)
            )

            if not accumulated_named_tensors:
                raise RuntimeError(
                    "LoRA weight sync failed: the weight iterator produced zero chunks. "
                    "No adapter weights were sent to the rollout engine. This usually means "
                    "the Megatron-Bridge or SGLang version is incompatible."
                )

            refs, long_lived_tensors = self._send_lora_params(accumulated_named_tensors)
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=True)

            # Only the gather rank waits for SGLang's HTTP response; other
            # ranks receive no Ray refs. Keep every producer allocation alive
            # until the receiver has copied all CUDA IPC tensors to CPU, then
            # explicitly return the staging storage to CUDA before training.
            dist.barrier(group=get_gloo_group())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del long_lived_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            if not self._lora_base_synced:
                self._lora_base_synced = True

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            # Skip when no fresh base bytes landed (skip_base_sync).
            if not skip_base_sync:
                end_weight_update(self.all_rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.all_rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        published_weight_version = self.published_weight_version()
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=weight_update_selector(self.args),
            weight_version=published_weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            self._acquire_distributed_engine_lock()
            try:
                distributed_kwargs = {}
                selector = weight_update_selector(self.args)
                if selector != "all":
                    distributed_kwargs["selector"] = selector
                refs_distributed = update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    published_weight_version,
                    self.distributed_rollout_engines,
                    hf_named_tensors,
                    **distributed_kwargs,
                )
            except BaseException:
                self._release_distributed_engine_lock()
                raise
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        return refs or [], long_lived_tensors

    def _acquire_distributed_engine_lock(self) -> None:
        if self.__dict__.get("_distributed_engine_lock_held", False):
            raise RuntimeError("distributed rollout-engine lock is already held")
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        self._distributed_engine_lock_held = True

    def _release_distributed_engine_lock(self) -> None:
        if not self.__dict__.pop("_distributed_engine_lock_held", False):
            return
        released = ray.get(self.rollout_engine_lock.release.remote())
        if released is False:
            raise RuntimeError("distributed rollout-engine lock release failed")

    def _send_lora_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")
        else:
            refs, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors=hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                selector=weight_update_selector(self.args),
                lora_config=self._lora_config,
                lora_name=LORA_ADAPTER_NAME,
                lora_loaded=self._lora_loaded,
                check_equal=getattr(self.args, "check_lora_weight_equal", False),
            )
            self._lora_loaded = True
            return refs or [], long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    lora_loaded: bool = False,
    check_equal: bool = False,
    selector: str = "all",
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if is_lora:
        # Keep every flat buffer single-dtype. Besides making bucket sizes
        # predictable, this avoids unaligned storage offsets when SGLang
        # reconstructs heterogeneous tensors from a uint8 backing buffer.
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            converted_named_tensors_by_dtypes.setdefault(tensor.dtype, []).append((name, tensor))
    elif getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            converted_named_tensors_by_dtypes.setdefault(tensor.dtype, []).append((name, tensor))

    serialized_tensors: list = []
    flattened_tensor_payloads: list[dict[str, Any]] = []
    with _bounded_lora_flatten_threads(is_lora):
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            tensor_buckets = _partition_lora_tensors(named_tensors) if is_lora else [named_tensors]
            for tensor_bucket in tensor_buckets:
                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=tensor_bucket)
                flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
                if is_lora and not flattened_tensor.is_cuda:
                    # Use independent CUDA allocations for IPC rather than CPU
                    # fds or storage tied to the trainer's offload lifecycle.
                    flattened_tensor = flattened_tensor.cuda()
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor,
                    "metadata": flattened_tensor_bucket.get_metadata(),
                }
                long_live_tensors.append(flattened_tensor_data)
                if is_lora:
                    flattened_tensor_payloads.append(flattened_tensor_data)
                else:
                    serialized_tensors.append(
                        MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                    )

    if is_lora:
        # One HTTP field per TP rank remains the public contract. The serialized
        # object inside that field carries multiple bounded flat buffers.
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_payloads, output_str=True))
        logger.info(
            "Packed %d LoRA tensors (%d bytes) into %d flattened buckets " "(max %d bytes/%d tensors)",
            len(hf_named_tensors),
            sum(t.numel() * t.element_size() for _, t in hf_named_tensors),
            len(flattened_tensor_payloads),
            _LORA_FLAT_BUCKET_MAX_BYTES,
            _LORA_FLAT_BUCKET_MAX_TENSORS,
        )

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        if is_lora:
            if lora_loaded:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))

            expected_checksums = None
            if check_equal:
                expected_checksums = {
                    n: hashlib.sha256(
                        t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for n, t in hf_named_tensors
                }

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_named_tensors=[
                        per_rank[0] if per_rank else None for per_rank in serialized_named_tensors
                    ],
                    load_format="flattened_buckets",
                    expected_checksums=expected_checksums,
                )
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                    "selector": selector,
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
