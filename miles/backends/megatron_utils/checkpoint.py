import logging
import os
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch.distributed as dist

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from miles.utils import megatron_bridge_utils

from .lora_utils import is_lora_enabled, is_lora_model, load_lora_adapter, save_lora_checkpoint

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "save_checkpoint_with_lora", "load_checkpoint"]


_CRITIC_HEAD_STATE_KEYS = frozenset({"output_layer.weight", "output_layer.bias"})


def _unwrap_module(module: Any) -> Any:
    """Unwrap the DDP/precision wrappers used by Megatron without importing their types."""
    seen: set[int] = set()
    while hasattr(module, "module") and id(module) not in seen:
        seen.add(id(module))
        nested = module.module
        if nested is module:
            break
        module = nested
    return module


def _is_critic_head_state_entry(name: object, value: object) -> bool:
    """Return whether a sharded-state entry belongs to Miles' added critic head."""
    candidates = (name, getattr(value, "key", None))
    return any(
        isinstance(candidate, str)
        and any(candidate == key or candidate.endswith(f".{key}") for key in _CRITIC_HEAD_STATE_KEYS)
        for candidate in candidates
    )


@contextmanager
def _omit_fresh_critic_head_from_sharded_state(ddp_model):
    """Keep a freshly initialized critic head out of an actor-checkpoint load.

    Megatron distributed checkpoints validate the model's requested sharded
    state before loading.  A Miles critic replaces the language-model output
    layer with a value head, so an actor checkpoint correctly has no tensors for
    that new head.  Temporarily filtering exactly those two entries preserves
    strict validation for every backbone tensor and leaves the already-created
    value head at its normal initialization.
    """
    patched: list[tuple[Any, bool, object | None]] = []
    omitted: set[str] = set()
    expects_head_on_rank = False

    try:
        for wrapped in ddp_model:
            module = _unwrap_module(wrapped)
            if getattr(module, "output_layer", None) is not None:
                expects_head_on_rank = True
            original = module.sharded_state_dict
            had_instance_override = "sharded_state_dict" in vars(module)
            prior_instance_value = vars(module).get("sharded_state_dict")

            def filtered_sharded_state_dict(*args, _original=original, **kwargs):
                state = _original(*args, **kwargs)
                filtered = state.copy()
                for name, value in tuple(state.items()):
                    if _is_critic_head_state_entry(name, value):
                        canonical_name = getattr(value, "key", None)
                        omitted.add(canonical_name if isinstance(canonical_name, str) else str(name))
                        filtered.pop(name)
                return filtered

            module.sharded_state_dict = filtered_sharded_state_dict
            patched.append((module, had_instance_override, prior_instance_value))

        yield omitted

        if expects_head_on_rank:
            normalized = {
                key
                for key in _CRITIC_HEAD_STATE_KEYS
                if any(name == key or name.endswith(f".{key}") for name in omitted)
            }
            if normalized != _CRITIC_HEAD_STATE_KEYS:
                raise RuntimeError(
                    "critic actor-checkpoint bootstrap did not omit exactly the fresh value head; "
                    f"expected={sorted(_CRITIC_HEAD_STATE_KEYS)} observed={sorted(omitted)}"
                )
        elif omitted:
            raise RuntimeError(f"critic actor-checkpoint bootstrap omitted a head on a non-post-process rank: {omitted}")
    finally:
        for module, had_instance_override, prior_instance_value in reversed(patched):
            if had_instance_override:
                module.sharded_state_dict = prior_instance_value
            else:
                delattr(module, "sharded_state_dict")


def _critic_actor_bootstrap_context(args, ddp_model):
    enabled = bool(getattr(args, "_critic_bootstrap_from_actor_checkpoint", False))
    if not enabled:
        return nullcontext()
    role = getattr(ddp_model[0], "role", None) if ddp_model else None
    if role != "critic":
        raise RuntimeError("actor-checkpoint critic bootstrap is only valid for the critic role")
    return _omit_fresh_critic_head_from_sharded_state(ddp_model)


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()

    load_path = args.load

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if has_local_checkpoint_manager or _is_megatron_checkpoint(load_path):
        with _critic_actor_bootstrap_context(args, ddp_model):
            result = _load_checkpoint_megatron(
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
    else:
        if getattr(args, "_critic_bootstrap_from_actor_checkpoint", False):
            raise RuntimeError(
                "fresh critic bootstrap currently requires a Megatron checkpoint; "
                "convert the actor checkpoint before offline value pretraining"
            )
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Load LoRA adapter weights if available
    if is_lora_enabled(args):
        adapter_path = getattr(args, "lora_adapter_path", None)
        if adapter_path is not None:
            loaded, iteration = load_lora_adapter(
                ddp_model,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
            )
            if loaded:
                logger.info(f"Successfully loaded LoRA adapter from {adapter_path}")
                if iteration is not None:
                    result = (iteration, result[1])
            else:
                logger.warning(
                    f"LoRA is enabled and --lora-adapter-path={adapter_path} was specified, "
                    f"but adapter weights could not be loaded. "
                    f"Training will start with freshly initialized adapter weights."
                )

    return result


def save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler):
    """Extended save that handles LoRA adapters separately."""
    args = get_args()

    if is_lora_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving LoRA checkpoint to {save_dir}")
        save_lora_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
