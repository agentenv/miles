"""Replicated Megatron LoRA state at an external policy-sync boundary."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_CANONICAL_PREFIX = "base_model.model."


@dataclass(frozen=True)
class TrainableState:
    policy_version: int
    layout_hash: str
    tensors: Mapping[str, torch.Tensor]
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None


def _layout_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"yeto-layout-v1\0")
    digest.update(struct.pack("<I", 1))
    digest.update(struct.pack("<BI", 0, len(tensors)))
    for name, tensor in sorted(tensors.items()):
        encoded_name = name.encode("utf-8")
        shape = tuple(int(dim) for dim in tensor.shape)
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", tensor.numel()))
        digest.update(struct.pack("<I", len(shape)))
        digest.update(struct.pack(f"<{len(shape)}Q", *shape))
    return digest.hexdigest()


def make_trainable_state(
    policy_version: int,
    tensors: Mapping[str, torch.Tensor],
    *,
    train_rollout_kl: float | None = None,
    ess_ratio: float | None = None,
    pg_clipfrac: float | None = None,
    train_seconds: float | None = None,
) -> TrainableState:
    if policy_version < 0:
        raise ValueError("policy version must be non-negative")
    if not tensors:
        raise ValueError("trainable state is empty")
    canonical = {}
    for name, tensor in sorted(tensors.items()):
        if not name.startswith(_CANONICAL_PREFIX) or not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"not a canonical PEFT LoRA tensor name: {name!r}")
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name!r} contains NaN or Inf")
        canonical[name] = value.clone()
    return TrainableState(
        policy_version,
        _layout_hash(canonical),
        canonical,
        train_rollout_kl,
        ess_ratio,
        pg_clipfrac,
        train_seconds,
    )


def _capture_external_train_metrics(args, metrics: Mapping[str, Any]) -> None:
    if getattr(args, "external_policy_sync_path", None) is None or dist.get_rank() != 0:
        return
    args._external_train_metrics = {
        name: float(metrics[name])
        for name in (
            "train/train_rollout_kl",
            "train/ess_ratio",
            "train/pg_clipfrac",
        )
        if name in metrics
    }


def _adapter_sides(actor) -> tuple[tuple[str, Any], ...]:
    cached = getattr(actor, "_external_trainable_sides", None)
    if cached is not None:
        return cached

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(actor.args.hf_checkpoint, trust_remote_code=True)
    model_bridge = getattr(bridge, "_model_bridge", None)
    build_tasks = getattr(model_bridge, "build_adapter_conversion_tasks", None)
    if build_tasks is None:
        raise RuntimeError("Megatron-Bridge lacks adapter conversion tasks")

    tasks_by_base = build_tasks(actor.model)
    sides = []
    for base_name in sorted(tasks_by_base):
        tasks = sorted(
            tasks_by_base[base_name],
            key=lambda task: task.adapter_key or "",
        )
        for task in tasks:
            for side in (task.linear_in_task, task.linear_out_task):
                parameter = side.param_weight
                hf_param = side.mapping.hf_param
                if isinstance(hf_param, str):
                    raw_name = hf_param
                elif isinstance(hf_param, dict) and len(set(hf_param.values())) == 1:
                    raw_name = next(iter(hf_param.values()))
                else:
                    raise RuntimeError(
                        f"ambiguous canonical LoRA mapping for {side.param_name!r}"
                    )
                name = raw_name if raw_name.startswith(_CANONICAL_PREFIX) else _CANONICAL_PREFIX + raw_name
                if parameter is not None:
                    main = getattr(parameter, "main_param", None)
                    if (
                        main is None
                        or main.dtype != torch.float32
                        or main.numel() != parameter.numel()
                    ):
                        raise RuntimeError(
                            f"LoRA parameter {side.param_name!r} has no complete "
                            "FP32 optimizer master"
                        )
                sides.append((name, side))

    names = [name for name, _ in sides]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Megatron produced an empty or duplicate LoRA mapping")
    mapped = {id(side.param_weight) for _, side in sides if side.param_weight is not None}
    trainable = {
        id(parameter)
        for model_chunk in actor.model
        for parameter in model_chunk.parameters()
        if parameter.requires_grad
    }
    if mapped != trainable:
        raise RuntimeError("adapter conversion does not cover every trainable parameter")

    actor._external_trainable_sides = tuple(sorted(sides))
    return actor._external_trainable_sides


@contextmanager
def _optimizer_masters_as_model_parameters(actor):
    """Expose f32 optimizer masters to Bridge conversion without replacing Parameters."""

    originals = []
    for _name, side in _adapter_sides(actor):
        parameter = side.param_weight
        if parameter is None:
            continue
        originals.append((parameter, parameter.data))
        parameter.data = parameter.main_param.view(parameter.shape)
    try:
        yield
    finally:
        for parameter, original in originals:
            parameter.data = original


def _collective_adapter_tensors(actor) -> dict[str, torch.Tensor]:
    """Collect a complete canonical f32 PEFT state across TP and PP ranks."""

    from megatron.bridge import AutoBridge

    from miles.utils import megatron_bridge_utils

    bridge = AutoBridge.from_hf_pretrained(
        actor.args.hf_checkpoint,
        trust_remote_code=True,
    )
    tensors: dict[str, torch.Tensor] = {}
    with _optimizer_masters_as_model_parameters(actor):
        with megatron_bridge_utils.patch_megatron_model(actor.model):
            for raw_name, weight, _megatron_name in bridge.export_adapter_weights(
                actor.model,
                cpu=True,
                show_progress=False,
            ):
                name = (
                    raw_name
                    if raw_name.startswith(_CANONICAL_PREFIX)
                    else _CANONICAL_PREFIX + raw_name
                )
                value = weight.detach().to(dtype=torch.float32).contiguous()
                previous = tensors.get(name)
                if previous is not None and not torch.equal(previous, value):
                    raise RuntimeError(f"conflicting collective LoRA tensor {name!r}")
                tensors[name] = value
    expected = {name for name, _side in _adapter_sides(actor)}
    if set(tensors) != expected:
        missing = sorted(expected - set(tensors))
        extra = sorted(set(tensors) - expected)
        raise RuntimeError(
            f"collective LoRA export mismatch: missing={missing}, extra={extra}"
        )
    return tensors


@torch.no_grad()
def export_trainable_state(actor, *, policy_version: int) -> TrainableState | None:
    tensor_parallel = int(getattr(actor.args, "tensor_model_parallel_size", 1))
    pipeline_parallel = int(getattr(actor.args, "pipeline_model_parallel_size", 1))
    if tensor_parallel > 1 or pipeline_parallel > 1:
        # Bridge conversion is collective: every Megatron rank must enter it.
        tensors = _collective_adapter_tensors(actor)
    else:
        tensors = {}
        for name, side in _adapter_sides(actor):
            parameter = side.param_weight
            if parameter is None:
                raise RuntimeError(f"single-rank LoRA side {name!r} is absent")
            converted = side.mapping.megatron_to_hf(
                parameter.main_param.view(parameter.shape),
                side.megatron_module,
            )
            tensors[name] = next(iter(converted.values()))
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    metrics = (
        getattr(actor.args, "_external_train_metrics", {})
        if getattr(actor.args, "external_policy_sync_path", None) is not None
        else {}
    )
    return make_trainable_state(
        policy_version,
        tensors,
        train_rollout_kl=metrics.get("train/train_rollout_kl"),
        ess_ratio=metrics.get("train/ess_ratio"),
        pg_clipfrac=metrics.get("train/pg_clipfrac"),
        train_seconds=getattr(actor.args, "_external_train_seconds", None),
    )


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


def _copy_masters_to_model(actor) -> None:
    for child in _optimizer_children(actor.optimizer):
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("Megatron optimizer lacks main-to-model copy")
        copy()


def _reset_optimizer_state(actor, parameters: list[torch.Tensor]) -> int:
    parameter_ids = {id(parameter) for parameter in parameters}
    for child in _optimizer_children(actor.optimizer):
        optimizer = getattr(child, "optimizer", child)
        for parameter in list(optimizer.state):
            if id(parameter) in parameter_ids:
                optimizer.state.pop(parameter, None)
    return len(parameter_ids)


def _align_scheduler(actor, policy_version: int) -> None:
    scheduler = actor.opt_param_scheduler
    batch_size = actor.args.global_batch_size
    target = policy_version * actor.args.num_steps_per_rollout * batch_size
    if scheduler is None or batch_size <= 0 or scheduler.num_steps % batch_size:
        raise RuntimeError("Megatron scheduler progress is not an integral optimizer step")
    if scheduler.num_steps > target:
        raise RuntimeError("Megatron scheduler is ahead of the committed policy")
    if scheduler.num_steps < target:
        scheduler.step(increment=target - scheduler.num_steps)


@torch.no_grad()
def apply_trainable_state(
    actor,
    state: TrainableState,
    *,
    reset_optimizer: bool,
) -> int:
    incoming = make_trainable_state(state.policy_version, state.tensors)
    if incoming.layout_hash != state.layout_hash:
        raise RuntimeError("trainable state layout hash mismatch")

    sides = dict(_adapter_sides(actor))
    if set(incoming.tensors) != set(sides):
        missing = sorted(set(sides) - set(incoming.tensors))
        extra = sorted(set(incoming.tensors) - set(sides))
        raise RuntimeError(f"global LoRA mapping mismatch: missing={missing}, extra={extra}")
    current = export_trainable_state(actor, policy_version=state.policy_version)
    current_layout = current.layout_hash if current is not None else None
    if dist.is_initialized():
        values = [current_layout]
        dist.broadcast_object_list(values, src=0)
        current_layout = values[0]
    if incoming.layout_hash != current_layout:
        raise RuntimeError("global LoRA layout hash mismatch")

    mapped = {}
    with _optimizer_masters_as_model_parameters(actor):
        for name, side in sides.items():
            if side.param_weight is None:
                continue
            value = incoming.tensors[name]
            target = side.mapping.hf_to_megatron(value, side.megatron_module)
            if target.numel() != side.param_weight.numel():
                raise RuntimeError(f"global LoRA shape mismatch for {name!r}")
            mapped[name] = target.reshape(side.param_weight.shape).contiguous()

        _align_scheduler(actor, state.policy_version)
        for name, target in mapped.items():
            side = sides[name]
            side.param_weight.main_param.view(side.param_weight.shape).copy_(target)

    _copy_masters_to_model(actor)
    if dist.is_initialized():
        dist.barrier()
    local_parameters = [
        side.param_weight.main_param
        for side in sides.values()
        if side.param_weight is not None
    ]
    reset_count = (
        _reset_optimizer_state(
            actor,
            local_parameters,
        )
        if reset_optimizer
        else 0
    )
    actor.weights_backuper.backup("actor")
    # Ray requires an identical result from every rank.  Report canonical
    # tensors rather than local TP/PP parameter shards.
    return len(incoming.tensors) if reset_optimizer else 0
