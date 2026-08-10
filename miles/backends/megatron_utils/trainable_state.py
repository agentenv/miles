"""Replicated Megatron LoRA state at an external policy-sync boundary."""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_CANONICAL_PREFIX = "base_model.model."
_CLONE_LAYERS = 43
_CLONE_ORIGINAL_EXPERTS = 256
_CLONE_TOTAL_EXPERTS = 288
_CLONE_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_CLONE_SIDES = ("A", "B")
_EXPERT_LORA = re.compile(
    r"^(?P<prefix>base_model\.model\.model\.layers\."
    r"(?P<layer>\d+)\.mlp\.experts\.)"
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"lora_(?P<side>A|B)\.weight$"
)


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


def _require_canonical_trainable_state(state: TrainableState) -> TrainableState:
    """Validate state from ``make_trainable_state`` without copying it again."""

    if state.policy_version < 0:
        raise ValueError("policy version must be non-negative")
    if not state.tensors:
        raise ValueError("trainable state is empty")
    for name, tensor in state.tensors.items():
        if not name.startswith(_CANONICAL_PREFIX) or not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"not a canonical PEFT LoRA tensor name: {name!r}")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError(f"{name!r} is not canonical contiguous CPU float32 state")
    if _layout_hash(state.tensors) != state.layout_hash:
        raise RuntimeError("trainable state layout hash mismatch")
    return state


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


def _clone_only_lora(actor) -> bool:
    return bool(getattr(actor.args, "yeto_rl_clone_only_lora", False))


def _expected_adapter_names(
    actor,
    sides: tuple[tuple[str, Any], ...] | None = None,
) -> frozenset[str]:
    """Return the external canonical policy names for this actor.

    A standard grouped-expert LoRA parameter has one representative Bridge
    side but exports 288 logical expert tensors.  The external policy only
    carries the 32 trainable clones; frozen original-expert adapters are
    reconstructed as exact zeros when the packed Megatron parameters are
    applied.
    """

    if not _clone_only_lora(actor):
        return frozenset(name for name, _side in (sides or _adapter_sides(actor)))
    cached = getattr(actor, "_external_expected_adapter_names", None)
    if cached is not None:
        return cached
    raw = getattr(actor.args, "yeto_rl_canonical_lora_names", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise RuntimeError(
            "clone-only LoRA has no synthesized canonical policy name contract"
        )
    names = tuple(str(name) for name in raw)
    expected = frozenset(names)
    if len(expected) != len(names):
        raise RuntimeError("clone-only canonical policy contains duplicate names")

    expert_keys = set()
    for name in expected:
        if ".mlp.experts." not in name:
            continue
        match = _EXPERT_LORA.fullmatch(name)
        if match is None:
            raise RuntimeError(f"malformed clone-only expert LoRA name {name!r}")
        expert_keys.add(
            (
                int(match.group("layer")),
                int(match.group("expert")),
                match.group("projection"),
                match.group("side"),
            )
        )
    required_keys = {
        (layer, expert, projection, side)
        for layer in range(_CLONE_LAYERS)
        for expert in range(_CLONE_ORIGINAL_EXPERTS, _CLONE_TOTAL_EXPERTS)
        for projection in _CLONE_PROJECTIONS
        for side in _CLONE_SIDES
    }
    if expert_keys != required_keys:
        missing = len(required_keys - expert_keys)
        extra = len(expert_keys - required_keys)
        raise RuntimeError(
            "clone-only canonical expert policy is incomplete: "
            f"missing={missing}, extra={extra}"
        )

    local_sides = sides or _adapter_sides(actor)
    missing_local = [
        name
        for name, _side in local_sides
        if _EXPERT_LORA.fullmatch(name) is None and name not in expected
    ]
    if missing_local:
        raise RuntimeError(
            "local Bridge sides are absent from the canonical policy: "
            f"{missing_local[:4]}"
        )
    actor._external_expected_adapter_names = expected
    return expected


def _validate_clone_canonical_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
    """Require a sparse external policy containing clone experts only."""

    found = 0
    for name, _value in tensors.items():
        if ".mlp.experts." not in name:
            continue
        match = _EXPERT_LORA.fullmatch(name)
        if match is None:
            raise RuntimeError(f"malformed clone-only expert LoRA name {name!r}")
        found += 1
        expert = int(match.group("expert"))
        if expert < 0 or expert >= _CLONE_TOTAL_EXPERTS:
            raise RuntimeError(f"clone-only expert ID is out of range in {name!r}")
        if expert < _CLONE_ORIGINAL_EXPERTS:
            raise RuntimeError(
                f"original expert adapter is present at policy boundary: {name!r}"
            )
    if not found:
        raise RuntimeError("clone-only canonical policy contains no expert tensors")


def _expert_name(match: re.Match[str], expert: int, projection: str, side: str) -> str:
    return (
        f"{match.group('prefix')}{expert}.{projection}.lora_{side}.weight"
    )


def _expert_parallel_coordinates(actor) -> tuple[int, int]:
    injected_rank = getattr(actor.args, "yeto_rl_expert_parallel_rank", None)
    injected_size = getattr(actor.args, "yeto_rl_expert_parallel_size", None)
    if injected_rank is not None or injected_size is not None:
        if injected_rank is None or injected_size is None:
            raise RuntimeError("incomplete injected expert-parallel coordinates")
        rank, size = int(injected_rank), int(injected_size)
    else:
        from megatron.core import parallel_state

        rank = int(parallel_state.get_expert_model_parallel_rank())
        size = int(parallel_state.get_expert_model_parallel_world_size())
    configured = int(getattr(actor.args, "expert_model_parallel_size", size))
    if size <= 0 or not 0 <= rank < size or configured != size:
        raise RuntimeError(
            f"invalid expert-parallel coordinates rank={rank}, size={size}, "
            f"configured={configured}"
        )
    return rank, size


def _pack_expert_side(
    name: str,
    side: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    expert_parallel_rank: int,
    expert_parallel_size: int,
) -> torch.Tensor:
    """Pack global canonical expert tensors into one rank's 3-D parameter."""

    match = _EXPERT_LORA.fullmatch(name)
    if match is None:
        raise RuntimeError(f"cannot pack non-expert LoRA side {name!r}")
    parameter = side.param_weight
    if parameter is None or parameter.ndim != 3:
        raise RuntimeError(f"packed expert side {name!r} is not a rank-3 parameter")
    local_count = int(parameter.shape[0])
    start = expert_parallel_rank * local_count
    if (
        local_count <= 0
        or local_count * expert_parallel_size != _CLONE_TOTAL_EXPERTS
        or start + local_count > _CLONE_TOTAL_EXPERTS
    ):
        raise RuntimeError(
            "invalid packed expert ownership "
            f"rank={expert_parallel_rank}, size={expert_parallel_size}, "
            f"start={start}, count={local_count}"
        )
    projection = match.group("projection")
    adapter_side = match.group("side")
    if projection == "up_proj":
        raise RuntimeError("packed fc1 representative unexpectedly uses up_proj")

    packed = []
    for expert in range(start, start + local_count):
        if expert < _CLONE_ORIGINAL_EXPERTS:
            value = torch.zeros(tuple(parameter.shape[1:]), dtype=torch.float32)
        elif projection == "down_proj":
            value = tensors[
                _expert_name(match, expert, "down_proj", adapter_side)
            ]
        elif adapter_side == "A":
            gate = tensors[_expert_name(match, expert, "gate_proj", "A")]
            up = tensors[_expert_name(match, expert, "up_proj", "A")]
            if not torch.equal(gate, up):
                raise RuntimeError(
                    f"fused fc1 gate/up LoRA A tensors differ for layer "
                    f"{match.group('layer')} expert {expert}"
                )
            value = gate
        else:
            gate = tensors[_expert_name(match, expert, "gate_proj", "B")]
            up = tensors[_expert_name(match, expert, "up_proj", "B")]
            value = torch.cat((gate, up), dim=0)
        packed.append(value)
    target = torch.stack(packed, dim=0).contiguous()
    if tuple(target.shape) != tuple(parameter.shape):
        raise RuntimeError(
            f"packed expert LoRA shape mismatch for {name!r}: "
            f"got {tuple(target.shape)}, expected {tuple(parameter.shape)}"
        )
    return target


def _sparse_expert_updates(
    name: str,
    side: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    expert_parallel_rank: int,
    expert_parallel_size: int,
) -> tuple[torch.Tensor, int, tuple[tuple[int, torch.Tensor], ...]]:
    """Prepare clone-only writes without materializing frozen packed slots."""

    match = _EXPERT_LORA.fullmatch(name)
    if match is None:
        raise RuntimeError(f"cannot pack non-expert LoRA side {name!r}")
    parameter = side.param_weight
    if parameter is None or parameter.ndim != 3:
        raise RuntimeError(f"packed expert side {name!r} is not a rank-3 parameter")
    local_count = int(parameter.shape[0])
    start = expert_parallel_rank * local_count
    if (
        local_count <= 0
        or local_count * expert_parallel_size != _CLONE_TOTAL_EXPERTS
        or start + local_count > _CLONE_TOTAL_EXPERTS
    ):
        raise RuntimeError(
            "invalid packed expert ownership "
            f"rank={expert_parallel_rank}, size={expert_parallel_size}, "
            f"start={start}, count={local_count}"
        )
    projection = match.group("projection")
    adapter_side = match.group("side")
    if projection == "up_proj":
        raise RuntimeError("packed fc1 representative unexpectedly uses up_proj")

    updates = []
    for expert in range(max(start, _CLONE_ORIGINAL_EXPERTS), start + local_count):
        if projection == "down_proj":
            value = tensors[_expert_name(match, expert, "down_proj", adapter_side)]
        elif adapter_side == "A":
            gate = tensors[_expert_name(match, expert, "gate_proj", "A")]
            up = tensors[_expert_name(match, expert, "up_proj", "A")]
            if not torch.equal(gate, up):
                raise RuntimeError(
                    f"fused fc1 gate/up LoRA A tensors differ for layer "
                    f"{match.group('layer')} expert {expert}"
                )
            value = gate
        else:
            gate = tensors[_expert_name(match, expert, "gate_proj", "B")]
            up = tensors[_expert_name(match, expert, "up_proj", "B")]
            value = torch.cat((gate, up), dim=0)
        if tuple(value.shape) != tuple(parameter.shape[1:]):
            raise RuntimeError(
                f"packed expert LoRA slice mismatch for {name!r}: "
                f"got {tuple(value.shape)}, expected {tuple(parameter.shape[1:])}"
            )
        updates.append((expert - start, value))

    original_count = max(0, min(local_count, _CLONE_ORIGINAL_EXPERTS - start))
    return parameter.main_param.view(parameter.shape), original_count, tuple(updates)


def _apply_sparse_expert_updates(
    prepared: list[tuple[torch.Tensor, int, tuple[tuple[int, torch.Tensor], ...]]],
) -> None:
    for master, original_count, updates in prepared:
        if original_count:
            master[:original_count].zero_()
        for local_index, value in updates:
            master[local_index].copy_(value)


def _assert_original_packed_masters_zero(
    sides: tuple[tuple[str, Any], ...],
    *,
    expert_parallel_rank: int,
) -> None:
    found = 0
    for name, side in sides:
        match = _EXPERT_LORA.fullmatch(name)
        parameter = side.param_weight
        if match is None or parameter is None:
            continue
        found += 1
        start = expert_parallel_rank * int(parameter.shape[0])
        original_count = max(
            0,
            min(int(parameter.shape[0]), _CLONE_ORIGINAL_EXPERTS - start),
        )
        master = parameter.main_param.view(parameter.shape)
        if original_count and torch.count_nonzero(master[:original_count]).item():
            raise RuntimeError(
                f"original packed expert LoRA master is nonzero for {name!r}"
            )
    if not found:
        raise RuntimeError("clone-only actor has no packed expert LoRA sides")


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
    """Collect canonical f32 PEFT state, retaining it only on global rank zero."""

    from megatron.bridge import AutoBridge

    from miles.utils import megatron_bridge_utils

    bridge = AutoBridge.from_hf_pretrained(
        actor.args.hf_checkpoint,
        trust_remote_code=True,
    )
    retain_tensors = not dist.is_initialized() or dist.get_rank() == 0
    tensors: dict[str, torch.Tensor] = {}
    names: set[str] = set()
    sides = _adapter_sides(actor)
    if _clone_only_lora(actor):
        expert_rank, _expert_size = _expert_parallel_coordinates(actor)
        # Validate frozen originals once in their packed GPU masters.  Checking
        # every expanded CPU expert tensor made r64 export spend tens of
        # minutes in torch.count_nonzero despite those tensors being omitted
        # from the external policy.
        _assert_original_packed_masters_zero(
            sides,
            expert_parallel_rank=expert_rank,
        )
    with _optimizer_masters_as_model_parameters(actor):
        with megatron_bridge_utils.patch_megatron_model(actor.model):
            for raw_name, weight, _megatron_name in bridge.export_adapter_weights(
                actor.model,
                cpu=False,
                show_progress=False,
            ):
                name = (
                    raw_name
                    if raw_name.startswith(_CANONICAL_PREFIX)
                    else _CANONICAL_PREFIX + raw_name
                )
                match = _EXPERT_LORA.fullmatch(name)
                if (
                    _clone_only_lora(actor)
                    and match is not None
                    and int(match.group("expert")) < _CLONE_ORIGINAL_EXPERTS
                ):
                    continue
                names.add(name)
                if retain_tensors:
                    value = (
                        weight.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    previous = tensors.get(name)
                    if previous is not None and not torch.equal(previous, value):
                        raise RuntimeError(
                            f"conflicting collective LoRA tensor {name!r}"
                        )
                    tensors[name] = value
    expected = _expected_adapter_names(actor, sides)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise RuntimeError(
            f"collective LoRA export mismatch: missing={missing}, extra={extra}"
        )
    if retain_tensors and _clone_only_lora(actor):
        _validate_clone_canonical_tensors(tensors)
    return tensors


@torch.no_grad()
def export_trainable_state(actor, *, policy_version: int) -> TrainableState | None:
    tensor_parallel = int(getattr(actor.args, "tensor_model_parallel_size", 1))
    pipeline_parallel = int(getattr(actor.args, "pipeline_model_parallel_size", 1))
    expert_parallel = int(getattr(actor.args, "expert_model_parallel_size", 1))
    if (
        tensor_parallel > 1
        or pipeline_parallel > 1
        or expert_parallel > 1
        or _clone_only_lora(actor)
    ):
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
    state = make_trainable_state(
        policy_version,
        tensors,
        train_rollout_kl=metrics.get("train/train_rollout_kl"),
        ess_ratio=metrics.get("train/ess_ratio"),
        pg_clipfrac=metrics.get("train/pg_clipfrac"),
        train_seconds=getattr(actor.args, "_external_train_seconds", None),
    )
    expected_layout = getattr(actor.args, "yeto_rl_layout_hash", None)
    if expected_layout is not None and state.layout_hash != expected_layout:
        raise RuntimeError(
            "exported LoRA layout does not match Yeto's synthesized policy contract"
        )
    return state


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
    incoming = _require_canonical_trainable_state(state)

    side_items = _adapter_sides(actor)
    sides = dict(side_items)
    expected = _expected_adapter_names(actor, side_items)
    if set(incoming.tensors) != expected:
        missing = sorted(expected - set(incoming.tensors))
        extra = sorted(set(incoming.tensors) - expected)
        raise RuntimeError(f"global LoRA mapping mismatch: missing={missing}, extra={extra}")
    if _clone_only_lora(actor):
        _validate_clone_canonical_tensors(incoming.tensors)
    current_layout = getattr(actor.args, "yeto_rl_layout_hash", None)
    if current_layout is None:
        # Standalone Miles fallback. Yeto supplies its synthesized global
        # contract, avoiding a collective data export for a metadata-only hash.
        current = export_trainable_state(actor, policy_version=state.policy_version)
        current_layout = current.layout_hash if current is not None else None
        if dist.is_initialized():
            values = [current_layout]
            dist.broadcast_object_list(values, src=0)
            current_layout = values[0]
    if incoming.layout_hash != current_layout:
        raise RuntimeError("global LoRA layout hash mismatch")

    mapped = {}
    sparse_expert_updates = []
    expert_coordinates = (
        _expert_parallel_coordinates(actor) if _clone_only_lora(actor) else None
    )
    with _optimizer_masters_as_model_parameters(actor):
        for name, side in sides.items():
            if side.param_weight is None:
                continue
            if _clone_only_lora(actor) and _EXPERT_LORA.fullmatch(name):
                sparse_expert_updates.append(
                    _sparse_expert_updates(
                        name,
                        side,
                        incoming.tensors,
                        expert_parallel_rank=expert_coordinates[0],
                        expert_parallel_size=expert_coordinates[1],
                    )
                )
                continue
            else:
                value = incoming.tensors[name]
                target = side.mapping.hf_to_megatron(value, side.megatron_module)
            if target.numel() != side.param_weight.numel():
                raise RuntimeError(f"global LoRA shape mismatch for {name!r}")
            mapped[name] = target.reshape(side.param_weight.shape).contiguous()

        _align_scheduler(actor, state.policy_version)
        _apply_sparse_expert_updates(sparse_expert_updates)
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
    _reset_count = (
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
