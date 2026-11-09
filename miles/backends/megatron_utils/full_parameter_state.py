"""Topology-bound FP32 parameter shards for external DiLoCo synchronization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import torch


_ROLE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_PARAMETER_NAME = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,511}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}\Z")

FULL_PARAMETER_DEFAULT_CHUNK_BYTES = 256 << 20
FULL_PARAMETER_MAX_CHUNK_BYTES = 1 << 30
_FP32_BYTES = 4

_FRAGMENT_HASH_DOMAIN = b"miles-full-parameter-fragment-v1\0"
_CHUNK_HASH_DOMAIN = b"miles-full-parameter-chunk-v1\0"
_SHARD_IDENTITY_DOMAIN = b"miles-full-parameter-chunked-shard-identity-v1\0"


@dataclass(frozen=True, order=True)
class FullParameterTopology:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    ep_rank: int
    ep_size: int
    cp_rank: int
    cp_size: int
    dp_rank: int
    dp_size: int

    def __post_init__(self) -> None:
        for prefix in ("tp", "pp", "ep", "cp", "dp"):
            rank = getattr(self, f"{prefix}_rank")
            size = getattr(self, f"{prefix}_size")
            if isinstance(rank, bool) or not isinstance(rank, int) or isinstance(size, bool) or not isinstance(size, int) or size < 1 or not 0 <= rank < size:
                raise ValueError(f"invalid {prefix} topology coordinates")

    @property
    def shard_id(self) -> str:
        return ".".join(f"{prefix}{getattr(self, f'{prefix}_rank')}-of-{getattr(self, f'{prefix}_size')}" for prefix in ("tp", "pp", "ep", "cp", "dp"))


@dataclass(frozen=True, order=True)
class FullParameterShardSpec:
    role: str
    shard_id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        if not self.shard_id or "::" in self.shard_id:
            raise ValueError("invalid full-parameter shard identity")
        if not _PARAMETER_NAME.fullmatch(self.name) or "::" in self.name:
            raise ValueError(f"invalid full-parameter name: {self.name!r}")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"invalid shape for {self.wire_name!r}")
        if self.dtype != "float32":
            raise ValueError("full-parameter synchronization requires FP32 masters")
        if math.prod(self.shape) != self.numel:
            raise ValueError(f"shape/numel mismatch for {self.wire_name!r}")

    @property
    def wire_name(self) -> str:
        return f"{self.role}::{self.shard_id}::{self.name}"


@dataclass(frozen=True)
class FullParameterShardState:
    policy_version: int
    topology: FullParameterTopology
    layout_hash: str
    specs: tuple[FullParameterShardSpec, ...]
    tensors: Mapping[str, torch.Tensor]
    local_step_generation: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("full-parameter policy version must be non-negative")
        if isinstance(self.local_step_generation, bool) or not isinstance(self.local_step_generation, int) or self.local_step_generation < 0:
            raise ValueError("full-parameter local-step generation must be non-negative")
        if not _SHA256.fullmatch(self.layout_hash):
            raise ValueError("full-parameter layout hash must be a lowercase SHA256")


@dataclass(frozen=True)
class FullParameterLocalStepReceipt:
    """Rank-local proof that an optimizer step advanced a global-policy base."""

    topology: FullParameterTopology
    role: str
    base_policy_version: int
    local_step_generation: int
    rollout_id: int
    optimizer_steps: int
    scheduler_start_steps: int
    scheduler_end_steps: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        for name in (
            "base_policy_version",
            "local_step_generation",
            "rollout_id",
            "optimizer_steps",
            "scheduler_start_steps",
            "scheduler_end_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid full-parameter local-step field: {name}")
        if self.local_step_generation < 1 or self.optimizer_steps < 1:
            raise ValueError("full-parameter local-step receipt has no optimizer progress")
        if self.scheduler_end_steps <= self.scheduler_start_steps:
            raise ValueError("full-parameter scheduler did not advance")


@dataclass(frozen=True)
class FullParameterOptimizerState:
    """Small, topology-bound proof that Adam state survived a policy apply.

    The full optimizer state is intentionally never copied through Ray.  Each
    rank instead fingerprints one deterministic populated parameter state and
    reports aggregate cardinalities for the complete local optimizer.
    """

    topology: FullParameterTopology
    role: str
    installed_policy_version: int
    local_step_generation: int
    last_rollout_id: int
    scheduler_num_steps: int
    populated_parameter_count: int
    optimizer_state_tensor_count: int
    optimizer_state_scalar_count: int
    selected_wire_name: str
    selected_state_sha256: str
    model_master_parameter_count: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        for name in (
            "installed_policy_version",
            "local_step_generation",
            "last_rollout_id",
            "scheduler_num_steps",
            "populated_parameter_count",
            "optimizer_state_tensor_count",
            "optimizer_state_scalar_count",
            "model_master_parameter_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid optimizer-state field: {name}")
        if self.populated_parameter_count < 1:
            raise ValueError("full-parameter optimizer state is empty")
        if self.optimizer_state_tensor_count < 1:
            raise ValueError("full-parameter optimizer has no tensor state")
        if self.optimizer_state_scalar_count < 1:
            raise ValueError("full-parameter optimizer tensor state is empty")
        if self.model_master_parameter_count < 1:
            raise ValueError("full-parameter model/master proof is empty")
        if "::" not in self.selected_wire_name:
            raise ValueError("selected optimizer-state parameter is not canonical")
        if not _SHA256.fullmatch(self.selected_state_sha256):
            raise ValueError("optimizer-state fingerprint must be a lowercase SHA256")


@dataclass(frozen=True)
class FullParameterShardManifest:
    """Small immutable description of one topology-owned FP32 master shard."""

    topology: FullParameterTopology
    role: str
    layout_hash: str
    specs: tuple[FullParameterShardSpec, ...]

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        if not _SHA256.fullmatch(self.layout_hash):
            raise ValueError("full-parameter manifest layout hash is malformed")
        if not self.specs or self.specs != tuple(sorted(self.specs)):
            raise ValueError("full-parameter manifest specs are not canonical")
        names = [spec.wire_name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("full-parameter manifest contains duplicate specs")
        if any(spec.role != self.role or spec.shard_id != self.topology.shard_id for spec in self.specs):
            raise ValueError("full-parameter manifest mixes roles or topology shards")
        if self.layout_hash != _layout_hash(self.topology, self.specs):
            raise ValueError("full-parameter manifest layout hash changed")


@dataclass(frozen=True, order=True)
class FullParameterFragmentPlan:
    """Canonical wire-name order for one topology-owned logical fragment.

    ``flat_ranges`` is empty when every member is a whole parameter.  A
    parameter larger than the fragment byte bound is split into several
    fragments; then ``flat_ranges`` holds one ``(start, stop)`` flat element
    range per wire name, and a whole member uses ``(0, numel)``.
    """

    fragment_id: int
    wire_names: tuple[str, ...]
    numel: int
    flat_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.fragment_id, bool) or not isinstance(self.fragment_id, int) or self.fragment_id < 0:
            raise ValueError("full-parameter fragment ID must be non-negative")
        if not self.wire_names or len(set(self.wire_names)) != len(self.wire_names) or any(not isinstance(name, str) or name.count("::") != 2 or not name for name in self.wire_names):
            raise ValueError("full-parameter fragment wire names are malformed")
        if isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 1:
            raise ValueError("full-parameter fragment must contain scalars")
        if self.flat_ranges:
            if len(self.flat_ranges) != len(self.wire_names) or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or any(isinstance(bound, bool) or not isinstance(bound, int) for bound in item)
                or not 0 <= item[0] < item[1]
                for item in self.flat_ranges
            ):
                raise ValueError("full-parameter fragment flat ranges are malformed")
            if sum(stop - start for start, stop in self.flat_ranges) != self.numel:
                raise ValueError("full-parameter fragment flat ranges do not match its scalar count")

    def members(self, numel_by_name: Mapping[str, int]) -> tuple[tuple[str, int, int], ...]:
        """``(wire_name, start, stop)`` flat element range of every member."""

        if self.flat_ranges:
            return tuple((name, start, stop) for name, (start, stop) in zip(self.wire_names, self.flat_ranges, strict=True))
        return tuple((name, 0, numel_by_name[name]) for name in self.wire_names)

    def identity(self) -> dict[str, object]:
        """Hash payload; a plan without split parameters keeps its old bytes."""

        payload: dict[str, object] = {"fragment_id": self.fragment_id, "wire_names": list(self.wire_names), "numel": self.numel}
        if self.flat_ranges:
            payload["flat_ranges"] = [list(item) for item in self.flat_ranges]
        return payload


@dataclass(frozen=True)
class FullParameterOwnerFragmentPlan:
    """Deterministic fragment plan installed on exactly one topology owner."""

    topology: FullParameterTopology
    parameter_layout_hash: str
    fragments: tuple[FullParameterFragmentPlan, ...]

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.parameter_layout_hash):
            raise ValueError("full-parameter global layout hash is malformed")
        if not self.fragments or self.fragments != tuple(sorted(self.fragments, key=lambda fragment: fragment.fragment_id)):
            raise ValueError("full-parameter owner fragments are not canonical")
        fragment_ids = [fragment.fragment_id for fragment in self.fragments]
        if len(set(fragment_ids)) != len(fragment_ids):
            raise ValueError("full-parameter owner plan has duplicate fragment IDs")
        # A split parameter repeats its wire name only across consecutive
        # members; the manifest check proves the ranges tile the parameter.
        names = [name for fragment in self.fragments for name in fragment.wire_names]
        runs = [name for index, name in enumerate(names) if index == 0 or names[index - 1] != name]
        if len(set(runs)) != len(runs):
            raise ValueError("full-parameter owner plan has duplicate parameters")

    @property
    def plan_hash(self) -> str:
        payload = {
            "schema": 1,
            "topology": asdict(self.topology),
            "parameter_layout_hash": self.parameter_layout_hash,
            "fragments": [fragment.identity() for fragment in self.fragments],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class FullParameterChunkDescriptor:
    chunk_index: int
    flat_offset: int
    numel: int
    payload_hash: str

    def __post_init__(self) -> None:
        for name in ("chunk_index", "flat_offset", "numel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "numel" else 0):
                raise ValueError(f"invalid full-parameter chunk field: {name}")
        if not _SHA256.fullmatch(self.payload_hash):
            raise ValueError("full-parameter chunk hash is malformed")
        if self.numel * _FP32_BYTES >= FULL_PARAMETER_MAX_CHUNK_BYTES:
            raise ValueError("full-parameter chunk exceeds the Ray transport bound")


@dataclass(frozen=True, order=True)
class FullParameterFragmentDescriptor:
    fragment_id: int
    numel: int
    payload_hash: str
    chunks: tuple[FullParameterChunkDescriptor, ...]

    def __post_init__(self) -> None:
        if isinstance(self.fragment_id, bool) or not isinstance(self.fragment_id, int) or self.fragment_id < 0 or isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 1:
            raise ValueError("full-parameter fragment descriptor is malformed")
        if not _SHA256.fullmatch(self.payload_hash):
            raise ValueError("full-parameter fragment hash is malformed")
        if not self.chunks:
            raise ValueError("full-parameter fragment has no chunks")
        offset = 0
        for index, chunk in enumerate(self.chunks):
            if chunk.chunk_index != index or chunk.flat_offset != offset:
                raise ValueError("full-parameter fragment chunks are not contiguous")
            offset += chunk.numel
        if offset != self.numel:
            raise ValueError("full-parameter fragment chunk coverage is incomplete")


@dataclass(frozen=True)
class FullParameterChunkedShardState:
    """Small cut descriptor whose payload stays in nested immutable Ray objects."""

    policy_version: int
    local_step_generation: int
    topology: FullParameterTopology
    parameter_layout_hash: str
    plan_hash: str
    fragments: tuple[FullParameterFragmentDescriptor, ...]
    chunk_refs: list[list[object | None]]

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.policy_version, "policy version")
        _validate_non_negative_integer(
            self.local_step_generation,
            "local-step generation",
        )
        if not _SHA256.fullmatch(self.parameter_layout_hash):
            raise ValueError("full-parameter global layout hash is malformed")
        if not _SHA256.fullmatch(self.plan_hash):
            raise ValueError("full-parameter owner plan hash is malformed")
        if not self.fragments or self.fragments != tuple(sorted(self.fragments, key=lambda fragment: fragment.fragment_id)):
            raise ValueError("full-parameter shard fragments are not canonical")
        if len({fragment.fragment_id for fragment in self.fragments}) != len(self.fragments):
            raise ValueError("full-parameter shard has duplicate fragments")
        _validate_chunk_ref_shape(self, require_refs=True)


@dataclass(frozen=True)
class _PreparedFullParameterChunkedShard:
    commit_token: str
    state: FullParameterChunkedShardState
    parameter_tensor_count: int
    base_policy_version: int
    base_local_step_generation: int
    base_exported_local_step_generation: int
    scheduler_num_steps: int
    state_identity_hash: str
    commit_started: bool
    # Fragments whose target bytes the optimizer masters already hold (same
    # fragment payload hash as the last export or commit): prepare does not
    # read them and commit does not copy them.
    resident_fragment_ids: frozenset[int] = frozenset()
    resident_record: object | None = None


@dataclass(frozen=True)
class _ResidentFullParameterFragments:
    """Which fragment bytes the optimizer masters hold, by fragment payload hash.

    Written by a chunked export (the bytes it hashed are the masters) and by a
    chunked commit (the masters now hold the target).  Valid only while the
    tracking state it was taken at is unchanged: a local step, a restore, a
    legacy apply or the start of a commit drops it."""

    parameter_layout_hash: str
    plan_hash: str
    policy_version: int
    local_step_generation: int
    scheduler_num_steps: int
    payload_hashes: Mapping[int, tuple[int, str]]


@dataclass(frozen=True)
class _FullParameterCommitOutcome:
    """Actor-local journal for an acknowledged or in-doubt commit."""

    commit_token: str
    policy_version: int
    parameter_tensor_count: int
    state_identity_hash: str


@dataclass(frozen=True)
class FullParameterCommitStatus:
    commit_token: str
    status: str
    policy_version: int
    parameter_tensor_count: int
    state_identity_hash: str

    def __post_init__(self) -> None:
        _validate_commit_token(self.commit_token)
        if self.status not in {"ABSENT", "PREPARED", "COMMITTING", "COMMITTED", "FINALIZED"}:
            raise ValueError("full-parameter commit status is invalid")
        _validate_non_negative_integer(self.policy_version, "policy version")
        if isinstance(self.parameter_tensor_count, bool) or not isinstance(self.parameter_tensor_count, int) or self.parameter_tensor_count < 0:
            raise ValueError("full-parameter commit parameter count is invalid")
        if not _SHA256.fullmatch(self.state_identity_hash):
            raise ValueError("full-parameter commit state identity is malformed")


class _OptimizerOwnedParameters(frozenset):
    """Identities of the tensors the base optimizers step, plus distributed-optimizer masters.

    ``distributed_masters`` maps ``id(model_param)`` to ``(owned, fp32)`` for
    Megatron's distributed optimizer when the parameter has no
    ``main_param`` tensor to read:

    * a native-FP32 parameter is stepped through a flat view
      (``shard_fp32_groups``); ``owned`` and ``fp32`` are that view;
    * under the precision-aware optimizer with ``--optimizer-cpu-offload``
      the base optimizer owns the BF16 model view (``shard_float16_groups``)
      and the HybridDeviceOptimizer steps an FP32 copy of it (host-pinned
      when offloaded); ``fp32`` is that copy.

    Empty for the non-distributed optimizer.
    """

    distributed_masters: Mapping[int, tuple[torch.Tensor, torch.Tensor]]

    def __new__(cls, identities, distributed_masters: Mapping[int, tuple[torch.Tensor, torch.Tensor]] | None = None):
        owned = super().__new__(cls, identities)
        owned.distributed_masters = dict(distributed_masters or {})
        return owned


def _resolve_master(
    parameter: torch.nn.Parameter,
    name: str,
    *,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(owned, master)``: the tensor the base optimizer owns and the FP32 master view.

    The non-distributed optimizer keeps a whole FP32 master per parameter;
    both tensors are the same object. Megatron's distributed optimizer keeps a
    flat FP32 range per parameter (``main_param_sharded``); at DP=1 that range
    is the whole parameter, so ``master`` is a parameter-shaped view of it and
    writes through the view land in the tensor that is stepped. A partial
    range (DP>1, or a rank owning none of the parameter) is refused.
    """

    sharded = bool(getattr(parameter, "main_param_sharded", False))
    owned = getattr(parameter, "main_param", None)
    fp32 = owned
    if owned is None:
        distributed = getattr(optimizer_owned_parameter_ids, "distributed_masters", {})
        if id(parameter) in distributed:
            owned, fp32 = distributed[id(parameter)]
            sharded = True
        elif parameter.dtype == torch.float32 and optimizer_owned_parameter_ids is not None and id(parameter) in optimizer_owned_parameter_ids:
            # Megatron keeps parameters such as Qwen3.5 ``A_log`` natively in
            # FP32 and places that same object directly in the base optimizer.
            owned = fp32 = parameter
        elif sharded:
            raise RuntimeError(
                f"{name!r} has no local FP32 master (DP-sharded optimizer, or precision-aware without "
                "CPU offload); the initial DiLoCo profile requires DP=1 and FP32 masters"
            )
        else:
            raise RuntimeError(f"{name!r} has no FP32 optimizer master")
    if not isinstance(fp32, torch.Tensor) or fp32.dtype != torch.float32 or fp32.numel() != parameter.numel():
        if sharded:
            raise RuntimeError(f"{name!r} optimizer master is a partial DP shard; the initial DiLoCo profile requires DP=1")
        raise RuntimeError(f"{name!r} has an incomplete FP32 optimizer master")
    if sharded:
        if fp32.dim() != 1 or not fp32.is_contiguous():
            raise RuntimeError(f"{name!r} distributed-optimizer master is not a flat contiguous range")
        master = fp32.view(parameter.shape)
    else:
        if tuple(fp32.shape) != tuple(parameter.shape):
            raise RuntimeError(f"{name!r} has an incomplete FP32 optimizer master")
        master = fp32
    if optimizer_owned_parameter_ids is not None and id(owned) not in optimizer_owned_parameter_ids:
        raise RuntimeError(f"{name!r} FP32 master is not owned by the optimizer")
    return owned, master


def _master_parameter(
    parameter: torch.nn.Parameter,
    name: str,
    *,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    return _resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned_parameter_ids)[1]


def _specs_and_masters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    role: str,
    topology: FullParameterTopology,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[
    tuple[FullParameterShardSpec, ...],
    dict[str, torch.Tensor],
]:
    if topology.dp_size != 1:
        raise RuntimeError("the initial full-parameter DiLoCo profile requires DP=1")
    masters = {}
    specs = []
    for name, parameter in sorted(named_parameters, key=lambda value: value[0]):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{name!r} is not a trainable parameter")
        master = _master_parameter(
            parameter,
            name,
            optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
        )
        spec = FullParameterShardSpec(
            role,
            topology.shard_id,
            name,
            tuple(int(dimension) for dimension in master.shape),
            "float32",
            master.numel(),
        )
        if spec.wire_name in masters:
            raise RuntimeError(f"duplicate full-parameter name: {spec.wire_name!r}")
        specs.append(spec)
        masters[spec.wire_name] = master
    if not specs:
        raise RuntimeError("full-parameter shard is empty")
    return tuple(specs), masters


def _layout_hash(
    topology: FullParameterTopology,
    specs: Sequence[FullParameterShardSpec],
) -> str:
    payload = {
        "schema": 1,
        "topology": asdict(topology),
        "specs": [asdict(spec) for spec in specs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"full-parameter {name} must be a non-negative integer")
    return value


def _validate_commit_token(commit_token: object) -> str:
    if not isinstance(commit_token, str) or not _COMMIT_TOKEN.fullmatch(commit_token):
        raise ValueError("full-parameter commit token is malformed")
    return commit_token


def _validate_chunk_byte_limit(max_chunk_bytes: object) -> int:
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes < _FP32_BYTES or max_chunk_bytes >= FULL_PARAMETER_MAX_CHUNK_BYTES:
        raise ValueError("full-parameter chunk bytes must be in [4, 1 GiB)")
    return max_chunk_bytes // _FP32_BYTES


def _validate_chunk_ref_shape(
    state: FullParameterChunkedShardState,
    *,
    require_refs: bool,
) -> None:
    if not isinstance(state.chunk_refs, list) or len(state.chunk_refs) != len(state.fragments):
        raise ValueError("full-parameter chunk references do not match fragments")
    for fragment, references in zip(
        state.fragments,
        state.chunk_refs,
        strict=True,
    ):
        if not isinstance(references, list) or len(references) != len(fragment.chunks):
            raise ValueError("full-parameter chunk references are incomplete")
        if require_refs and any(reference is None for reference in references):
            raise ValueError("full-parameter chunk reference was released")


def validate_full_parameter_chunked_shard_state(
    state: FullParameterChunkedShardState,
    *,
    require_refs: bool = True,
) -> FullParameterChunkedShardState:
    if not isinstance(state, FullParameterChunkedShardState):
        raise TypeError("full-parameter chunked shard has the wrong type")
    _validate_non_negative_integer(state.policy_version, "policy version")
    _validate_non_negative_integer(
        state.local_step_generation,
        "local-step generation",
    )
    if not _SHA256.fullmatch(state.parameter_layout_hash) or not _SHA256.fullmatch(state.plan_hash):
        raise ValueError("full-parameter chunked shard identity is malformed")
    if not state.fragments or state.fragments != tuple(sorted(state.fragments, key=lambda fragment: fragment.fragment_id)):
        raise ValueError("full-parameter shard fragments are not canonical")
    for fragment in state.fragments:
        FullParameterFragmentDescriptor(
            fragment.fragment_id,
            fragment.numel,
            fragment.payload_hash,
            fragment.chunks,
        )
    _validate_chunk_ref_shape(state, require_refs=require_refs)
    return state


def _fragment_hasher(
    fragment_id: int,
    parameter_layout_hash: str,
    plan_hash: str,
):
    digest = hashlib.sha256()
    digest.update(_FRAGMENT_HASH_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment_id.to_bytes(8, "little"))
    return digest


def _all_finite(value: torch.Tensor) -> bool:
    """``torch.isfinite(value).all()`` without the full-size bool tensor.

    ``amax``/``amin`` propagate NaN, +inf is the maximum and -inf the minimum,
    so both extremes are finite exactly when every value is.  Two vectorized
    reductions instead of an elementwise pass plus an allocation (about 40x
    faster on 256 MiB of f32 with one thread)."""

    if not value.is_floating_point():
        return True
    if value.numel() == 0:
        return True
    return bool(torch.isfinite(value.amax()).item() and torch.isfinite(value.amin()).item())


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    return memoryview(value.numpy()).cast("B")


def _chunk_hasher(
    fragment_id: int,
    chunk_index: int,
    flat_offset: int,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
):
    """The chunk hash before its value bytes (feed them in any split)."""

    digest = hashlib.sha256()
    digest.update(_CHUNK_HASH_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment_id.to_bytes(8, "little"))
    digest.update(chunk_index.to_bytes(8, "little"))
    digest.update(flat_offset.to_bytes(8, "little"))
    return digest


def _chunk_hash(
    fragment_id: int,
    chunk_index: int,
    flat_offset: int,
    value: torch.Tensor,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
) -> str:
    digest = _chunk_hasher(
        fragment_id,
        chunk_index,
        flat_offset,
        parameter_layout_hash=parameter_layout_hash,
        plan_hash=plan_hash,
    )
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


_HASH_POOL: ThreadPoolExecutor | None = None
_HASH_POOL_LOCK = threading.Lock()


def _hash_pool() -> ThreadPoolExecutor:
    """Two hashing threads (hashlib and torch release the GIL on large buffers)."""

    global _HASH_POOL
    with _HASH_POOL_LOCK:
        if _HASH_POOL is None:
            _HASH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="miles-fp-hash")
        return _HASH_POOL


_SCAN_THREADS_ENV = "MILES_FULL_PARAMETER_HASH_THREADS"
_SCAN_POOL: ThreadPoolExecutor | None = None
_SCAN_POOL_THREADS = 0
# The finite check and both hashes of a deferred export read the masters in
# blocks of this many floats (8 MiB): one pass, cache-resident, no copy.
_SCAN_BLOCK_NUMEL = 1 << 21


def _scan_threads() -> int:
    raw = os.environ.get(_SCAN_THREADS_ENV, "8")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{_SCAN_THREADS_ENV} must be an integer, got {raw!r}") from exc
    if not 1 <= threads <= 64:
        raise RuntimeError(f"{_SCAN_THREADS_ENV} must be within 1..64, got {threads}")
    return threads


def _scan_pool() -> ThreadPoolExecutor:
    """Fragment-parallel workers for deferred exports, materialization and
    prepare verification (``MILES_FULL_PARAMETER_HASH_THREADS``, default 8).

    Fragments are independent hash streams, so the result does not depend on
    the number of workers; hashlib and torch release the GIL on large buffers."""

    global _SCAN_POOL, _SCAN_POOL_THREADS
    threads = _scan_threads()
    with _HASH_POOL_LOCK:
        if _SCAN_POOL is None or _SCAN_POOL_THREADS != threads:
            if _SCAN_POOL is not None:
                _SCAN_POOL.shutdown(wait=False)
            _SCAN_POOL = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="miles-fp-scan")
            _SCAN_POOL_THREADS = threads
        return _SCAN_POOL


class _DeferredFullParameterChunk:
    """The ref of a chunk a deferred export hashed but did not store in Ray.

    A deferred export (``defer_payload=True``) describes every fragment from
    the masters (finite check, chunk hashes, fragment digest) but puts no
    bytes into the object store.  Its chunk refs are this marker.  The bytes
    stay readable only while the masters hold them, which the resident record
    proves: ``materialize_full_parameter_fragments`` then stores a fragment on
    demand, and a skip-resident apply never reads a resident one.  Reading a
    marker anywhere else is an error (``ray.get`` refuses a non-ObjectRef)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFERRED_FULL_PARAMETER_CHUNK"

    def __reduce__(self):
        return (_deferred_full_parameter_chunk, ())


DEFERRED_FULL_PARAMETER_CHUNK = _DeferredFullParameterChunk()


def _deferred_full_parameter_chunk() -> _DeferredFullParameterChunk:
    return DEFERRED_FULL_PARAMETER_CHUNK


def is_deferred_full_parameter_chunk(reference: object) -> bool:
    return isinstance(reference, _DeferredFullParameterChunk)


def _deferred_fragment_ids(state: FullParameterChunkedShardState) -> frozenset[int]:
    return frozenset(
        fragment.fragment_id
        for fragment, references in zip(state.fragments, state.chunk_refs, strict=True)
        if any(is_deferred_full_parameter_chunk(reference) for reference in references)
    )


_RESIDENT_SKIP_ENV = "MILES_FULL_PARAMETER_SKIP_RESIDENT"


def _resident_skip_enabled() -> bool:
    return os.environ.get(_RESIDENT_SKIP_ENV, "1") != "0"


def forget_full_parameter_resident_fragments(actor) -> None:
    """Drop the record of which fragment bytes the masters hold.

    Call after anything writes optimizer masters outside the chunked
    export/commit path (a checkpoint or peer snapshot restore)."""

    actor._external_full_parameter_resident = None


def _record_resident_fragments(
    actor,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
    policy_version: int,
    local_step_generation: int,
    fragments: Sequence[FullParameterFragmentDescriptor],
) -> None:
    actor._external_full_parameter_resident = _ResidentFullParameterFragments(
        parameter_layout_hash=parameter_layout_hash,
        plan_hash=plan_hash,
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        scheduler_num_steps=_scheduler_num_steps(actor),
        payload_hashes={fragment.fragment_id: (fragment.numel, fragment.payload_hash) for fragment in fragments},
    )


def _resident_fragment_ids(
    actor,
    state: FullParameterChunkedShardState,
    *,
    honor_kill_switch: bool = True,
) -> tuple[frozenset[int], object | None]:
    """Fragments of ``state`` whose bytes the masters provably hold now.

    ``honor_kill_switch=False``: ignore ``MILES_FULL_PARAMETER_SKIP_RESIDENT``
    (deferred fragments have no stored bytes; residency is their only source)."""

    record = getattr(actor, "_external_full_parameter_resident", None)
    if (honor_kill_switch and not _resident_skip_enabled()) or not isinstance(record, _ResidentFullParameterFragments):
        return frozenset(), None
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if (
        record.parameter_layout_hash != state.parameter_layout_hash
        or record.plan_hash != state.plan_hash
        or record.policy_version != installed
        or record.local_step_generation != generation
        or exported != generation
        or record.scheduler_num_steps != _scheduler_num_steps(actor)
    ):
        return frozenset(), None
    resident = frozenset(
        fragment.fragment_id
        for fragment in state.fragments
        if record.payload_hashes.get(fragment.fragment_id) == (fragment.numel, fragment.payload_hash)
    )
    return resident, record


def _note_apply_stats(actor, phase: str, **values) -> None:
    stats = dict(getattr(actor, "_external_full_parameter_apply_stats", None) or {})
    stats[phase] = values
    actor._external_full_parameter_apply_stats = stats


def full_parameter_chunked_shard_identity(
    state: FullParameterChunkedShardState,
) -> str:
    """Hash immutable metadata that gives every referenced byte its meaning."""

    validate_full_parameter_chunked_shard_state(state, require_refs=False)
    payload = {
        "policy_version": state.policy_version,
        "local_step_generation": state.local_step_generation,
        "topology": asdict(state.topology),
        "parameter_layout_hash": state.parameter_layout_hash,
        "plan_hash": state.plan_hash,
        "fragments": [
            {
                "fragment_id": fragment.fragment_id,
                "numel": fragment.numel,
                "payload_hash": fragment.payload_hash,
                "chunks": [asdict(chunk) for chunk in fragment.chunks],
            }
            for fragment in state.fragments
        ],
    }
    digest = hashlib.sha256(_SHARD_IDENTITY_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


@torch.no_grad()
def make_full_parameter_shard_state(
    *,
    policy_version: int,
    local_step_generation: int = 0,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> FullParameterShardState:
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    tensors = {}
    for spec in specs:
        value = masters[spec.wire_name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not _all_finite(value):
            raise ValueError(f"{spec.wire_name!r} contains NaN or Inf")
        tensors[spec.wire_name] = value.clone()
    return FullParameterShardState(
        policy_version,
        topology,
        _layout_hash(topology, specs),
        specs,
        tensors,
        local_step_generation,
    )


def validate_full_parameter_shard_state(
    state: FullParameterShardState,
) -> FullParameterShardState:
    """Revalidate a shard after crossing a Ray/object-store boundary."""

    if state.specs != tuple(sorted(state.specs)):
        raise ValueError("full-parameter specs are not canonical")
    expected_names = {spec.wire_name for spec in state.specs}
    if set(state.tensors) != expected_names:
        raise ValueError("full-parameter state is incomplete")
    if state.layout_hash != _layout_hash(state.topology, state.specs):
        raise ValueError("full-parameter state layout hash changed")
    for spec in state.specs:
        tensor = state.tensors[spec.wire_name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous() or tuple(tensor.shape) != spec.shape or not _all_finite(tensor):
            raise ValueError(f"malformed full-parameter tensor {spec.wire_name!r}")
    return state


def _prepare_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    incoming, specs, masters = _validate_full_parameter_shard_metadata(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    staged = []
    for spec in specs:
        target = masters[spec.wire_name]
        source = incoming.tensors[spec.wire_name]
        staged.append((target, source.to(device=target.device, dtype=torch.float32)))
    return staged


def _validate_full_parameter_shard_metadata(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[
    FullParameterShardState,
    tuple[FullParameterShardSpec, ...],
    dict[str, torch.Tensor],
]:
    incoming = validate_full_parameter_shard_state(state)
    if incoming.topology != topology:
        raise RuntimeError("full-parameter topology identity changed")
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    if incoming.specs != specs or incoming.layout_hash != _layout_hash(topology, specs):
        raise RuntimeError("full-parameter local layout changed")
    return incoming, specs, masters


def validate_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> int:
    """Validate one complete shard without changing model or optimizer state."""

    _incoming, specs, _masters = _validate_full_parameter_shard_metadata(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    return len(specs)


@torch.no_grad()
def apply_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> int:
    """Apply a complete shard to optimizer masters without resetting Adam state."""

    staged = _prepare_full_parameter_shard_values(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    for target, source in staged:
        target.copy_(source)
    return len(staged)


def require_full_parameter_actor(args, action: str) -> None:
    """Refuse a LoRA actor unless it opted into the LoRA hybrid profile.

    A hybrid actor streams every ``requires_grad`` parameter (adapters plus
    recipe-unfrozen base parameters); the rest of the profile (DP=1, FP32
    optimizer masters for every trainable parameter) is unchanged.
    """

    from miles.utils.lora.utils import full_parameter_lora_hybrid_enabled, is_lora_enabled

    hybrid = full_parameter_lora_hybrid_enabled(args)
    if not is_lora_enabled(args):
        if hybrid:
            raise RuntimeError("--full-parameter-lora-hybrid requires a LoRA actor (--lora-rank > 0)")
        return
    if not hybrid:
        raise RuntimeError(
            f"full-parameter DiLoCo cannot {action} a LoRA actor without --full-parameter-lora-hybrid"
        )
    if getattr(args, "lora_base_cpu_backup", False):
        raise RuntimeError("--full-parameter-lora-hybrid is incompatible with --lora-base-cpu-backup")


def full_parameter_topology() -> FullParameterTopology:
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    return FullParameterTopology(
        tp_rank=state.tp.rank,
        tp_size=state.tp.size,
        pp_rank=state.pp.rank,
        pp_size=state.pp.size,
        ep_rank=state.ep.rank,
        ep_size=state.ep.size,
        cp_rank=state.cp.rank,
        cp_size=state.cp.size,
        dp_rank=state.intra_dp_cp.rank,
        dp_size=state.intra_dp_cp.size,
    )


def _named_actor_parameters(actor) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers

    values = tuple(
        (name, value)
        for name, value in named_params_and_buffers(
            actor.args,
            actor.model,
            convert_to_global_name=True,
        )
        if isinstance(value, torch.nn.Parameter) and value.requires_grad
    )
    if len({name for name, _parameter in values}) != len(values):
        raise RuntimeError("Megatron produced duplicate full-parameter names")
    return values


def _optimizer_owned_parameter_ids(optimizer) -> frozenset[int]:
    """Return exact parameter objects owned by the non-distributed optimizer."""

    bases = _optimizer_bases(optimizer)
    owned: set[int] = set()
    for base in bases:
        groups = getattr(base, "param_groups", None)
        if not isinstance(groups, list):
            raise RuntimeError("full-parameter optimizer has no parameter groups")
        for group in groups:
            parameters = group.get("params") if isinstance(group, Mapping) else None
            if not isinstance(parameters, list):
                raise RuntimeError("full-parameter optimizer group is malformed")
            for parameter in parameters:
                if not isinstance(parameter, torch.Tensor):
                    raise RuntimeError("optimizer owns a non-tensor parameter")
                identity = id(parameter)
                if identity in owned:
                    raise RuntimeError("optimizer owns a parameter more than once")
                owned.add(identity)
    if not owned:
        raise RuntimeError("full-parameter optimizer owns no parameters")
    return _OptimizerOwnedParameters(owned, _distributed_masters(optimizer))


def _optimizer_wrappers(optimizer) -> tuple[object, ...]:
    wrappers = getattr(optimizer, "chained_optimizers", None)
    return tuple((optimizer,) if wrappers is None else wrappers)


def _paired_groups(wrapper, model_attribute: str, shard_attribute: str):
    model_groups = getattr(wrapper, model_attribute, None)
    shard_groups = getattr(wrapper, shard_attribute, None)
    if model_groups is None and shard_groups is None:
        return
    if model_groups is None or shard_groups is None:
        raise RuntimeError("distributed optimizer parameter groups are malformed")
    for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
        yield from zip(model_group, shard_group, strict=True)


def _distributed_masters(optimizer) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Index the masters of distributed-optimizer parameters that have no ``main_param``."""

    masters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def add(model_param, value):
        if id(model_param) in masters:
            raise RuntimeError("distributed optimizer lists a parameter twice")
        masters[id(model_param)] = value

    for wrapper in _optimizer_wrappers(optimizer):
        for model_param, shard in _paired_groups(wrapper, "model_fp32_groups", "shard_fp32_groups"):
            add(model_param, (shard, shard))
        inner = getattr(getattr(wrapper, "optimizer", None), "param_to_inner_param", None)
        for model_param, shard in _paired_groups(wrapper, "model_float16_groups", "shard_float16_groups"):
            if getattr(model_param, "main_param", None) is not None:
                continue  # read through main_param
            fp32 = None if inner is None or shard is None else inner.get(shard)
            if fp32 is not None and fp32.dtype == torch.float32:
                add(model_param, (shard, fp32))
    return masters


@torch.no_grad()
def _propagate_master_writes(actor) -> int:
    """Make the tensors Megatron's HybridDeviceOptimizer steps agree with written masters.

    Under ``--optimizer-cpu-offload`` each base-optimizer parameter may have an
    inner FP32 copy (``param_to_inner_param``) that the sub-optimizer steps
    and copies back after every step; a write to only one side is undone by
    the next step. The FP32 side is the master the sync wrote, so:

    * an FP32 owned tensor (device master, or native-FP32 view) is copied
      into its inner copy;
    * a BF16 owned tensor (precision-aware: the model's own view) receives
      its FP32 inner copy, which is the master.

    Adam moments are left as they are (the non-distributed path keeps them too).
    """

    pairs = 0
    for base in _optimizer_bases(actor.optimizer):
        inner = getattr(base, "param_to_inner_param", None)
        if inner is None:
            continue
        for owned, copy in inner.items():
            if copy is owned:
                continue
            if copy.numel() != owned.numel():
                raise RuntimeError("hybrid optimizer inner copy has a different size")
            if owned.dtype == torch.float32:
                copy.copy_(owned.view(copy.shape), non_blocking=False)
            else:
                owned.copy_(copy.view(owned.shape), non_blocking=False)
            pairs += 1
    return pairs


def _optimizer_bases(optimizer) -> tuple[object, ...]:
    bases = []
    for wrapper in _optimizer_wrappers(optimizer):
        base = getattr(wrapper, "optimizer", wrapper)
        if any(base is existing for existing in bases):
            raise RuntimeError("full-parameter optimizer base is duplicated")
        bases.append(base)
    if not bases:
        raise RuntimeError("full-parameter optimizer has no base optimizer")
    return tuple(bases)


def _actor_parameter_context(actor):
    return _named_actor_parameters(actor), _optimizer_owned_parameter_ids(actor.optimizer)


def full_parameter_shard_manifest(
    actor,
    *,
    role: str = "actor",
) -> FullParameterShardManifest:
    """Describe this actor rank without copying any parameter payload."""

    require_full_parameter_actor(actor.args, "describe")
    topology = full_parameter_topology()
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    specs, _masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    return FullParameterShardManifest(
        topology=topology,
        role=role,
        layout_hash=_layout_hash(topology, specs),
        specs=specs,
    )


def _validate_fragment_plan_against_manifest(
    plan: FullParameterOwnerFragmentPlan,
    manifest: FullParameterShardManifest,
) -> None:
    if not isinstance(plan, FullParameterOwnerFragmentPlan):
        raise TypeError("full-parameter owner plan has the wrong type")
    if plan.topology != manifest.topology:
        raise RuntimeError("full-parameter owner plan topology changed")
    canonical_names = tuple(spec.wire_name for spec in manifest.specs)
    numel_by_name = {spec.wire_name: spec.numel for spec in manifest.specs}
    planned_names: list[str] = []
    cursor = {}
    for fragment in plan.fragments:
        unknown = [name for name in fragment.wire_names if name not in numel_by_name]
        if unknown:
            raise RuntimeError(f"full-parameter owner plan contains unknown parameter {unknown[0]!r}")
        members = fragment.members(numel_by_name)
        if sum(stop - start for _name, start, stop in members) != fragment.numel:
            raise RuntimeError("full-parameter fragment scalar count changed")
        for name, start, stop in members:
            # Each parameter is covered by consecutive ranges that start at
            # zero and end at its scalar count, in canonical order.
            if start != cursor.get(name, 0) or stop > numel_by_name[name]:
                raise RuntimeError("full-parameter fragment ranges do not tile the parameter")
            if start == 0:
                planned_names.append(name)
            cursor[name] = stop
    if any(cursor.get(name) != numel for name, numel in numel_by_name.items()):
        raise RuntimeError("full-parameter owner plan does not exactly cover the local shard")
    if tuple(planned_names) != canonical_names:
        raise RuntimeError("full-parameter owner plan does not preserve canonical spec order")


def install_full_parameter_fragment_plan(
    actor,
    plan: FullParameterOwnerFragmentPlan,
    *,
    role: str = "actor",
) -> int:
    """Install one immutable exact-coverage owner plan on an actor rank."""

    if getattr(actor, "_external_full_parameter_prepared_cut", None) is not None:
        raise RuntimeError("cannot change a full-parameter plan with a prepared cut")
    manifest = full_parameter_shard_manifest(actor, role=role)
    _validate_fragment_plan_against_manifest(plan, manifest)
    installed = getattr(actor, "_external_full_parameter_fragment_plan", None)
    if installed is None:
        actor._external_full_parameter_fragment_plan = plan
    elif installed != plan:
        raise RuntimeError("full-parameter owner plan is already installed")
    return len(plan.fragments)


def _require_full_parameter_fragment_plan(
    actor,
    *,
    role: str,
) -> tuple[FullParameterOwnerFragmentPlan, FullParameterShardManifest]:
    plan = getattr(actor, "_external_full_parameter_fragment_plan", None)
    if not isinstance(plan, FullParameterOwnerFragmentPlan):
        raise RuntimeError("full-parameter owner plan is not installed")
    manifest = full_parameter_shard_manifest(actor, role=role)
    _validate_fragment_plan_against_manifest(plan, manifest)
    return plan, manifest


# HybridDeviceOptimizer (param_update_in_fp32) mirrors each inner FP32 copy
# into its per-parameter state as ``master_param``: that is the master a
# policy apply overwrites by design (and _validate_model_matches_masters
# checks), not Adam state, so the proof leaves it out.
_HYBRID_MASTER_STATE_KEY = "master_param"


def _adam_state(state: object) -> object:
    if isinstance(state, Mapping) and _HYBRID_MASTER_STATE_KEY in state:
        return {key: value for key, value in state.items() if key != _HYBRID_MASTER_STATE_KEY}
    return state


def _optimizer_state_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"miles-full-parameter-optimizer-state-v1\0")
    if not state:
        raise RuntimeError("selected optimizer parameter has no state")
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise RuntimeError("optimizer state has a non-canonical key")
        encoded_name = name.encode()
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        if isinstance(value, torch.Tensor):
            canonical = value.detach().to(device="cpu").contiguous()
            if canonical.is_floating_point() and not torch.isfinite(canonical).all().item():
                raise RuntimeError("optimizer state contains NaN or Inf")
            dtype = str(canonical.dtype).encode()
            shape = tuple(int(dimension) for dimension in canonical.shape)
            digest.update(b"tensor\0")
            digest.update(len(dtype).to_bytes(4, "little"))
            digest.update(dtype)
            digest.update(len(shape).to_bytes(4, "little"))
            for dimension in shape:
                digest.update(dimension.to_bytes(8, "little"))
            digest.update(memoryview(canonical.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        elif isinstance(value, bool):
            digest.update(b"bool\0" + bytes([value]))
        elif isinstance(value, int):
            digest.update(b"int\0" + struct.pack("<q", value))
        elif isinstance(value, float) and math.isfinite(value):
            digest.update(b"float\0" + struct.pack("<d", value))
        else:
            raise RuntimeError("optimizer state contains an unsupported value")
    return digest.hexdigest()


def full_parameter_optimizer_state(
    actor,
    *,
    role: str = "actor",
    check_model: bool = True,
) -> FullParameterOptimizerState:
    """Return a bounded proof of real optimizer progress on one actor rank.

    ``check_model=False`` skips the whole-model model/master comparison (a
    host-to-device pass over every master).  A caller that brackets a chunked
    apply with two proofs may skip it: the commit ends with the same
    comparison, and it copies every master into the model first.  The
    reported ``model_master_parameter_count`` is the number of compared
    parameters either way."""

    require_full_parameter_actor(actor.args, "inspect")
    assert_full_parameter_boundary_clear(actor)
    installed = getattr(actor, "_external_full_policy_version", None)
    local_step_generation = getattr(
        actor,
        "_external_full_local_step_generation",
        None,
    )
    last_rollout_id = getattr(actor, "_last_rollout_id", None)
    scheduler_num_steps = getattr(actor.opt_param_scheduler, "num_steps", None)
    for name, value in (
        ("installed policy version", installed),
        ("local-step generation", local_step_generation),
        ("last rollout ID", last_rollout_id),
        ("scheduler progress", scheduler_num_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"full-parameter {name} is unavailable")

    topology = full_parameter_topology()
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if check_model:
        model_count = _validate_model_matches_masters(
            named_parameters,
            optimizer_owned_parameter_ids=optimizer_owned,
        )
    else:
        model_count = len(tuple(named_parameters))
        if model_count < 1:
            raise RuntimeError("full-parameter model/master check is empty")
    spec_by_name = {spec.name: spec for spec in specs}
    spec_by_master = {
        id(_resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned)[0]): spec_by_name[name]
        for name, parameter in named_parameters
    }
    populated: dict[str, tuple[FullParameterShardSpec, Mapping[str, object]]] = {}
    tensor_count = 0
    scalar_count = 0
    for base in _optimizer_bases(actor.optimizer):
        states = getattr(base, "state", None)
        if not isinstance(states, Mapping):
            raise RuntimeError("full-parameter optimizer state is unavailable")
        for parameter, state in states.items():
            state = _adam_state(state)
            if not state:
                continue
            spec = spec_by_master.get(id(parameter))
            if spec is None:
                raise RuntimeError("optimizer state belongs to an unknown parameter")
            if spec.wire_name in populated or not isinstance(state, Mapping):
                raise RuntimeError("optimizer parameter state is malformed")
            populated[spec.wire_name] = (spec, state)
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    tensor_count += 1
                    scalar_count += value.numel()
    if not populated:
        raise RuntimeError("full-parameter optimizer has not completed a step")
    selected_spec, selected_state = min(
        populated.values(),
        key=lambda item: (item[0].numel, item[0].wire_name),
    )
    return FullParameterOptimizerState(
        topology=topology,
        role=role,
        installed_policy_version=installed,
        local_step_generation=local_step_generation,
        last_rollout_id=last_rollout_id,
        scheduler_num_steps=scheduler_num_steps,
        populated_parameter_count=len(populated),
        optimizer_state_tensor_count=tensor_count,
        optimizer_state_scalar_count=scalar_count,
        selected_wire_name=selected_spec.wire_name,
        selected_state_sha256=_optimizer_state_sha256(selected_state),
        model_master_parameter_count=model_count,
    )


def _validate_model_matches_masters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    optimizer_owned_parameter_ids: frozenset[int],
) -> int:
    count = 0
    for name, parameter in named_parameters:
        master = _master_parameter(
            parameter,
            name,
            optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
        )
        expected = master.detach().to(device=parameter.device, dtype=parameter.dtype)
        if not torch.equal(parameter.detach(), expected):
            raise RuntimeError(f"{name!r} model value differs from its optimizer master")
        count += 1
    if count < 1:
        raise RuntimeError("full-parameter model/master check is empty")
    return count


def _scheduler_num_steps(actor) -> int:
    value = getattr(actor.opt_param_scheduler, "num_steps", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("full-parameter scheduler progress is unavailable")
    return value


def _optimizer_steps_per_local_round(
    actor,
    *,
    role: str = "actor",
) -> tuple[int, int]:
    optimizer_steps = getattr(actor.args, "num_steps_per_rollout", None)
    global_batch_size = getattr(actor.args, "global_batch_size", None)
    for name, value in (
        ("optimizer steps per rollout", optimizer_steps),
        ("global batch size", global_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"full-parameter {name} is invalid")
    if role == "critic":
        critic_epochs = getattr(actor.args, "num_critic_epochs", 1)
        if (
            isinstance(critic_epochs, bool)
            or not isinstance(critic_epochs, int)
            or critic_epochs < 1
        ):
            raise RuntimeError("full-parameter critic epochs is invalid")
        optimizer_steps *= critic_epochs
    return optimizer_steps, optimizer_steps * global_batch_size


def _validate_scheduler_tracking_congruence(
    actor,
    *,
    role: str = "actor",
) -> None:
    _optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(
        actor,
        role=role,
    )
    _installed, generation, _exported, anchor = _require_full_parameter_tracking(actor)
    if _scheduler_num_steps(actor) != anchor + generation * scheduler_increment:
        raise RuntimeError("full-parameter scheduler progress has no local-step receipt")


def full_parameter_initial_policy_version(args, loaded_rollout_id: int) -> int:
    """Match the external policy identity to Miles' effective train-loop start."""

    if isinstance(loaded_rollout_id, bool) or not isinstance(loaded_rollout_id, int) or loaded_rollout_id < -1:
        raise RuntimeError("full-parameter loaded rollout ID is invalid")
    start = getattr(args, "start_rollout_id", None)
    if start is None:
        return loaded_rollout_id + 1
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise RuntimeError("full-parameter configured rollout start is invalid")
    return start


def initialize_full_parameter_tracking(actor, policy_version: int) -> None:
    """Initialize the global-version boundary from Miles checkpoint progress."""

    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise RuntimeError("full-parameter initial policy version is invalid")
    installed = getattr(actor, "_external_full_policy_version", None)
    if installed is None:
        anchor = _scheduler_num_steps(actor)
        actor._external_full_policy_version = policy_version
        actor._external_full_local_step_generation = 0
        actor._external_full_exported_local_step_generation = 0
        actor._external_full_scheduler_anchor_steps = anchor
        actor._external_full_last_local_rollout_id = None
        actor._external_full_last_local_step_receipt = None
        forget_full_parameter_resident_fragments(actor)
        return
    installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != policy_version:
        raise RuntimeError("full-parameter policy tracking is already initialized")


def rebase_full_parameter_scheduler_anchor(actor, *, reason: str) -> int:
    """Re-anchor scheduler tracking after an external whole-state restore.

    A peer snapshot restore loads the donor's LR scheduler progress into a
    trainer that has taken no local step since tracking began, so the old
    anchor no longer describes it. The restore itself is the proof of the new
    progress (the caller checks it against the donor's digest); this only
    moves the anchor, and only when no local step or policy commit is pending.
    The normal path keeps requiring a local-step receipt for any progress.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("full-parameter scheduler rebase needs an explicit reason")
    assert_full_parameter_boundary_clear(actor)
    _installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if generation != 0 or exported != 0:
        raise RuntimeError("full-parameter scheduler rebase would discard local-step progress")
    if getattr(actor, "_external_full_last_local_step_receipt", None) is not None:
        raise RuntimeError("full-parameter scheduler rebase would discard a local-step receipt")
    anchor = _scheduler_num_steps(actor)
    actor._external_full_scheduler_anchor_steps = anchor
    actor._external_full_scheduler_rebase_reason = reason
    # The restore that needs this rebase wrote the masters.
    forget_full_parameter_resident_fragments(actor)
    return anchor


def _require_full_parameter_tracking(actor) -> tuple[int, int, int, int]:
    """Return initialized tracking state without silently creating it."""

    installed = getattr(actor, "_external_full_policy_version", None)
    generation = getattr(actor, "_external_full_local_step_generation", None)
    exported = getattr(
        actor,
        "_external_full_exported_local_step_generation",
        None,
    )
    anchor = getattr(actor, "_external_full_scheduler_anchor_steps", None)
    for name, value in (
        ("installed policy version", installed),
        ("local-step generation", generation),
        ("exported local-step generation", exported),
        ("scheduler anchor", anchor),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"full-parameter {name} is unavailable")
    if exported > generation:
        raise RuntimeError("full-parameter exported generation is ahead of local progress")
    last_rollout = getattr(actor, "_external_full_last_local_rollout_id", None)
    if last_rollout is not None and (isinstance(last_rollout, bool) or not isinstance(last_rollout, int) or last_rollout < 0):
        raise RuntimeError("full-parameter local rollout progress is unavailable")
    return installed, generation, exported, anchor


def assert_full_parameter_boundary_clear(actor) -> None:
    """Reject any operation that could observe an unresolved policy commit."""

    if getattr(actor, "_external_full_parameter_prepared_cut", None) is not None:
        raise RuntimeError("a prepared full-parameter cut blocks boundary mutation until commit or abort")
    if getattr(actor, "_external_full_parameter_committed_cut", None) is not None:
        raise RuntimeError("an unfinalized full-parameter commit blocks boundary mutation")


def _reject_prepared_full_parameter_boundary_mutation(actor) -> None:
    # Compatibility name for the materialized transport path.
    assert_full_parameter_boundary_clear(actor)


def record_full_parameter_local_step(
    actor,
    *,
    base_policy_version: int,
    rollout_id: int,
    role: str = "actor",
) -> FullParameterLocalStepReceipt:
    """Commit one local-step generation after exact scheduler progress."""

    require_full_parameter_actor(actor.args, "record")
    _reject_prepared_full_parameter_boundary_mutation(actor)
    if not _ROLE.fullmatch(role):
        raise RuntimeError("full-parameter local-step role is invalid")
    installed, generation, _exported, anchor = _require_full_parameter_tracking(actor)
    previous_rollout = actor._external_full_last_local_rollout_id
    if installed != base_policy_version:
        raise RuntimeError("full-parameter local step has a stale global base")
    previous_receipt = getattr(
        actor,
        "_external_full_last_local_step_receipt",
        None,
    )
    if previous_receipt is not None:
        if not isinstance(previous_receipt, FullParameterLocalStepReceipt):
            raise RuntimeError("full-parameter local-step journal is malformed")
        if previous_receipt.base_policy_version == base_policy_version and previous_receipt.rollout_id == rollout_id and previous_receipt.role == role:
            if generation != previous_receipt.local_step_generation or previous_rollout != rollout_id or _scheduler_num_steps(actor) != previous_receipt.scheduler_end_steps:
                raise RuntimeError("full-parameter local-step journal disagrees with actor")
            return previous_receipt
    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0 or getattr(actor, "_last_rollout_id", None) != rollout_id or (previous_rollout is not None and rollout_id <= previous_rollout):
        raise RuntimeError("full-parameter local step has invalid rollout progress")
    optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(
        actor,
        role=role,
    )
    start = anchor + generation * scheduler_increment
    end = _scheduler_num_steps(actor)
    if end != start + scheduler_increment:
        raise RuntimeError("full-parameter local step lacks exact optimizer progress")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    _validate_model_matches_masters(
        named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    generation += 1
    # The local step changed the masters.
    forget_full_parameter_resident_fragments(actor)
    receipt = FullParameterLocalStepReceipt(
        topology=full_parameter_topology(),
        role=role,
        base_policy_version=base_policy_version,
        local_step_generation=generation,
        rollout_id=rollout_id,
        optimizer_steps=optimizer_steps,
        scheduler_start_steps=start,
        scheduler_end_steps=end,
    )
    actor._external_full_local_step_generation = generation
    actor._external_full_last_local_rollout_id = rollout_id
    actor._external_full_last_local_step_receipt = receipt
    return receipt


def export_full_parameter_shard(
    actor,
    *,
    policy_version: int,
    local_step_generation: int = 0,
    role: str = "actor",
) -> FullParameterShardState:
    require_full_parameter_actor(actor.args, "export")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if policy_version != installed:
        raise RuntimeError("full-parameter export policy version is not installed")
    if local_step_generation != generation:
        raise RuntimeError("full-parameter export local-step generation is stale")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    state = make_full_parameter_shard_state(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    actor._external_full_exported_local_step_generation = local_step_generation
    return state


def validate_full_parameter_shard(
    actor,
    state: FullParameterShardState,
    *,
    role: str = "actor",
) -> int:
    require_full_parameter_actor(actor.args, "validate")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed = getattr(actor, "_external_full_policy_version", -1)
    if getattr(actor, "_external_full_exported_local_step_generation", -1) != getattr(
        actor,
        "_external_full_local_step_generation",
        -2,
    ):
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    return validate_full_parameter_shard_values(
        state,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )


@torch.no_grad()
def apply_full_parameter_shard(
    actor,
    state: FullParameterShardState,
    *,
    role: str = "actor",
) -> int:
    require_full_parameter_actor(actor.args, "apply to")
    _reject_prepared_full_parameter_boundary_mutation(actor)
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed = getattr(actor, "_external_full_policy_version", -1)
    if getattr(actor, "_external_full_exported_local_step_generation", -1) != getattr(
        actor,
        "_external_full_local_step_generation",
        -2,
    ):
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    forget_full_parameter_resident_fragments(actor)
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    count = apply_full_parameter_shard_values(
        state,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    _propagate_master_writes(actor)
    _copy_masters_to_model(actor)
    model_count = _validate_model_matches_masters(
        named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if model_count != count:
        raise RuntimeError("full-parameter model/master check is incomplete")
    if role == "actor":
        actor.weights_backuper.backup("actor")
    actor._external_full_policy_version = state.policy_version
    actor._external_full_local_step_generation = 0
    actor._external_full_exported_local_step_generation = 0
    actor._external_full_scheduler_anchor_steps = _scheduler_num_steps(actor)
    actor._external_full_last_local_rollout_id = None
    actor._external_full_last_local_step_receipt = None
    return count


def _resolve_ray_module(ray_module):
    if ray_module is None:
        import ray as ray_module

    for name in ("put", "get"):
        if not callable(getattr(ray_module, name, None)):
            raise TypeError(f"full-parameter Ray module has no {name} operation")
    return ray_module


_ZERO_COPY_ENV = "MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS"
# A chunk read back through Ray's zero-copy path is a tensor over the object
# store's read-only memory; every full-parameter reader only reads it.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)


class _ObjectStoreChunk:
    """The ``ray.put`` form of one CPU chunk: pickled as a numpy array.

    Ray pickles a torch tensor through an in-memory ``torch.save`` on put and
    a ``torch.load`` on get (measured 0.30 and 0.58 GB/s per thread for 256
    MiB of f32).  A numpy array travels as one pickle-5 out-of-band buffer:
    the put is one copy into the object store and the get maps the stored
    bytes, so a reader receives a tensor of the same dtype and shape over the
    object store's memory.  Stored chunks are immutable and every reader only
    reads them (checks, hashes, deltas, copies into the masters)."""

    __slots__ = ("value",)

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def __reduce__(self):
        return (_chunk_from_object_store, (self.value.detach().contiguous().numpy(),))


def _chunk_from_object_store(array) -> torch.Tensor:
    return torch.from_numpy(array)


def put_full_parameter_chunk(ray_module, value: torch.Tensor):
    """``ray_module.put`` of one CPU chunk, in the zero-copy form for the
    real Ray module (``MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS=0`` keeps the
    tensor pickling).  An in-process stand-in stores the tensor itself."""

    if getattr(ray_module, "__name__", None) == "ray" and os.environ.get(_ZERO_COPY_ENV, "1") != "0":
        return ray_module.put(_ObjectStoreChunk(value))
    return ray_module.put(value)


def _validate_cpu_chunk(
    value: object,
    descriptor: FullParameterChunkDescriptor,
    *,
    fragment_id: int,
    verify_values: bool,
    parameter_layout_hash: str,
    plan_hash: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 1 or not value.is_contiguous() or value.numel() != descriptor.numel:
        raise ValueError("full-parameter Ray chunk is malformed")
    if verify_values:
        if not _all_finite(value):
            raise ValueError("full-parameter Ray chunk contains NaN or Inf")
        if descriptor.payload_hash != _chunk_hash(
            fragment_id,
            descriptor.chunk_index,
            descriptor.flat_offset,
            value,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        ):
            raise ValueError("full-parameter Ray chunk hash changed")
    return value


def _export_fragment_chunks(
    fragment: FullParameterFragmentPlan,
    *,
    masters: Mapping[str, torch.Tensor],
    max_chunk_numel: int,
    parameter_layout_hash: str,
    plan_hash: str,
    ray_module,
    overlap_checks: bool = True,
) -> tuple[FullParameterFragmentDescriptor, list[object | None]]:
    """``overlap_checks=False``: run the finite check and the chunk hash in
    this thread (callers that export several fragments on the scan pool)."""

    submit = _hash_pool().submit if overlap_checks else _run_now
    sources = [masters[name].detach().reshape(-1)[start:stop] for name, start, stop in fragment.members({name: masters[name].numel() for name in fragment.wire_names})]
    source_index = 0
    source_offset = 0
    flat_offset = 0
    chunk_descriptors = []
    chunk_refs: list[object | None] = []
    fragment_digest = _fragment_hasher(
        fragment.fragment_id,
        parameter_layout_hash,
        plan_hash,
    )
    while flat_offset < fragment.numel:
        chunk_numel = min(max_chunk_numel, fragment.numel - flat_offset)
        chunk = torch.empty(chunk_numel, device="cpu", dtype=torch.float32)
        destination_offset = 0
        while destination_offset < chunk_numel:
            if source_index >= len(sources):
                raise RuntimeError("full-parameter fragment source ended early")
            source = sources[source_index]
            count = min(
                chunk_numel - destination_offset,
                source.numel() - source_offset,
            )
            chunk[destination_offset : destination_offset + count].copy_(
                source[source_offset : source_offset + count],
                non_blocking=False,
            )
            destination_offset += count
            source_offset += count
            if source_offset == source.numel():
                source_index += 1
                source_offset = 0
        chunk_index = len(chunk_descriptors)
        # The finite check, the chunk hash and the fragment digest read the
        # same immutable chunk; they run concurrently and give the same bytes.
        finite = submit(_all_finite, chunk)
        chunk_hash = submit(
            _chunk_hash,
            fragment.fragment_id,
            chunk_index,
            flat_offset,
            chunk,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        raw = _tensor_bytes(chunk)
        if not finite.result():
            chunk_hash.result()
            raise ValueError("full-parameter optimizer master contains NaN or Inf")
        fragment_digest.update(raw)
        descriptor = FullParameterChunkDescriptor(
            chunk_index=chunk_index,
            flat_offset=flat_offset,
            numel=chunk_numel,
            payload_hash=chunk_hash.result(),
        )
        reference = put_full_parameter_chunk(ray_module, chunk)
        if reference is None:
            raise RuntimeError("Ray returned an empty full-parameter chunk reference")
        chunk_descriptors.append(descriptor)
        chunk_refs.append(reference)
        flat_offset += chunk_numel
        del chunk, raw
    if source_index != len(sources) or source_offset != 0:
        raise RuntimeError("full-parameter fragment did not consume every source")
    return (
        FullParameterFragmentDescriptor(
            fragment_id=fragment.fragment_id,
            numel=fragment.numel,
            payload_hash=fragment_digest.hexdigest(),
            chunks=tuple(chunk_descriptors),
        ),
        chunk_refs,
    )


def _describe_fragment(
    fragment: FullParameterFragmentPlan,
    *,
    masters: Mapping[str, torch.Tensor],
    max_chunk_numel: int,
    parameter_layout_hash: str,
    plan_hash: str,
) -> FullParameterFragmentDescriptor:
    """The descriptor ``_export_fragment_chunks`` returns, hashed in place.

    Same finite check, chunk hashes and fragment digest over the same bytes in
    the same order, read straight from the masters in cache-sized blocks: no
    chunk copy and no Ray object.  SHA-256 is a stream, so feeding a chunk in
    blocks gives the digest of the whole chunk."""

    sources = [masters[name].detach().reshape(-1)[start:stop] for name, start, stop in fragment.members({name: masters[name].numel() for name in fragment.wire_names})]
    source_index = 0
    source_offset = 0
    flat_offset = 0
    chunk_descriptors = []
    fragment_digest = _fragment_hasher(
        fragment.fragment_id,
        parameter_layout_hash,
        plan_hash,
    )
    while flat_offset < fragment.numel:
        chunk_numel = min(max_chunk_numel, fragment.numel - flat_offset)
        chunk_index = len(chunk_descriptors)
        chunk_digest = _chunk_hasher(
            fragment.fragment_id,
            chunk_index,
            flat_offset,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        remaining = chunk_numel
        while remaining:
            if source_index >= len(sources):
                raise RuntimeError("full-parameter fragment source ended early")
            source = sources[source_index]
            count = min(remaining, source.numel() - source_offset, _SCAN_BLOCK_NUMEL)
            block = source[source_offset : source_offset + count]
            if block.device.type != "cpu" or block.dtype != torch.float32 or not block.is_contiguous():
                block = block.to(device="cpu", dtype=torch.float32).contiguous()
            if not _all_finite(block):
                raise ValueError("full-parameter optimizer master contains NaN or Inf")
            raw = _tensor_bytes(block)
            chunk_digest.update(raw)
            fragment_digest.update(raw)
            del raw, block
            remaining -= count
            source_offset += count
            if source_offset == source.numel():
                source_index += 1
                source_offset = 0
        chunk_descriptors.append(
            FullParameterChunkDescriptor(
                chunk_index=chunk_index,
                flat_offset=flat_offset,
                numel=chunk_numel,
                payload_hash=chunk_digest.hexdigest(),
            )
        )
        flat_offset += chunk_numel
    if source_index != len(sources) or source_offset != 0:
        raise RuntimeError("full-parameter fragment did not consume every source")
    return FullParameterFragmentDescriptor(
        fragment_id=fragment.fragment_id,
        numel=fragment.numel,
        payload_hash=fragment_digest.hexdigest(),
        chunks=tuple(chunk_descriptors),
    )


class _Done:
    """A finished future (``_hash_pool().submit`` without the pool)."""

    __slots__ = ("_value", "_error")

    def __init__(self, value=None, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._value


def _run_now(function, *args, **kwargs) -> _Done:
    try:
        return _Done(function(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 - raised by result(), like a future
        return _Done(error=error)


def _parallel_map(function, items: Sequence) -> list:
    """``[function(item) for item in items]`` on the scan pool, in order.

    Every item is awaited before an error propagates, so no worker still
    touches the masters (or the object store) when the caller moves on."""

    if len(items) <= 1:
        return [function(item) for item in items]
    futures = [_scan_pool().submit(function, item) for item in items]
    results, failure = [], None
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as error:  # noqa: BLE001 - re-raised after every worker ended
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return results


def export_full_parameter_chunked_shard(
    actor,
    *,
    policy_version: int,
    local_step_generation: int = 0,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    role: str = "actor",
    ray_module=None,
    defer_payload: bool = False,
) -> FullParameterChunkedShardState:
    """Export bounded owner chunks without materializing a rank or global cut.

    ``defer_payload``: describe every fragment exactly as the full export
    does (same descriptors, same hashes, same NaN/Inf refusal) but store no
    bytes in Ray; every chunk ref is ``DEFERRED_FULL_PARAMETER_CHUNK``.
    ``materialize_full_parameter_fragments`` stores the fragments a caller
    needs while the masters still hold them."""

    require_full_parameter_actor(actor.args, "export")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    _validate_non_negative_integer(policy_version, "policy version")
    _validate_non_negative_integer(
        local_step_generation,
        "local-step generation",
    )
    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if policy_version != installed:
        raise RuntimeError("full-parameter export policy version is not installed")
    if local_step_generation != generation:
        raise RuntimeError("full-parameter export local-step generation is stale")
    started = time.monotonic()
    max_chunk_numel = _validate_chunk_byte_limit(max_chunk_bytes)
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest = _require_full_parameter_fragment_plan(actor, role=role)
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=manifest.topology,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if specs != manifest.specs:
        raise RuntimeError("full-parameter local specs changed after plan installation")
    fragment_descriptors = []
    chunk_refs = []
    if defer_payload:
        fragment_descriptors = _parallel_map(
            lambda fragment: _describe_fragment(
                fragment,
                masters=masters,
                max_chunk_numel=max_chunk_numel,
                parameter_layout_hash=plan.parameter_layout_hash,
                plan_hash=plan.plan_hash,
            ),
            plan.fragments,
        )
        chunk_refs = [[DEFERRED_FULL_PARAMETER_CHUNK] * len(descriptor.chunks) for descriptor in fragment_descriptors]
    for fragment in () if defer_payload else plan.fragments:
        descriptor, references = _export_fragment_chunks(
            fragment,
            masters=masters,
            max_chunk_numel=max_chunk_numel,
            parameter_layout_hash=plan.parameter_layout_hash,
            plan_hash=plan.plan_hash,
            ray_module=ray_module,
        )
        fragment_descriptors.append(descriptor)
        chunk_refs.append(references)
    state = FullParameterChunkedShardState(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        topology=manifest.topology,
        parameter_layout_hash=plan.parameter_layout_hash,
        plan_hash=plan.plan_hash,
        fragments=tuple(fragment_descriptors),
        chunk_refs=chunk_refs,
    )
    actor._external_full_exported_local_step_generation = local_step_generation
    _record_resident_fragments(
        actor,
        parameter_layout_hash=plan.parameter_layout_hash,
        plan_hash=plan.plan_hash,
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        fragments=fragment_descriptors,
    )
    _note_apply_stats(
        actor,
        "export",
        fragments=len(fragment_descriptors),
        deferred=bool(defer_payload),
        seconds=time.monotonic() - started,
    )
    return state


def materialize_full_parameter_fragments(
    actor,
    *,
    state: FullParameterChunkedShardState,
    fragment_ids: Sequence[int],
    role: str = "actor",
    ray_module=None,
) -> tuple[tuple[int, list[object]], ...]:
    """Store the bytes of deferred fragments of ``state`` while the masters hold them.

    Each requested fragment must be resident: its payload hash equals the one
    the live resident record holds (same tracking state: no local step,
    restore or commit start since the export or commit that recorded it).
    The fragment is exported again from the masters with the descriptor's
    chunk geometry and its descriptor must come out identical, so the stored
    bytes are exactly the bytes the deferred export hashed.  Returns
    ``(fragment_id, chunk refs)`` in fragment order; the masters, the record
    and the tracking state do not change."""

    require_full_parameter_actor(actor.args, "materialize")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    requested = []
    for fragment_id in fragment_ids:
        if isinstance(fragment_id, bool) or not isinstance(fragment_id, int) or fragment_id < 0:
            raise ValueError("full-parameter materialize fragment ID is malformed")
        requested.append(fragment_id)
    if len(set(requested)) != len(requested):
        raise ValueError("full-parameter materialize repeats a fragment")
    if not requested:
        return ()
    started = time.monotonic()
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest = _require_full_parameter_fragment_plan(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    descriptors = {descriptor.fragment_id: descriptor for descriptor in state.fragments}
    fragments = {fragment.fragment_id: fragment for fragment in plan.fragments}
    if any(fragment_id not in descriptors for fragment_id in requested):
        raise ValueError("full-parameter materialize names a fragment this rank does not own")
    resident, _record = _resident_fragment_ids(actor, state, honor_kill_switch=False)
    stale = sorted(fragment_id for fragment_id in requested if fragment_id not in resident)
    if stale:
        raise RuntimeError(f"full-parameter fragments {stale} are no longer held by the masters; a deferred export cannot be materialized after they changed")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=manifest.topology,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if specs != manifest.specs:
        raise RuntimeError("full-parameter local specs changed after plan installation")

    def store(fragment_id: int) -> tuple[int, list[object]]:
        expected = descriptors[fragment_id]
        descriptor, references = _export_fragment_chunks(
            fragments[fragment_id],
            masters=masters,
            max_chunk_numel=expected.chunks[0].numel,
            parameter_layout_hash=plan.parameter_layout_hash,
            plan_hash=plan.plan_hash,
            ray_module=ray_module,
            overlap_checks=len(requested) <= 1,
        )
        if descriptor != expected:
            raise RuntimeError(f"full-parameter fragment {fragment_id} bytes changed since its deferred export")
        return fragment_id, references

    stored = _parallel_map(store, sorted(requested))
    _note_apply_stats(
        actor,
        "materialize",
        fragments=len(stored),
        seconds=time.monotonic() - started,
    )
    return tuple(stored)


def _validate_chunked_shard_against_plan(
    state: FullParameterChunkedShardState,
    plan: FullParameterOwnerFragmentPlan,
    manifest: FullParameterShardManifest,
) -> None:
    validate_full_parameter_chunked_shard_state(state)
    if state.topology != manifest.topology or state.topology != plan.topology:
        raise RuntimeError("full-parameter chunked shard topology changed")
    if state.parameter_layout_hash != plan.parameter_layout_hash or state.plan_hash != plan.plan_hash:
        raise RuntimeError("full-parameter chunked shard plan identity changed")
    if len(state.fragments) != len(plan.fragments):
        raise RuntimeError("full-parameter chunked shard fragment count changed")
    for descriptor, fragment in zip(state.fragments, plan.fragments, strict=True):
        if descriptor.fragment_id != fragment.fragment_id or descriptor.numel != fragment.numel:
            raise RuntimeError("full-parameter chunked fragment identity changed")


def prepare_full_parameter_chunked_shard(
    actor,
    state: FullParameterChunkedShardState,
    *,
    commit_token: str,
    role: str = "actor",
    ray_module=None,
    skip_resident: bool = False,
) -> int:
    """Validate every immutable chunk and stage only its refs for later commit.

    ``skip_resident``: the caller states that it built ``state`` from the cut
    this actor last exported or committed, changing only some fragments.  A
    fragment whose payload hash equals the one the actor recorded for its
    masters at that export/commit is then neither read nor copied: the masters
    already hold those bytes.  The record is dropped by every local step,
    restore rebase and legacy apply, so the skip never applies across one."""

    require_full_parameter_actor(actor.args, "prepare")
    token = _validate_commit_token(commit_token)
    state_identity_hash = full_parameter_chunked_shard_identity(state)
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    for outcome in (committed, finalized):
        if outcome is not None and not isinstance(outcome, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
    if isinstance(committed, _FullParameterCommitOutcome):
        if committed.commit_token == token and committed.policy_version == state.policy_version and committed.state_identity_hash == state_identity_hash:
            return committed.parameter_tensor_count
        raise RuntimeError("an unfinalized full-parameter commit must be reconciled")
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token and finalized.policy_version == state.policy_version and finalized.state_identity_hash == state_identity_hash):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    if prepared is not None:
        if not isinstance(prepared, _PreparedFullParameterChunkedShard):
            raise RuntimeError("full-parameter prepared context is malformed")
        if prepared.commit_token == token and prepared.state_identity_hash == state_identity_hash:
            return prepared.parameter_tensor_count
        raise RuntimeError("a stale full-parameter prepared cut must be aborted")
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    if exported != generation:
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest = _require_full_parameter_fragment_plan(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    started = time.monotonic()
    deferred = _deferred_fragment_ids(state)
    resident, resident_record = (
        _resident_fragment_ids(actor, state) if skip_resident else (frozenset(), None)
    )
    if deferred - resident:
        # A deferred fragment has no stored bytes: only the masters hold them.
        # Its residency does not depend on the skip switch.
        forced, forced_record = _resident_fragment_ids(actor, state, honor_kill_switch=False)
        missing = sorted(deferred - forced)
        if missing:
            raise RuntimeError(f"deferred full-parameter fragments {missing} are not held by the masters and have no stored bytes")
        if resident_record is None:
            resident_record = forced_record
        resident = resident | deferred
    rows = [
        (fragment, references)
        for fragment, references in zip(state.fragments, state.chunk_refs, strict=True)
        if fragment.fragment_id not in resident
    ]
    # One fragment: its finite check and chunk hash overlap its digest on the
    # hash pool.  Several: each scan worker verifies one fragment serially.
    overlap_checks = len(rows) <= 1

    def verify(row) -> None:
        fragment, references = row
        fragment_digest = _fragment_hasher(
            fragment.fragment_id,
            state.parameter_layout_hash,
            state.plan_hash,
        )
        for descriptor, reference in zip(fragment.chunks, references, strict=True):
            raw = ray_module.get(reference)
            checked = (_hash_pool().submit if overlap_checks else _run_now)(
                _validate_cpu_chunk,
                raw,
                descriptor,
                fragment_id=fragment.fragment_id,
                verify_values=True,
                parameter_layout_hash=state.parameter_layout_hash,
                plan_hash=state.plan_hash,
            )
            if isinstance(raw, torch.Tensor) and raw.device.type == "cpu" and raw.dtype == torch.float32 and raw.ndim == 1 and raw.is_contiguous():
                fragment_digest.update(_tensor_bytes(raw))
                checked.result()
            else:
                fragment_digest.update(_tensor_bytes(checked.result()))
            del raw, checked
        if fragment_digest.hexdigest() != fragment.payload_hash:
            raise ValueError("full-parameter fragment hash changed")

    # Resident fragments: the masters already hold these exact bytes; commit
    # skips them.  The others are independent; verify them concurrently.
    _parallel_map(verify, rows)
    _note_apply_stats(
        actor,
        "prepare",
        fragments=len(state.fragments),
        resident_fragments=len(resident),
        read_fragments=len(state.fragments) - len(resident),
        seconds=time.monotonic() - started,
    )
    parameter_count = len(manifest.specs)
    actor._external_full_parameter_prepared_cut = _PreparedFullParameterChunkedShard(
        commit_token=token,
        state=state,
        parameter_tensor_count=parameter_count,
        base_policy_version=installed,
        base_local_step_generation=generation,
        base_exported_local_step_generation=exported,
        scheduler_num_steps=_scheduler_num_steps(actor),
        state_identity_hash=state_identity_hash,
        commit_started=False,
        resident_fragment_ids=resident,
        resident_record=resident_record,
    )
    return parameter_count


def _copy_fragment_chunks_to_masters(
    fragment: FullParameterFragmentPlan,
    descriptor: FullParameterFragmentDescriptor,
    references: Sequence[object | None],
    *,
    masters: Mapping[str, torch.Tensor],
    parameter_layout_hash: str,
    plan_hash: str,
    ray_module,
) -> int:
    master_index = 0
    master_offset = 0
    copied_parameters = 0
    members = fragment.members({name: masters[name].numel() for name in fragment.wire_names})
    flattened = [masters[name].detach().reshape(-1)[start:stop] for name, start, stop in members]
    # A split parameter counts once: in the fragment holding its last range.
    completes = [stop == masters[name].numel() for name, _start, stop in members]
    for chunk_descriptor, reference in zip(
        descriptor.chunks,
        references,
        strict=True,
    ):
        if is_deferred_full_parameter_chunk(reference):
            raise RuntimeError("a deferred full-parameter chunk has no stored bytes to copy")
        value = _validate_cpu_chunk(
            ray_module.get(reference),
            chunk_descriptor,
            fragment_id=fragment.fragment_id,
            verify_values=False,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        source_offset = 0
        while source_offset < value.numel():
            if master_index >= len(flattened):
                raise RuntimeError("full-parameter chunk target ended early")
            master = flattened[master_index]
            count = min(
                value.numel() - source_offset,
                master.numel() - master_offset,
            )
            master[master_offset : master_offset + count].copy_(
                value[source_offset : source_offset + count],
                non_blocking=False,
            )
            source_offset += count
            master_offset += count
            if master_offset == master.numel():
                copied_parameters += completes[master_index]
                master_index += 1
                master_offset = 0
        del value
    if master_index != len(flattened) or master_offset != 0:
        raise RuntimeError("full-parameter chunks did not cover every target")
    return copied_parameters


def _fragment_completed_parameters(
    fragment: FullParameterFragmentPlan,
    masters: Mapping[str, torch.Tensor],
) -> int:
    """Parameters whose last flat range lies in ``fragment`` (as the copy counts them)."""

    members = fragment.members({name: masters[name].numel() for name in fragment.wire_names})
    return sum(stop == masters[name].numel() for name, _start, stop in members)


@torch.no_grad()
def commit_prepared_full_parameter_shard(
    actor,
    *,
    commit_token: str,
    role: str = "actor",
    ray_module=None,
) -> int:
    """Commit a fully prepared cut while retaining Adam state objects."""


    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    for outcome in (committed, finalized):
        if outcome is not None and not isinstance(outcome, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
    if isinstance(committed, _FullParameterCommitOutcome):
        if committed.commit_token != token:
            raise RuntimeError("another full-parameter commit is unresolved")
        installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != committed.policy_version or generation != 0 or exported != 0:
            raise RuntimeError("committed full-parameter journal disagrees with actor")
        return committed.parameter_tensor_count
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    if not isinstance(prepared, _PreparedFullParameterChunkedShard):
        raise RuntimeError("no full-parameter cut is prepared")
    if prepared.commit_token != token:
        raise RuntimeError("full-parameter commit token does not match prepared state")
    state = prepared.state
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != prepared.base_policy_version or generation != prepared.base_local_step_generation or exported != prepared.base_exported_local_step_generation or exported != generation or _scheduler_num_steps(actor) != prepared.scheduler_num_steps or state.policy_version <= installed:
        raise RuntimeError("prepared full-parameter cut became stale")
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest = _require_full_parameter_fragment_plan(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=manifest.topology,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if specs != manifest.specs:
        raise RuntimeError("full-parameter local specs changed after prepare")
    if not prepared.commit_started:
        # Resident fragments are skipped only while the masters provably hold
        # them: the record taken at prepare must still be the live one.  A
        # retried commit (commit_started) wrote only non-resident fragments,
        # which never overlap a resident one, so the skip stays exact.
        if prepared.resident_fragment_ids and (
            getattr(actor, "_external_full_parameter_resident", None) is not prepared.resident_record
        ):
            raise RuntimeError("full-parameter resident fragments changed after prepare")
        prepared = replace(prepared, commit_started=True)
        actor._external_full_parameter_prepared_cut = prepared
    # From here the masters may be partly written: no record describes them.
    forget_full_parameter_resident_fragments(actor)
    started = time.monotonic()
    count = 0
    copied_fragments = 0
    for fragment, descriptor, references in zip(
        plan.fragments,
        state.fragments,
        state.chunk_refs,
        strict=True,
    ):
        if fragment.fragment_id in prepared.resident_fragment_ids:
            count += _fragment_completed_parameters(fragment, masters)
            continue
        copied_fragments += 1
        count += _copy_fragment_chunks_to_masters(
            fragment,
            descriptor,
            references,
            masters=masters,
            parameter_layout_hash=state.parameter_layout_hash,
            plan_hash=state.plan_hash,
            ray_module=ray_module,
        )
    if count != prepared.parameter_tensor_count or count != len(specs):
        raise RuntimeError("full-parameter chunked apply is incomplete")
    copied_seconds = time.monotonic() - started
    _propagate_master_writes(actor)
    _copy_masters_to_model(actor)
    model_count = _validate_model_matches_masters(
        named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if model_count != count:
        raise RuntimeError("full-parameter model/master check is incomplete")
    if role == "actor":
        actor.weights_backuper.backup("actor")
    actor._external_full_policy_version = state.policy_version
    actor._external_full_local_step_generation = 0
    actor._external_full_exported_local_step_generation = 0
    actor._external_full_scheduler_anchor_steps = _scheduler_num_steps(actor)
    actor._external_full_last_local_rollout_id = None
    actor._external_full_last_local_step_receipt = None
    actor._external_full_parameter_committed_cut = _FullParameterCommitOutcome(
        commit_token=token,
        policy_version=state.policy_version,
        parameter_tensor_count=count,
        state_identity_hash=prepared.state_identity_hash,
    )
    _record_resident_fragments(
        actor,
        parameter_layout_hash=state.parameter_layout_hash,
        plan_hash=state.plan_hash,
        policy_version=state.policy_version,
        local_step_generation=0,
        fragments=state.fragments,
    )
    _note_apply_stats(
        actor,
        "commit",
        fragments=len(state.fragments),
        copied_fragments=copied_fragments,
        resident_fragments=len(prepared.resident_fragment_ids),
        copy_seconds=copied_seconds,
        seconds=time.monotonic() - started,
    )
    del actor._external_full_parameter_prepared_cut
    return count


def finalize_full_parameter_commit(actor, *, commit_token: str) -> int:
    """Acknowledge a group-wide commit after every rank has converged."""

    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    if not isinstance(committed, _FullParameterCommitOutcome):
        raise RuntimeError("no committed full-parameter cut is pending finalization")
    if committed.commit_token != token:
        raise RuntimeError("full-parameter finalize token does not match journal")
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != committed.policy_version or generation != 0 or exported != 0:
        raise RuntimeError("committed full-parameter journal disagrees with actor")
    actor._external_full_parameter_last_finalized_cut = committed
    del actor._external_full_parameter_committed_cut
    return committed.parameter_tensor_count


def full_parameter_commit_status(
    actor,
    *,
    commit_token: str,
    policy_version: int,
    state_identity_hash: str,
) -> FullParameterCommitStatus:
    """Attest whether a transaction is safe to abort or must be reconciled."""

    token = _validate_commit_token(commit_token)
    _validate_non_negative_integer(policy_version, "policy version")
    if not _SHA256.fullmatch(state_identity_hash):
        raise ValueError("full-parameter commit state identity is malformed")
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    if prepared is not None:
        if not isinstance(prepared, _PreparedFullParameterChunkedShard):
            raise RuntimeError("full-parameter prepared context is malformed")
        if prepared.commit_token != token or prepared.state.policy_version != policy_version or prepared.state_identity_hash != state_identity_hash:
            raise RuntimeError("another full-parameter transaction is active")
        return FullParameterCommitStatus(
            token,
            "COMMITTING" if prepared.commit_started else "PREPARED",
            policy_version,
            prepared.parameter_tensor_count,
            state_identity_hash,
        )
    if committed is not None:
        if not isinstance(committed, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
        if committed.commit_token != token or committed.policy_version != policy_version or committed.state_identity_hash != state_identity_hash:
            raise RuntimeError("another full-parameter transaction is active")
        return FullParameterCommitStatus(
            token,
            "COMMITTED",
            policy_version,
            committed.parameter_tensor_count,
            state_identity_hash,
        )
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token and finalized.policy_version == policy_version and finalized.state_identity_hash == state_identity_hash):
        return FullParameterCommitStatus(
            token,
            "FINALIZED",
            policy_version,
            finalized.parameter_tensor_count,
            state_identity_hash,
        )
    return FullParameterCommitStatus(
        token,
        "ABSENT",
        policy_version,
        0,
        state_identity_hash,
    )


def abort_prepared_full_parameter_shard(
    actor,
    *,
    commit_token: str,
) -> bool:
    """Discard actor-local prepare metadata without deleting shared Ray objects."""

    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    if committed is not None:
        if not isinstance(committed, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
        if committed.commit_token == token:
            raise RuntimeError("a committed full-parameter cut cannot be aborted")
        raise RuntimeError("another full-parameter commit is unresolved")
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    if prepared is None:
        return False
    if not isinstance(prepared, _PreparedFullParameterChunkedShard):
        raise RuntimeError("full-parameter prepared context is malformed")
    if prepared.commit_token != token:
        raise RuntimeError("full-parameter abort token does not match prepared state")
    if prepared.commit_started:
        raise RuntimeError("a committing full-parameter cut cannot be aborted")
    del actor._external_full_parameter_prepared_cut
    return True


def _copy_masters_to_model(actor) -> None:
    """Copy every FP32 optimizer master back into its model parameter."""

    children = getattr(actor.optimizer, "chained_optimizers", (actor.optimizer,))
    for child in children:
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("Megatron optimizer lacks main-to-model copy")
        copy()
