"""Topology-bound FP32 parameter shards for external DiLoCo synchronization."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
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
    """Canonical wire-name order for one topology-owned logical fragment."""

    fragment_id: int
    wire_names: tuple[str, ...]
    numel: int

    def __post_init__(self) -> None:
        if isinstance(self.fragment_id, bool) or not isinstance(self.fragment_id, int) or self.fragment_id < 0:
            raise ValueError("full-parameter fragment ID must be non-negative")
        if not self.wire_names or len(set(self.wire_names)) != len(self.wire_names) or any(not isinstance(name, str) or name.count("::") != 2 or not name for name in self.wire_names):
            raise ValueError("full-parameter fragment wire names are malformed")
        if isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 1:
            raise ValueError("full-parameter fragment must contain scalars")


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
        names = [name for fragment in self.fragments for name in fragment.wire_names]
        if len(set(names)) != len(names):
            raise ValueError("full-parameter owner plan has duplicate parameters")

    @property
    def plan_hash(self) -> str:
        payload = {
            "schema": 1,
            "topology": asdict(self.topology),
            "parameter_layout_hash": self.parameter_layout_hash,
            "fragments": [asdict(fragment) for fragment in self.fragments],
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


def _master_parameter(
    parameter: torch.nn.Parameter,
    name: str,
    *,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    master = getattr(parameter, "main_param", None)
    if master is None:
        if getattr(parameter, "main_param_sharded", False):
            raise RuntimeError(f"{name!r} has a DP-sharded optimizer master; the initial DiLoCo profile requires DP=1")
        if parameter.dtype == torch.float32 and optimizer_owned_parameter_ids is not None and id(parameter) in optimizer_owned_parameter_ids:
            # Megatron keeps parameters such as Qwen3.5 ``A_log`` natively in
            # FP32 and places that same object directly in the base optimizer.
            master = parameter
        else:
            raise RuntimeError(f"{name!r} has no FP32 optimizer master")
    if not isinstance(master, torch.Tensor) or master.dtype != torch.float32 or master.numel() != parameter.numel() or tuple(master.shape) != tuple(parameter.shape):
        raise RuntimeError(f"{name!r} has an incomplete FP32 optimizer master")
    if optimizer_owned_parameter_ids is not None and id(master) not in optimizer_owned_parameter_ids:
        raise RuntimeError(f"{name!r} FP32 master is not owned by the optimizer")
    return master


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


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    return memoryview(value.numpy()).cast("B")


def _chunk_hash(
    fragment_id: int,
    chunk_index: int,
    flat_offset: int,
    value: torch.Tensor,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_CHUNK_HASH_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment_id.to_bytes(8, "little"))
    digest.update(chunk_index.to_bytes(8, "little"))
    digest.update(flat_offset.to_bytes(8, "little"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


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
        if not torch.isfinite(value).all().item():
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
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous() or tuple(tensor.shape) != spec.shape or not torch.isfinite(tensor).all().item():
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
    from miles.backends.megatron_utils.update_weight.common import (
        named_params_and_buffers,
    )

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
    return frozenset(owned)


def _optimizer_bases(optimizer) -> tuple[object, ...]:
    wrappers = getattr(optimizer, "chained_optimizers", None)
    if wrappers is None:
        wrappers = (optimizer,)
    bases = []
    for wrapper in wrappers:
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

    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot describe a LoRA actor")
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
    canonical_index = {name: index for index, name in enumerate(canonical_names)}
    planned_names = []
    specs_by_name = {spec.wire_name: spec for spec in manifest.specs}
    for fragment in plan.fragments:
        unknown = [name for name in fragment.wire_names if name not in specs_by_name]
        if unknown:
            raise RuntimeError(f"full-parameter owner plan contains unknown parameter {unknown[0]!r}")
        indices = [canonical_index[name] for name in fragment.wire_names]
        if indices != sorted(indices):
            raise RuntimeError("full-parameter fragment does not preserve canonical spec order")
        actual_numel = sum(specs_by_name[name].numel for name in fragment.wire_names)
        if actual_numel != fragment.numel:
            raise RuntimeError("full-parameter fragment scalar count changed")
        planned_names.extend(fragment.wire_names)
    if len(planned_names) != len(canonical_names) or set(planned_names) != set(canonical_names):
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
) -> FullParameterOptimizerState:
    """Return a bounded proof of real optimizer progress on one actor rank."""

    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot inspect a LoRA actor")
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
    model_count = _validate_model_matches_masters(
        named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    spec_by_master = {id(masters[spec.wire_name]): spec for spec in specs}
    populated: dict[str, tuple[FullParameterShardSpec, Mapping[str, object]]] = {}
    tensor_count = 0
    scalar_count = 0
    for base in _optimizer_bases(actor.optimizer):
        states = getattr(base, "state", None)
        if not isinstance(states, Mapping):
            raise RuntimeError("full-parameter optimizer state is unavailable")
        for parameter, state in states.items():
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


def _optimizer_steps_per_local_round(actor) -> tuple[int, int]:
    optimizer_steps = getattr(actor.args, "num_steps_per_rollout", None)
    global_batch_size = getattr(actor.args, "global_batch_size", None)
    for name, value in (
        ("optimizer steps per rollout", optimizer_steps),
        ("global batch size", global_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"full-parameter {name} is invalid")
    return optimizer_steps, optimizer_steps * global_batch_size


def _validate_scheduler_tracking_congruence(actor) -> None:
    _optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(actor)
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
        return
    installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != policy_version:
        raise RuntimeError("full-parameter policy tracking is already initialized")


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

    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot record a LoRA actor")
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
    optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(actor)
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
    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot export a LoRA actor")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor)
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
    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot validate a LoRA actor")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor)
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
    from miles.backends.megatron_utils.lora_utils import is_lora_enabled
    from miles.backends.megatron_utils.trainable_state import (
        _copy_masters_to_model,
    )

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot apply to a LoRA actor")
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
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    count = apply_full_parameter_shard_values(
        state,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
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
        if not torch.isfinite(value).all().item():
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
) -> tuple[FullParameterFragmentDescriptor, list[object | None]]:
    sources = [masters[name].detach().reshape(-1) for name in fragment.wire_names]
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
        if not torch.isfinite(chunk).all().item():
            raise ValueError("full-parameter optimizer master contains NaN or Inf")
        chunk_index = len(chunk_descriptors)
        raw = _tensor_bytes(chunk)
        fragment_digest.update(raw)
        descriptor = FullParameterChunkDescriptor(
            chunk_index=chunk_index,
            flat_offset=flat_offset,
            numel=chunk_numel,
            payload_hash=_chunk_hash(
                fragment.fragment_id,
                chunk_index,
                flat_offset,
                chunk,
                parameter_layout_hash=parameter_layout_hash,
                plan_hash=plan_hash,
            ),
        )
        reference = ray_module.put(chunk)
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


def export_full_parameter_chunked_shard(
    actor,
    *,
    policy_version: int,
    local_step_generation: int = 0,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    role: str = "actor",
    ray_module=None,
) -> FullParameterChunkedShardState:
    """Export bounded owner chunks without materializing a rank or global cut."""

    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot export a LoRA actor")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor)
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
    for fragment in plan.fragments:
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
    return state


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
) -> int:
    """Validate every immutable chunk and stage only its refs for later commit."""

    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if is_lora_enabled(actor.args):
        raise RuntimeError("full-parameter DiLoCo cannot prepare a LoRA actor")
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
    _validate_scheduler_tracking_congruence(actor)
    if exported != generation:
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest = _require_full_parameter_fragment_plan(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    for fragment, references in zip(
        state.fragments,
        state.chunk_refs,
        strict=True,
    ):
        fragment_digest = _fragment_hasher(
            fragment.fragment_id,
            state.parameter_layout_hash,
            state.plan_hash,
        )
        for descriptor, reference in zip(fragment.chunks, references, strict=True):
            value = _validate_cpu_chunk(
                ray_module.get(reference),
                descriptor,
                fragment_id=fragment.fragment_id,
                verify_values=True,
                parameter_layout_hash=state.parameter_layout_hash,
                plan_hash=state.plan_hash,
            )
            fragment_digest.update(_tensor_bytes(value))
            del value
        if fragment_digest.hexdigest() != fragment.payload_hash:
            raise ValueError("full-parameter fragment hash changed")
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
    flattened = [masters[name].detach().reshape(-1) for name in fragment.wire_names]
    for chunk_descriptor, reference in zip(
        descriptor.chunks,
        references,
        strict=True,
    ):
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
                copied_parameters += 1
                master_index += 1
                master_offset = 0
        del value
    if master_index != len(flattened) or master_offset != 0:
        raise RuntimeError("full-parameter chunks did not cover every target")
    return copied_parameters


@torch.no_grad()
def commit_prepared_full_parameter_shard(
    actor,
    *,
    commit_token: str,
    role: str = "actor",
    ray_module=None,
) -> int:
    """Commit a fully prepared cut while retaining Adam state objects."""

    from miles.backends.megatron_utils.trainable_state import (
        _copy_masters_to_model,
    )

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
        prepared = replace(prepared, commit_started=True)
        actor._external_full_parameter_prepared_cut = prepared
    count = 0
    for fragment, descriptor, references in zip(
        plan.fragments,
        state.fragments,
        state.chunk_refs,
        strict=True,
    ):
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
