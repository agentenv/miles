"""Reference-backed Ray transport for topology-owned full-parameter fragments."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    FullParameterCommitStatus,
    FullParameterFragmentDescriptor,
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterOwnerFragmentPlan,
    FullParameterShardManifest,
    FullParameterTopology,
    _validate_commit_token,
    _chunk_hash,
    _fragment_hasher,
    _resolve_ray_module,
    _tensor_bytes,
    _validate_cpu_chunk,
    full_parameter_chunked_shard_identity,
    validate_full_parameter_chunked_shard_state,
)


_CUT_CONTENT_IDENTITY_DOMAIN = b"miles-full-parameter-chunked-cut-content-v1\0"


@dataclass(frozen=True)
class FullParameterChunkedCut:
    """Complete cut descriptor whose tensor bytes remain in nested ObjectRefs."""

    policy_version: int
    local_step_generation: int
    parameter_layout_hash: str
    shards: tuple[FullParameterChunkedShardState, ...]

    def __post_init__(self) -> None:
        _validate_chunked_cut(self)


class FullParameterCommitInDoubtError(RuntimeError):
    """A policy commit began but did not reach unanimous finalization."""

    def __init__(self, commit_token: str, phase: str) -> None:
        self.commit_token = commit_token
        self.phase = phase
        super().__init__(f"full-parameter commit {commit_token!r} is in doubt during {phase}")


def _validate_chunked_cut(
    cut: FullParameterChunkedCut,
    *,
    require_refs: bool = True,
) -> FullParameterChunkedCut:
    if not isinstance(cut, FullParameterChunkedCut):
        raise TypeError("full-parameter chunked cut has the wrong type")
    if not cut.shards or cut.shards != tuple(sorted(cut.shards, key=lambda state: state.topology)):
        raise ValueError("full-parameter chunked cut shards are not canonical")
    topologies = [state.topology for state in cut.shards]
    if len(set(topologies)) != len(topologies):
        raise ValueError("full-parameter chunked cut has duplicate topologies")
    fragment_ids = []
    for state in cut.shards:
        validate_full_parameter_chunked_shard_state(
            state,
            require_refs=require_refs,
        )
        if state.policy_version != cut.policy_version or state.local_step_generation != cut.local_step_generation or state.parameter_layout_hash != cut.parameter_layout_hash:
            raise ValueError("full-parameter chunked cut mixes policy identities")
        fragment_ids.extend(fragment.fragment_id for fragment in state.fragments)
    if sorted(fragment_ids) != list(range(len(fragment_ids))):
        raise ValueError("full-parameter chunked cut is not a complete fragment sweep")
    return cut


def full_parameter_chunked_cut_identity(cut: FullParameterChunkedCut) -> str:
    """Return a stable transaction identity without resolving payload refs."""

    _validate_chunked_cut(cut)
    digest = hashlib.sha256(b"miles-full-parameter-chunked-cut-v1\0")
    for state in cut.shards:
        digest.update(full_parameter_chunked_shard_identity(state).encode("ascii"))
    return digest.hexdigest()


def full_parameter_chunked_cut_content_identity(
    cut: FullParameterChunkedCut,
) -> str:
    """Hash cut content and owner metadata independently of version labels."""

    _validate_chunked_cut(cut)
    digest = hashlib.sha256(_CUT_CONTENT_IDENTITY_DOMAIN)
    for state in cut.shards:
        normalized = replace(
            state,
            policy_version=0,
            local_step_generation=0,
        )
        digest.update(full_parameter_chunked_shard_identity(normalized).encode("ascii"))
    return digest.hexdigest()


def full_parameter_commit_token(
    cut: FullParameterChunkedCut,
    caller_nonce: str | None = None,
) -> str:
    """Derive a non-reusable actor token from cut identity and caller nonce."""

    identity = full_parameter_chunked_cut_identity(cut)
    nonce = "" if caller_nonce is None else _validate_commit_token(caller_nonce)
    digest = hashlib.sha256(b"miles-full-parameter-commit-token-v1\0")
    digest.update(identity.encode("ascii"))
    digest.update(b"\0")
    digest.update(nonce.encode("ascii"))
    return _validate_commit_token(f"fullparam:{digest.hexdigest()}")


def _fragment_row(
    cut: FullParameterChunkedCut,
    fragment_id: int,
) -> tuple[
    FullParameterChunkedShardState,
    FullParameterFragmentDescriptor,
    list[object | None],
]:
    _validate_chunked_cut(cut)
    if isinstance(fragment_id, bool) or not isinstance(fragment_id, int):
        raise TypeError("full-parameter fragment ID has the wrong type")
    matches = [
        (shard, descriptor, references)
        for shard in cut.shards
        for descriptor, references in zip(
            shard.fragments,
            shard.chunk_refs,
            strict=True,
        )
        if descriptor.fragment_id == fragment_id
    ]
    if len(matches) != 1:
        raise ValueError("full-parameter fragment has no unique owner")
    return matches[0]


def iter_full_parameter_fragment_parts(
    cut: FullParameterChunkedCut,
    fragment_id: int,
    *,
    ray_module=None,
):
    """Yield one canonical FP32 fragment without driver-side reassembly."""

    shard, descriptor, references = _fragment_row(cut, fragment_id)
    ray_module = _resolve_ray_module(ray_module)
    for chunk, reference in zip(descriptor.chunks, references, strict=True):
        value = _validate_cpu_chunk(
            ray_module.get(reference),
            chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=shard.parameter_layout_hash,
            plan_hash=shard.plan_hash,
        )
        yield _tensor_bytes(value)
        del value


def iter_full_parameter_fragment_delta_parts(
    anchor: FullParameterChunkedCut,
    local: FullParameterChunkedCut,
    fragment_id: int,
    *,
    ray_module=None,
):
    """Yield bounded canonical FP32 ``local - anchor`` byte parts."""

    if anchor.parameter_layout_hash != local.parameter_layout_hash or anchor.policy_version != local.policy_version or anchor.local_step_generation != 0 or local.local_step_generation != 1:
        raise ValueError("full-parameter delta cuts do not share one local base")
    anchor_shard, anchor_descriptor, anchor_refs = _fragment_row(
        anchor,
        fragment_id,
    )
    local_shard, local_descriptor, local_refs = _fragment_row(local, fragment_id)
    if anchor_shard.topology != local_shard.topology or anchor_shard.plan_hash != local_shard.plan_hash or anchor_descriptor.fragment_id != local_descriptor.fragment_id or anchor_descriptor.numel != local_descriptor.numel or len(anchor_descriptor.chunks) != len(local_descriptor.chunks):
        raise ValueError("full-parameter delta fragment identity changed")
    ray_module = _resolve_ray_module(ray_module)
    for anchor_chunk, anchor_ref, local_chunk, local_ref in zip(
        anchor_descriptor.chunks,
        anchor_refs,
        local_descriptor.chunks,
        local_refs,
        strict=True,
    ):
        if anchor_chunk.chunk_index != local_chunk.chunk_index or anchor_chunk.flat_offset != local_chunk.flat_offset or anchor_chunk.numel != local_chunk.numel:
            raise ValueError("full-parameter delta chunk identity changed")
        anchor_value = _validate_cpu_chunk(
            ray_module.get(anchor_ref),
            anchor_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=anchor_shard.parameter_layout_hash,
            plan_hash=anchor_shard.plan_hash,
        )
        local_value = _validate_cpu_chunk(
            ray_module.get(local_ref),
            local_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=local_shard.parameter_layout_hash,
            plan_hash=local_shard.plan_hash,
        )
        delta = local_value.sub(anchor_value)
        if not torch.isfinite(delta).all().item():
            raise ValueError("full-parameter delta contains NaN or Inf")
        yield _tensor_bytes(delta)
        del anchor_value, local_value, delta


def iter_full_parameter_authoritative_fragment_delta_parts(
    local_cut: FullParameterChunkedCut,
    fragment_id: int,
    *,
    authoritative_parameter_layout_hash: str,
    authoritative_topology: FullParameterTopology,
    authoritative_plan_hash: str,
    authoritative_descriptor: FullParameterFragmentDescriptor,
    authoritative_refs: Sequence[object | None],
    ray_module=None,
):
    """Yield bounded FP32 ``local - authoritative`` parts for any local round."""

    local_shard, local_descriptor, local_refs = _fragment_row(
        local_cut,
        fragment_id,
    )
    if local_cut.local_step_generation < 1:
        raise ValueError("full-parameter local cut has no optimizer progress")
    if authoritative_parameter_layout_hash != local_cut.parameter_layout_hash:
        raise ValueError("full-parameter authoritative layout identity changed")
    if not isinstance(authoritative_topology, FullParameterTopology) or authoritative_topology != local_shard.topology:
        raise ValueError("full-parameter authoritative topology identity changed")
    if authoritative_plan_hash != local_shard.plan_hash:
        raise ValueError("full-parameter authoritative plan identity changed")
    if not isinstance(
        authoritative_descriptor,
        FullParameterFragmentDescriptor,
    ):
        raise TypeError("full-parameter authoritative fragment is malformed")
    if authoritative_descriptor.fragment_id != local_descriptor.fragment_id or authoritative_descriptor.numel != local_descriptor.numel or len(authoritative_descriptor.chunks) != len(local_descriptor.chunks):
        raise ValueError("full-parameter authoritative fragment identity changed")
    if not isinstance(authoritative_refs, Sequence) or isinstance(
        authoritative_refs,
        (str, bytes, bytearray, memoryview),
    ):
        raise TypeError("full-parameter authoritative fragment refs are malformed")
    if len(authoritative_refs) != len(authoritative_descriptor.chunks) or any(reference is None for reference in authoritative_refs):
        raise ValueError("full-parameter authoritative fragment refs are incomplete")

    ray_module = _resolve_ray_module(ray_module)
    local_fragment_hash = _fragment_hasher(
        fragment_id,
        local_shard.parameter_layout_hash,
        local_shard.plan_hash,
    )
    authoritative_fragment_hash = _fragment_hasher(
        fragment_id,
        authoritative_parameter_layout_hash,
        authoritative_plan_hash,
    )
    for local_chunk, local_ref, authoritative_chunk, authoritative_ref in zip(
        local_descriptor.chunks,
        local_refs,
        authoritative_descriptor.chunks,
        authoritative_refs,
        strict=True,
    ):
        if local_chunk.chunk_index != authoritative_chunk.chunk_index or local_chunk.flat_offset != authoritative_chunk.flat_offset or local_chunk.numel != authoritative_chunk.numel:
            raise ValueError("full-parameter authoritative chunk identity changed")
        local_value = _validate_cpu_chunk(
            ray_module.get(local_ref),
            local_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=local_shard.parameter_layout_hash,
            plan_hash=local_shard.plan_hash,
        )
        authoritative_value = _validate_cpu_chunk(
            ray_module.get(authoritative_ref),
            authoritative_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=authoritative_parameter_layout_hash,
            plan_hash=authoritative_plan_hash,
        )
        local_fragment_hash.update(_tensor_bytes(local_value))
        authoritative_fragment_hash.update(_tensor_bytes(authoritative_value))
        delta = local_value.sub(authoritative_value)
        if not torch.isfinite(delta).all().item():
            raise ValueError("full-parameter authoritative delta contains NaN or Inf")
        yield _tensor_bytes(delta)
        del local_value, authoritative_value, delta
    if local_fragment_hash.hexdigest() != local_descriptor.payload_hash:
        raise ValueError("full-parameter local fragment hash changed")
    if authoritative_fragment_hash.hexdigest() != authoritative_descriptor.payload_hash:
        raise ValueError("full-parameter authoritative fragment hash changed")


def store_full_parameter_fragment_payload(
    reference_cut: FullParameterChunkedCut,
    fragment_id: int,
    payload: bytes | bytearray | memoryview,
    *,
    ray_module=None,
) -> tuple[FullParameterFragmentDescriptor, list[object | None]]:
    """Store one bounded authoritative fragment as immutable Ray chunks."""

    shard, reference, _references = _fragment_row(reference_cut, fragment_id)
    try:
        raw = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("full-parameter fragment payload is not contiguous bytes") from exc
    if raw.nbytes != reference.numel * 4:
        raw.release()
        raise ValueError("full-parameter fragment payload byte count changed")
    ray_module = _resolve_ray_module(ray_module)
    fragment_digest = _fragment_hasher(
        fragment_id,
        shard.parameter_layout_hash,
        shard.plan_hash,
    )
    descriptors = []
    references: list[object | None] = []
    try:
        for expected in reference.chunks:
            start = expected.flat_offset * 4
            stop = start + expected.numel * 4
            view = raw[start:stop]
            value = torch.frombuffer(view, dtype=torch.float32).clone().contiguous()
            view.release()
            if not torch.isfinite(value).all().item():
                raise ValueError("full-parameter fragment contains NaN or Inf")
            fragment_digest.update(_tensor_bytes(value))
            descriptor = FullParameterChunkDescriptor(
                chunk_index=expected.chunk_index,
                flat_offset=expected.flat_offset,
                numel=expected.numel,
                payload_hash=_chunk_hash(
                    fragment_id,
                    expected.chunk_index,
                    expected.flat_offset,
                    value,
                    parameter_layout_hash=shard.parameter_layout_hash,
                    plan_hash=shard.plan_hash,
                ),
            )
            stored = ray_module.put(value)
            if stored is None:
                raise RuntimeError("Ray returned an empty fragment reference")
            descriptors.append(descriptor)
            references.append(stored)
            del value
    finally:
        raw.release()
    return (
        FullParameterFragmentDescriptor(
            fragment_id=fragment_id,
            numel=reference.numel,
            payload_hash=fragment_digest.hexdigest(),
            chunks=tuple(descriptors),
        ),
        references,
    )


def promote_full_parameter_chunked_cut(
    cut: FullParameterChunkedCut,
    target_policy_version: int,
) -> FullParameterChunkedCut:
    """Promote immutable payload refs to a newer committed global version."""

    _validate_chunked_cut(cut)
    if isinstance(target_policy_version, bool) or not isinstance(target_policy_version, int) or target_policy_version <= cut.policy_version:
        raise ValueError("promoted full-parameter cut must advance policy version")
    shards = tuple(
        replace(
            state,
            policy_version=target_policy_version,
            local_step_generation=0,
            chunk_refs=[list(references) for references in state.chunk_refs],
        )
        for state in cut.shards
    )
    return FullParameterChunkedCut(
        policy_version=target_policy_version,
        local_step_generation=0,
        parameter_layout_hash=cut.parameter_layout_hash,
        shards=shards,
    )


def assemble_full_parameter_chunked_target(
    reference_cut: FullParameterChunkedCut,
    *,
    target_policy_version: int,
    authoritative_fragments: Mapping[
        int,
        tuple[FullParameterFragmentDescriptor, Sequence[object | None]],
    ],
) -> FullParameterChunkedCut:
    """Build a target cut by reindexing authoritative payload refs only.

    ``reference_cut`` fixes the global layout, owner topology, owner plan, and
    canonical fragment order.  The authoritative mapping may select each
    fragment descriptor/ref row from a different local cut or synchronizer
    output, without resolving any tensor onto the orchestration driver.
    """

    _validate_chunked_cut(reference_cut)
    if isinstance(target_policy_version, bool) or not isinstance(target_policy_version, int) or target_policy_version <= reference_cut.policy_version:
        raise ValueError("assembled full-parameter cut must advance policy version")
    if not isinstance(authoritative_fragments, Mapping):
        raise TypeError("authoritative full-parameter fragments must be a mapping")
    if any(isinstance(fragment_id, bool) or not isinstance(fragment_id, int) or fragment_id < 0 for fragment_id in authoritative_fragments):
        raise ValueError("authoritative full-parameter fragment IDs are malformed")

    expected_ids = {fragment.fragment_id for shard in reference_cut.shards for fragment in shard.fragments}
    if set(authoritative_fragments) != expected_ids:
        raise ValueError("authoritative full-parameter fragments do not exactly cover the cut")

    target_shards = []
    for shard in reference_cut.shards:
        fragments = []
        chunk_refs = []
        for expected in shard.fragments:
            payload = authoritative_fragments[expected.fragment_id]
            if not isinstance(payload, tuple) or len(payload) != 2 or not isinstance(payload[0], FullParameterFragmentDescriptor) or not isinstance(payload[1], Sequence):
                raise TypeError("authoritative full-parameter fragment is malformed")
            descriptor, references = payload
            if descriptor.fragment_id != expected.fragment_id or descriptor.numel != expected.numel:
                raise ValueError("authoritative full-parameter fragment changed owner-plan identity")
            copied_references = list(references)
            if len(copied_references) != len(descriptor.chunks) or any(reference is None for reference in copied_references):
                raise ValueError("authoritative full-parameter fragment refs are incomplete")
            fragments.append(descriptor)
            chunk_refs.append(copied_references)
        target_shards.append(
            replace(
                shard,
                policy_version=target_policy_version,
                local_step_generation=0,
                fragments=tuple(fragments),
                chunk_refs=chunk_refs,
            )
        )
    return FullParameterChunkedCut(
        policy_version=target_policy_version,
        local_step_generation=0,
        parameter_layout_hash=reference_cut.parameter_layout_hash,
        shards=tuple(target_shards),
    )


def release_full_parameter_chunked_cut(cut: FullParameterChunkedCut) -> int:
    """Drop only this caller's nested refs and let Ray ownership release them."""

    if not isinstance(cut, FullParameterChunkedCut):
        raise TypeError("full-parameter chunked cut has the wrong type")
    released = 0
    for state in cut.shards:
        for references in state.chunk_refs:
            for index, reference in enumerate(references):
                if reference is not None:
                    references[index] = None
                    released += 1
    return released


def _actor_handles(group) -> tuple[Any, ...]:
    handles = tuple(getattr(group, "_actor_handles", ()))
    if not handles:
        raise RuntimeError("full-parameter transport has no actor handles")
    return handles


async def shard_manifests(group) -> tuple[FullParameterShardManifest, ...]:
    handles = _actor_handles(group)
    results = await group._broadcast("full_parameter_shard_manifest")
    if len(results) != len(handles) or any(not isinstance(manifest, FullParameterShardManifest) for manifest in results):
        raise RuntimeError("Megatron returned incomplete full-parameter manifests")
    topologies = [manifest.topology for manifest in results]
    if len(set(topologies)) != len(topologies):
        raise RuntimeError("Megatron returned duplicate full-parameter manifests")
    return tuple(sorted(results, key=lambda manifest: manifest.topology))


def _validate_group_plans(
    plans: tuple[FullParameterOwnerFragmentPlan, ...],
) -> dict[FullParameterTopology, FullParameterOwnerFragmentPlan]:
    if not plans or any(not isinstance(plan, FullParameterOwnerFragmentPlan) for plan in plans):
        raise ValueError("full-parameter owner plans are empty or malformed")
    by_topology = {plan.topology: plan for plan in plans}
    if len(by_topology) != len(plans):
        raise ValueError("full-parameter owner plans contain duplicate topologies")
    layout_hashes = {plan.parameter_layout_hash for plan in plans}
    if len(layout_hashes) != 1:
        raise ValueError("full-parameter owner plans mix global layouts")
    fragment_ids = [fragment.fragment_id for plan in plans for fragment in plan.fragments]
    if sorted(fragment_ids) != list(range(len(fragment_ids))):
        raise ValueError("full-parameter owner plans are not a complete fragment sweep")
    return by_topology


async def install_fragment_plans(
    group,
    plans: tuple[FullParameterOwnerFragmentPlan, ...],
) -> int:
    handles = _actor_handles(group)
    by_topology = _validate_group_plans(tuple(plans))
    topologies = await group._broadcast("full_parameter_topology")
    if len(topologies) != len(handles) or len(set(topologies)) != len(topologies) or set(topologies) != set(by_topology):
        raise RuntimeError("full-parameter owner plans do not match the actor group")
    results = await asyncio.gather(*(actor.install_full_parameter_fragment_plan.remote(by_topology[topology]) for actor, topology in zip(handles, topologies, strict=True)))
    expected = [len(by_topology[topology].fragments) for topology in topologies]
    if any(isinstance(result, bool) or not isinstance(result, int) or result != count for result, count in zip(results, expected, strict=True)):
        raise RuntimeError("Megatron ranks rejected full-parameter owner plans")
    return sum(results)


async def export_chunked_cut(
    group,
    policy_version: int,
    local_step_generation: int = 0,
    *,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
) -> FullParameterChunkedCut:
    handles = _actor_handles(group)
    results = await group._broadcast(
        "export_full_parameter_chunked_shard",
        policy_version,
        local_step_generation,
        max_chunk_bytes,
    )
    if len(results) != len(handles) or any(not isinstance(state, FullParameterChunkedShardState) for state in results):
        raise RuntimeError("Megatron returned an incomplete chunked cut")
    ordered = tuple(sorted(results, key=lambda state: state.topology))
    layout_hashes = {state.parameter_layout_hash for state in ordered}
    if len(layout_hashes) != 1:
        raise RuntimeError("Megatron chunked shards mix global layouts")
    return FullParameterChunkedCut(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        parameter_layout_hash=next(iter(layout_hashes)),
        shards=ordered,
    )


async def abort_prepared_chunked_cut(group, commit_token: str) -> tuple[object, ...]:
    token = _validate_commit_token(commit_token)
    handles = _actor_handles(group)
    return tuple(
        await asyncio.gather(
            *(actor.abort_prepared_full_parameter_shard.remote(token) for actor in handles),
            return_exceptions=True,
        )
    )


def _first_failure(results: list[object], *, expected_positive: bool) -> BaseException | None:
    for result in results:
        if isinstance(result, BaseException):
            return result
        if expected_positive and (isinstance(result, bool) or not isinstance(result, int) or result < 1):
            return RuntimeError("Megatron returned an invalid full-parameter count")
    return None


async def apply_chunked_cut(
    group,
    cut: FullParameterChunkedCut,
    *,
    commit_token: str | None = None,
    max_commit_attempts: int = 3,
) -> int:
    """Prepare, idempotently converge, then finalize every owner rank."""

    _validate_chunked_cut(cut)
    if cut.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    token = full_parameter_commit_token(cut, commit_token)
    if isinstance(max_commit_attempts, bool) or not isinstance(max_commit_attempts, int) or max_commit_attempts < 1 or max_commit_attempts > 8:
        raise ValueError("full-parameter commit retry budget is invalid")
    handles = _actor_handles(group)
    by_topology = {state.topology: state for state in cut.shards}
    topologies = await group._broadcast("full_parameter_topology")
    if len(topologies) != len(handles) or len(by_topology) != len(handles) or set(topologies) != set(by_topology):
        raise RuntimeError("full-parameter chunked cut does not match the actor group")

    prepare_results = list(
        await asyncio.gather(
            *(
                actor.prepare_full_parameter_chunked_shard.remote(
                    by_topology[topology],
                    token,
                )
                for actor, topology in zip(handles, topologies, strict=True)
            ),
            return_exceptions=True,
        )
    )
    prepare_failure = _first_failure(prepare_results, expected_positive=True)
    if prepare_failure is not None:
        status_results = list(
            await asyncio.gather(
                *(
                    actor.full_parameter_commit_status.remote(
                        token,
                        by_topology[topology].policy_version,
                        full_parameter_chunked_shard_identity(by_topology[topology]),
                    )
                    for actor, topology in zip(handles, topologies, strict=True)
                ),
                return_exceptions=True,
            )
        )
        safe_to_abort = all(isinstance(status, FullParameterCommitStatus) and status.status in {"ABSENT", "PREPARED"} for status in status_results)
        if safe_to_abort:
            abort_results = await abort_prepared_chunked_cut(group, token)
            abort_safe = all(
                type(result) is bool and result == (status.status == "PREPARED")
                for result, status in zip(
                    abort_results,
                    status_results,
                    strict=True,
                )
            )
            if not abort_safe:
                raise FullParameterCommitInDoubtError(
                    token,
                    "prepare-abort-race",
                )
            raise RuntimeError("Megatron rank rejected full-parameter prepare") from prepare_failure
        raise FullParameterCommitInDoubtError(token, "prepare-reconciliation")

    commit_results: list[object] = []
    for _attempt in range(max_commit_attempts):
        commit_results = list(
            await asyncio.gather(
                *(actor.commit_prepared_full_parameter_shard.remote(token) for actor in handles),
                return_exceptions=True,
            )
        )
        if _first_failure(commit_results, expected_positive=True) is None:
            break
    else:
        # A successful rank may already expose the target in its masters/model.
        # Never abort after commit begins: retain actor journals and immutable
        # refs so the exact token can be reconciled, or tear down and restore a
        # known complete Miles checkpoint before any publication/training.
        raise FullParameterCommitInDoubtError(token, "commit")
    if commit_results != prepare_results:
        raise FullParameterCommitInDoubtError(token, "count-reconciliation")

    finalize_results: list[object] = []
    for _attempt in range(max_commit_attempts):
        finalize_results = list(
            await asyncio.gather(
                *(actor.finalize_full_parameter_commit.remote(token) for actor in handles),
                return_exceptions=True,
            )
        )
        if _first_failure(finalize_results, expected_positive=True) is None:
            break
    else:
        raise FullParameterCommitInDoubtError(token, "finalize")
    if finalize_results != commit_results:
        raise FullParameterCommitInDoubtError(token, "final-count-reconciliation")
    return sum(finalize_results)
