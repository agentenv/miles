"""TrainerController methods of the full-parameter transport (yeto full-parameter DiLoCo).

Every rank is reached through ``run_plugin``; see
``miles.ray.full_parameter_transport`` for the transaction protocol.

The controller's public methods are its rpc surface
(``collect_rpc_method_specs``), so every parameter and return is annotated.
Cuts carry Ray ObjectRefs and plans carry topology keys, which only the Ray
worker-comm backend moves; their annotations are ``Any``.
"""

from __future__ import annotations

from typing import Any

from miles.ray import full_parameter_transport


class FullParameterTransportMixin:
    async def full_parameter_shard_manifests(self) -> tuple[Any, ...]:
        """Topology/spec metadata of every rank, without parameter payloads."""
        return await full_parameter_transport.shard_manifests(self)

    async def install_full_parameter_fragment_plans(self, plans: tuple[Any, ...]) -> int:
        """Install one exact-coverage fragment plan on every owner rank."""
        return await full_parameter_transport.install_fragment_plans(self, plans)

    async def full_parameter_partition_reports(self) -> tuple[Any, ...]:
        """Every rank's parameters with their tensor-parallel partitions (shape-free layout)."""
        return await full_parameter_transport.partition_reports(self)

    async def install_full_parameter_owner_maps(self, owner_maps: dict[Any, Any]) -> int:
        """Install every rank's parameter owners before its plan (shape-free layout)."""
        return await full_parameter_transport.install_owner_maps(self, owner_maps)

    async def export_full_parameter_chunked_cut(
        self,
        policy_version: int,
        local_step_generation: int = 0,
        *,
        max_chunk_bytes: int,
        defer_payload: bool = False,
        lazy: bool = False,
    ) -> Any:
        """Export small descriptors and nested Ray ObjectRefs only.

        ``defer_payload``: descriptors only, no Ray objects; see
        ``materialize_full_parameter_fragments``.  ``lazy``: labels instead of
        content hashes, no master read (``full_parameter_transport``)."""
        return await full_parameter_transport.export_chunked_cut(
            self,
            policy_version,
            local_step_generation,
            max_chunk_bytes=max_chunk_bytes,
            defer_payload=defer_payload,
            lazy=lazy,
        )

    async def materialize_full_parameter_fragments(
        self, cut: Any, fragment_ids: list[int], *, return_descriptors: bool = False
    ) -> dict[Any, Any]:
        """Store deferred fragments of ``cut`` while the masters hold them."""
        return await full_parameter_transport.materialize_chunked_fragments(
            self, cut, tuple(fragment_ids), return_descriptors=return_descriptors
        )

    async def apply_full_parameter_chunked_cut(
        self,
        cut: Any,
        *,
        commit_token: str | None = None,
        max_commit_attempts: int = 3,
        skip_resident_fragments: bool = False,
        verified_fragments: dict[int, str] | None = None,
    ) -> int:
        """Prepare, idempotently converge, then finalize every rank."""
        return await full_parameter_transport.apply_chunked_cut(
            self,
            cut,
            commit_token=commit_token,
            max_commit_attempts=max_commit_attempts,
            skip_resident_fragments=skip_resident_fragments,
            verified_fragments=verified_fragments,
        )

    async def record_full_parameter_local_step(
        self, *, base_policy_version: int, rollout_id: int, check_model: bool = True
    ) -> tuple[Any, ...]:
        """Record exact successful scheduler progress on every rank."""
        return await full_parameter_transport.record_local_step(
            self, base_policy_version=base_policy_version, rollout_id=rollout_id, check_model=check_model
        )

    async def full_parameter_optimizer_states(self, *, check_model: bool = True) -> tuple[Any, ...]:
        """One bounded optimizer-progress proof from every rank."""
        return await full_parameter_transport.optimizer_states(self, check_model=check_model)
