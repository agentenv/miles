# FROZEN: v1 RayTrainGroup is the non-FT default path. Only critical bugfixes
# go here; new features land in miles/ray/train/group.py (v2). Dispatch between
# v1 and v2 happens in miles/ray/placement_group.py based on the env var
# MILES_EXPERIMENTAL_FT_TRAINER (default off -> v1).

import asyncio

import ray
from ray.util.placement_group import PlacementGroup

from miles.backends.megatron_utils.full_parameter_state import (
    FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    FullParameterLocalStepReceipt,
    FullParameterOptimizerState,
    FullParameterOwnerFragmentPlan,
    FullParameterShardManifest,
    FullParameterShardState,
)
from miles.backends.megatron_utils.trainable_state import TrainableState
from miles.ray.full_parameter_transport import (
    FullParameterChunkedCut,
    abort_prepared_chunked_cut,
    apply_chunked_cut,
    export_chunked_cut,
    install_fragment_plans,
    shard_manifests,
)
from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.utils.ft_utils.indep_dp import IndepDPInfo


class RayTrainGroup:
    """
    A group of ray actors

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        rollout_manager: object | None,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref
        self._rollout_manager = rollout_manager
        self.with_opd_teacher = with_opd_teacher

        # Allocate the GPUs for actors w/o instantiating them
        self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        return allocate_gpus_for_actor(
            args=self.args,
            gpus_per_cell=self._num_nodes * self._num_gpus_per_node,
            pg=pg,
            num_gpus_per_actor=num_gpus_per_actor,
            indep_dp_store_addr=None,
            role=self.role,
            cell_index=0,
        )

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        indep_dp_info = IndepDPInfo.create_trivial()
        return await self._broadcast(
            "init",
            self.args,
            self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=indep_dp_info,
        )

    async def train(self, rollout_id, rollout_data_pack):
        """Do one rollout training"""
        await self._broadcast(
            "train",
            rollout_id,
            rollout_data_pack["data_ref"],
            witness_info=None,
            attempt=0,
        )

    async def evaluate_critic(self, rollout_id, rollout_data_pack):
        """Run one optimizer-free critic evaluation batch on every DP rank."""
        return await self._broadcast(
            "evaluate_critic",
            rollout_id,
            rollout_data_pack["data_ref"],
        )

    async def export_trainable_state(self) -> TrainableState:
        """Export replicated trainable state from the Megatron main rank."""
        results = await self._broadcast("export_trainable_state")
        exported = [result for result in results if result is not None]
        if len(exported) != 1:
            raise RuntimeError("trainable state must be exported by exactly one rank")
        return exported[0]

    async def apply_trainable_state(self, state: TrainableState, *, reset_optimizer: bool) -> int:
        """Apply one object-store copy of trainable state to every Megatron rank."""
        # Passing a large state directly to each ``.remote`` call makes Ray
        # serialize one independent copy per actor.  A rank-64 E288 adapter is
        # roughly 55 GiB, so the per-rank copies exhaust host memory before the
        # first rollout.  Top-level ObjectRefs are resolved for the actor, while
        # all calls share the same plasma object.
        shared_state = ray.put(state)
        results = await self._broadcast(
            "apply_trainable_state",
            shared_state,
            reset_optimizer=reset_optimizer,
        )
        if not results or any(result != results[0] for result in results[1:]):
            raise RuntimeError("Megatron ranks disagree after applying trainable state")
        return results[0]

    async def export_full_parameter_shards(
        self,
        policy_version: int,
        local_step_generation: int = 0,
    ) -> tuple[FullParameterShardState, ...]:
        """Export one deterministic FP32 shard from every Megatron rank."""
        results = await self._broadcast(
            "export_full_parameter_shard",
            policy_version,
            local_step_generation,
        )
        if len(results) != len(self._actor_handles) or any(
            not isinstance(state, FullParameterShardState) for state in results
        ):
            raise RuntimeError("Megatron returned an incomplete full-parameter cut")
        topologies = [state.topology for state in results]
        if len(set(topologies)) != len(topologies):
            raise RuntimeError("Megatron returned duplicate full-parameter shards")
        return tuple(sorted(results, key=lambda state: state.topology))

    async def full_parameter_shard_manifests(
        self,
    ) -> tuple[FullParameterShardManifest, ...]:
        """Return topology/spec metadata without copying parameter payloads."""
        return await shard_manifests(self)

    async def install_full_parameter_fragment_plans(
        self,
        plans: tuple[FullParameterOwnerFragmentPlan, ...],
    ) -> int:
        """Install one exact-coverage fragment plan on every owner rank."""
        return await install_fragment_plans(self, plans)

    async def export_full_parameter_chunked_cut(
        self,
        policy_version: int,
        local_step_generation: int = 0,
        *,
        max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    ) -> FullParameterChunkedCut:
        """Export only small descriptors and nested Ray ObjectRefs."""
        return await export_chunked_cut(
            self,
            policy_version,
            local_step_generation,
            max_chunk_bytes=max_chunk_bytes,
        )

    async def apply_full_parameter_chunked_cut(
        self,
        cut: FullParameterChunkedCut,
        *,
        commit_token: str | None = None,
        max_commit_attempts: int = 3,
    ) -> int:
        """Prepare, idempotently converge, then finalize every rank."""
        return await apply_chunked_cut(
            self,
            cut,
            commit_token=commit_token,
            max_commit_attempts=max_commit_attempts,
        )

    async def abort_full_parameter_chunked_cut(
        self,
        commit_token: str,
    ) -> tuple[object, ...]:
        """Abort actor-local prepare contexts without deleting shared refs."""
        return await abort_prepared_chunked_cut(self, commit_token)

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version: int,
        rollout_id: int,
        max_receipt_attempts: int = 3,
    ) -> tuple[FullParameterLocalStepReceipt, ...]:
        """Record exact successful scheduler progress on every rank."""

        if (
            isinstance(max_receipt_attempts, bool)
            or not isinstance(max_receipt_attempts, int)
            or not 1 <= max_receipt_attempts <= 8
        ):
            raise ValueError("full-parameter receipt retry budget is invalid")
        results = []
        for _attempt in range(max_receipt_attempts):
            results = list(
                await asyncio.gather(
                    *(
                        actor.record_full_parameter_local_step.remote(
                            base_policy_version,
                            rollout_id,
                        )
                        for actor in self._actor_handles
                    ),
                    return_exceptions=True,
                )
            )
            if all(isinstance(receipt, FullParameterLocalStepReceipt) for receipt in results):
                break
        else:
            raise RuntimeError("Megatron local-step receipt is in doubt; learner must fail stop")
        if len(results) != len(self._actor_handles) or any(
            not isinstance(receipt, FullParameterLocalStepReceipt) for receipt in results
        ):
            raise RuntimeError("Megatron returned incomplete local-step receipts")
        topologies = [receipt.topology for receipt in results]
        if len(set(topologies)) != len(topologies):
            raise RuntimeError("Megatron returned duplicate local-step receipts")
        progress = {
            (
                receipt.role,
                receipt.base_policy_version,
                receipt.local_step_generation,
                receipt.rollout_id,
                receipt.optimizer_steps,
                receipt.scheduler_start_steps,
                receipt.scheduler_end_steps,
            )
            for receipt in results
        }
        if len(progress) != 1:
            raise RuntimeError("Megatron ranks disagree on local optimizer progress")
        return tuple(sorted(results, key=lambda receipt: receipt.topology))

    async def apply_full_parameter_shards(
        self,
        states: tuple[FullParameterShardState, ...],
    ) -> int:
        """Route a complete topology cut back to its owning Megatron ranks."""
        if not states or len(states) != len(self._actor_handles):
            raise RuntimeError("full-parameter cut does not cover every Megatron rank")
        by_topology = {state.topology: state for state in states}
        if len(by_topology) != len(states):
            raise RuntimeError("full-parameter cut contains duplicate topologies")
        topologies = await self._broadcast("full_parameter_topology")
        if len(topologies) != len(self._actor_handles) or set(topologies) != set(by_topology):
            raise RuntimeError("full-parameter cut topology does not match the actor group")
        references = {topology: ray.put(by_topology[topology]) for topology in topologies}
        validations = await asyncio.gather(
            *(
                actor.validate_full_parameter_shard.remote(references[topology])
                for actor, topology in zip(
                    self._actor_handles,
                    topologies,
                    strict=True,
                )
            )
        )
        if not validations or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in validations
        ):
            raise RuntimeError("Megatron ranks rejected full-parameter prevalidation")
        results = await asyncio.gather(
            *(
                actor.apply_full_parameter_shard.remote(references[topology])
                for actor, topology in zip(
                    self._actor_handles,
                    topologies,
                    strict=True,
                )
            )
        )
        if not results or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in results):
            raise RuntimeError("Megatron ranks rejected the full-parameter cut")
        return sum(results)

    async def full_parameter_optimizer_states(
        self,
    ) -> tuple[FullParameterOptimizerState, ...]:
        """Return one bounded optimizer-progress proof from every rank."""

        results = await self._broadcast("full_parameter_optimizer_state")
        if len(results) != len(self._actor_handles) or any(
            not isinstance(state, FullParameterOptimizerState) for state in results
        ):
            raise RuntimeError("Megatron returned incomplete optimizer-state proofs")
        topologies = [state.topology for state in results]
        if len(set(topologies)) != len(topologies):
            raise RuntimeError("Megatron returned duplicate optimizer-state proofs")
        progress = {
            (
                state.installed_policy_version,
                state.last_rollout_id,
                state.scheduler_num_steps,
            )
            for state in results
        }
        if len(progress) != 1:
            raise RuntimeError("Megatron ranks disagree on optimizer progress")
        return tuple(sorted(results, key=lambda state: state.topology))

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights from rank 0 to all other ranks."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            await self.rollout_manager.recover_updatable_engines.remote()

        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        await self.rollout_manager.health_monitoring_pause.remote()

        await self._broadcast("update_weights", info=info)
        return info

    async def prepare_weight_update(self):
        await self._broadcast("prepare_weight_update")

    async def reconcile_adapters(self) -> None:
        """Multi-LoRA: reconcile loaded adapters with the controller's active set
        (load new, cleanup gone). Called by the trainer before generate."""
        await self._broadcast("reconcile_adapters")

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self):
        self.rollout_manager = self._rollout_manager
        await self._broadcast("set_rollout_manager", self._rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
