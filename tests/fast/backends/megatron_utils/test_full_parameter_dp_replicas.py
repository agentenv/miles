"""Full-parameter sync with data-parallel replicas (DP>1, EP spanning DP), on CPU.

Several fake Megatron ranks run in one process behind a fake trainer group
whose ``run_plugin`` calls the real per-rank entry point
(``full_parameter_rank_call``) for each rank in turn.  Each rank has its own
model, its own non-distributed optimizer masters and its own topology and
replica groups:

* rank r: TP1, PP1, DP = world, EP = ``ep``; ``ep_rank = r % ep``;
* a dense parameter is replicated on every rank (DP group with CP);
* an expert parameter (``allreduce=False``) is replicated on the ranks with
  the same EP rank (expert data-parallel group).

The tests prove that the owners' specs cover every parameter exactly once,
that every replica applies the merged bytes identically (bit-identical masters
and model on every rank), that a skip-resident apply after a deferred export
reads only the changed fragments on owners and replicas alike, and that a
diverged replica is detected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH, full_parameter_rank_call
from miles.backends.megatron_utils.full_parameter_replicas import VERIFY_REPLICAS_ENV
from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    FullParameterReplication,
    FullParameterTopology,
)
from miles.ray import full_parameter_transport as transport
from miles.ray.full_parameter_transport import (
    FullParameterChunkedCut,
    apply_chunked_cut,
    export_chunked_cut,
    install_fragment_plans,
    materialize_chunked_fragments,
    optimizer_states,
    record_local_step,
    replica_owner_manifests,
    shard_manifests,
    store_full_parameter_fragment_payload,
)

pytestmark = pytest.mark.asyncio

_LAYOUT_HASH = "9" * 64
_EXPERTS_PER_RANK = 2
# Dense parameters (replicated on every rank).  The embedding is the largest
# and is split across two fragments (flat ranges).
_DENSE = {
    "embedding.word_embeddings.weight": (6, 3),
    "decoder.layers.0.self_attention.linear_qkv.weight": (4, 3),
    "decoder.layers.0.mlp.router.weight": (4, 3),
    "decoder.layers.0.input_layernorm.weight": (3,),
    "decoder.final_layernorm.weight": (3,),
    "output_layer.weight": (5, 3),
}
_EXPERT_SHAPES = {"linear_fc1": (4, 3), "linear_fc2": (3, 2)}
_SPLIT_NUMEL = 16  # a parameter above this many scalars spans two fragments


def _initial_value(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """The same deterministic master on every replica of ``name``."""

    seed = sum(ord(character) * (index + 1) for index, character in enumerate(name)) % 997
    numel = 1
    for dimension in shape:
        numel *= dimension
    return (torch.arange(numel, dtype=torch.float32) * 0.125 + seed / 64.0).reshape(shape)


def _expert_names(ep_rank: int) -> dict[str, tuple[int, ...]]:
    names = {}
    for local in range(_EXPERTS_PER_RANK):
        expert = ep_rank * _EXPERTS_PER_RANK + local
        for projection, shape in _EXPERT_SHAPES.items():
            names[f"decoder.layers.0.mlp.experts.local_experts.{expert}.{projection}.weight"] = shape
    return names


class _Rank:
    """One fake Megatron rank with a non-distributed optimizer (whole FP32 masters)."""

    def __init__(self, rank: int, *, world: int, ep: int) -> None:
        self.rank = rank
        self.role = "actor"
        self.topology = FullParameterTopology(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            ep_rank=rank % ep,
            ep_size=ep,
            cp_rank=0,
            cp_size=1,
            dp_rank=rank,
            dp_size=world,
        )
        self.replication = FullParameterReplication(
            global_rank=rank,
            dense_ranks=tuple(range(world)),
            expert_ranks=tuple(other for other in range(world) if other % ep == rank % ep),
        )
        named = []
        for name, shape in {**_DENSE, **_expert_names(rank % ep)}.items():
            parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
            parameter.main_param = _initial_value(name, shape)
            parameter.data.copy_(parameter.main_param.to(torch.bfloat16))
            if ".experts." in name:
                parameter.allreduce = False
            named.append((name, parameter))
        self.named = tuple(named)
        masters = [parameter.main_param for _name, parameter in self.named]
        self.optimizer = SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=[{"params": masters}],
                state={
                    master: {"exp_avg": torch.ones_like(master), "exp_avg_sq": torch.ones_like(master)}
                    for master in masters
                },
            )
        )
        self.args = SimpleNamespace(
            lora_rank=0, lora_adapter_path=None, num_steps_per_rollout=1, global_batch_size=1, start_rollout_id=None
        )
        self.opt_param_scheduler = SimpleNamespace(num_steps=0)
        self.weights_backuper = SimpleNamespace(backup=lambda role: None)
        self._last_rollout_id = 0

    def masters(self) -> dict[str, torch.Tensor]:
        return {name: parameter.main_param for name, parameter in self.named}

    def model(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach() for name, parameter in self.named}

    def local_step(self, rollout_id: int, delta: float) -> None:
        """What an identical optimizer step does on every replica."""

        for _name, parameter in self.named:
            parameter.main_param.add_(delta)
            parameter.data.copy_(parameter.main_param.to(torch.bfloat16))
        self.opt_param_scheduler.num_steps += 1
        self._last_rollout_id = rollout_id


class _FakeRay:
    def __init__(self) -> None:
        self.objects = {}
        self.gets = 0

    def put(self, value):
        reference = object()
        self.objects[reference] = value.detach().clone()
        return reference

    def get(self, reference):
        self.gets += 1
        return self.objects[reference]


class _Group:
    """A trainer controller: ``run_plugin`` runs the rank entry point on each rank in order."""

    def __init__(self, ranks, current) -> None:
        self.ranks = list(ranks)
        self._current = current

    async def run_plugin(self, fn_path, kwargs):
        assert fn_path == PLUGIN_PATH
        results = []
        for rank in self.ranks:
            self._current["rank"] = rank
            results.append(full_parameter_rank_call(rank, **kwargs))
        return results


@pytest.fixture
def cluster(monkeypatch):
    """``make(world, ep)`` -> (group, ranks, fake ray); the per-rank hooks follow the current rank."""

    current = {}
    ray = _FakeRay()
    monkeypatch.setattr(fps, "full_parameter_topology", lambda: current["rank"].topology)
    monkeypatch.setattr(fps, "full_parameter_replication", lambda: current["rank"].replication)
    monkeypatch.setattr(fps, "_named_actor_parameters", lambda actor: actor.named)
    monkeypatch.setattr(
        fps,
        "_copy_masters_to_model",
        lambda actor: [parameter.data.copy_(parameter.main_param.to(torch.bfloat16)) for _n, parameter in actor.named],
    )
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: ray)
    monkeypatch.setattr(transport, "_resolve_ray_module", lambda _module: ray)

    def make(world: int, ep: int):
        ranks = [_Rank(rank, world=world, ep=ep) for rank in range(world)]
        return _Group(ranks, current), ranks, ray

    return make


def _plans(manifests) -> tuple[FullParameterOwnerFragmentPlan, ...]:
    """One owner plan per rank, like yeto's owner-affine planner in miniature.

    Each rank's specs, in canonical order, become fragments of at most two
    whole parameters; a parameter above ``_SPLIT_NUMEL`` is split in half
    across two fragments of its own (flat ranges)."""

    plans, next_id = [], 0
    for manifest in sorted(manifests, key=lambda value: value.topology):
        groups: list[tuple[tuple[object, int, int], ...]] = []  # members: (spec, start, stop)
        pending: list[tuple[object, int, int]] = []
        for spec in manifest.specs:
            if spec.numel > _SPLIT_NUMEL:
                if pending:
                    groups.append(tuple(pending))
                    pending = []
                half = spec.numel // 2
                groups.extend((((spec, 0, half),), ((spec, half, spec.numel),)))
                continue
            pending.append((spec, 0, spec.numel))
            if len(pending) == 2:
                groups.append(tuple(pending))
                pending = []
        if pending:
            groups.append(tuple(pending))
        fragments = []
        for members in groups:
            split = any(stop - start != spec.numel for spec, start, stop in members)
            fragments.append(
                FullParameterFragmentPlan(
                    fragment_id=next_id,
                    wire_names=tuple(spec.wire_name for spec, _start, _stop in members),
                    numel=sum(stop - start for _spec, start, stop in members),
                    **({"flat_ranges": tuple((start, stop) for _spec, start, stop in members)} if split else {}),
                )
            )
            next_id += 1
        plans.append(
            FullParameterOwnerFragmentPlan(
                topology=manifest.topology, parameter_layout_hash=_LAYOUT_HASH, fragments=tuple(fragments)
            )
        )
    return tuple(plans)


def _target(cut, ray, policy_version, values_for):
    """``cut`` at ``policy_version`` with the fragments ``values_for`` names replaced."""

    shards = []
    for shard in cut.shards:
        fragments, refs = list(shard.fragments), [list(row) for row in shard.chunk_refs]
        for index, descriptor in enumerate(shard.fragments):
            values = values_for(descriptor)
            if values is not None:
                payload = values.numpy().tobytes()
                fragments[index], refs[index] = store_full_parameter_fragment_payload(
                    cut, descriptor.fragment_id, payload, ray_module=ray
                )
        shards.append(
            replace(
                shard,
                policy_version=policy_version,
                local_step_generation=0,
                fragments=tuple(fragments),
                chunk_refs=refs,
            )
        )
    return FullParameterChunkedCut(
        policy_version=policy_version,
        local_step_generation=0,
        parameter_layout_hash=cut.parameter_layout_hash,
        shards=tuple(shards),
    )


def _merged(descriptor) -> torch.Tensor:
    """A distinct 'merged' payload for every fragment."""

    return torch.arange(descriptor.numel, dtype=torch.float32) * -0.5 + 100.0 * (descriptor.fragment_id + 1)


async def _install(group):
    manifests = await shard_manifests(group)
    plans = _plans(manifests)
    assert await install_fragment_plans(group, plans) == sum(len(plan.fragments) for plan in plans)
    return manifests, plans


def _assert_replicas_identical(ranks):
    """Every rank holding a parameter holds the same master and model bits."""

    seen_master, seen_model = {}, {}
    for rank in ranks:
        for name, master in rank.masters().items():
            bits = master.detach().clone().view(torch.int32)
            if name in seen_master:
                assert torch.equal(bits, seen_master[name]), name
            seen_master[name] = bits
        for name, model in rank.model().items():
            bits = model.clone().view(torch.int16)
            if name in seen_model:
                assert torch.equal(bits, seen_model[name]), name
            seen_model[name] = bits
            assert torch.equal(model, rank.masters()[name].to(torch.bfloat16)), name
    return seen_master


@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2), (4, 4)])
async def test_owners_cover_every_parameter_exactly_once(cluster, world, ep):
    group, ranks, _ray = cluster(world, ep)
    manifests = await shard_manifests(group)

    owned = Counter(spec.name for manifest in manifests for spec in manifest.specs)
    held = {name for rank in ranks for name, _parameter in rank.named}
    assert set(owned) == held and set(owned.values()) == {1}
    # Every rank owns part of the dense set (balanced ownership), and the
    # wire names are those of the owner's topology.
    for manifest in manifests:
        assert any(".experts." not in spec.name for spec in manifest.specs)
        assert all(spec.shard_id == manifest.topology.shard_id for spec in manifest.specs)
    # Each holding names the same owner on every replica.
    owners = {}
    for manifest in manifests:
        for holding in manifest.holdings:
            assert owners.setdefault(holding.name, holding) == holding
    expected_routing = {manifest.topology for manifest in manifests if manifest.replicated_holdings()}
    assert set(replica_owner_manifests(manifests)) == expected_routing == {rank.topology for rank in ranks}


async def test_dp1_manifest_is_unchanged_by_replication(cluster):
    group, ranks, _ray = cluster(1, 1)
    (manifest,) = await shard_manifests(group)
    assert {spec.name for spec in manifest.specs} == {name for name, _parameter in ranks[0].named}
    assert manifest.replicated_holdings() == ()
    assert manifest.layout_hash == fps._layout_hash(manifest.topology, manifest.specs)
    assert replica_owner_manifests((manifest,)) == {}


@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2)])
async def test_every_replica_applies_the_merged_fragments_identically(cluster, world, ep):
    group, ranks, ray = cluster(world, ep)
    manifests, plans = await _install(group)
    exported = await export_chunked_cut(group, 0, 0, max_chunk_bytes=16)

    target = _target(exported, ray, 1, _merged)
    applied = await apply_chunked_cut(group, target, commit_token="dp-apply-1")
    assert applied == sum(len(manifest.specs) for manifest in manifests)

    masters = _assert_replicas_identical(ranks)
    # The values are the merged payloads, read through the owners' plans.
    numel = {spec.wire_name: spec.numel for manifest in manifests for spec in manifest.specs}
    name_of = {spec.wire_name: spec.name for manifest in manifests for spec in manifest.specs}
    expected = {name: torch.full((numel[wire],), float("nan")) for wire, name in name_of.items()}
    for plan in plans:
        for fragment in plan.fragments:
            payload, offset = _merged(SimpleNamespace(numel=fragment.numel, fragment_id=fragment.fragment_id)), 0
            for wire, start, stop in fragment.members(numel):
                expected[name_of[wire]][start:stop] = payload[offset : offset + stop - start]
                offset += stop - start
    for name, bits in masters.items():
        assert torch.equal(bits.view(torch.float32).reshape(-1), expected[name]), name
    # Adam state objects survive; every rank proves the same progress.
    proofs = await optimizer_states(group)
    assert {proof.installed_policy_version for proof in proofs} == {1}
    assert all(
        proof.model_master_parameter_count == len(rank.named) for proof, rank in zip(proofs, ranks, strict=True)
    )


async def test_skip_resident_after_deferred_export_reads_only_changed_fragments(cluster):
    group, ranks, ray = cluster(4, 2)
    manifests, plans = await _install(group)
    first = await export_chunked_cut(group, 0, 0, max_chunk_bytes=16)
    await apply_chunked_cut(group, _target(first, ray, 1, _merged), commit_token="dp-apply-1")

    # One identical local step on every replica, then the boundary's deferred export.
    for rank in ranks:
        rank.local_step(rollout_id=1, delta=0.25)
    await record_local_step(group, base_policy_version=1, rollout_id=1)
    deferred = await export_chunked_cut(group, 1, 1, max_chunk_bytes=16, defer_payload=True)
    before = _assert_replicas_identical(ranks)

    # Only fragment 0 (an owner's dense parameters) changed; the rest stays deferred.
    changed = 0
    target = _target(
        deferred, ray, 2, lambda descriptor: _merged(descriptor) * 3 if descriptor.fragment_id == changed else None
    )
    await apply_chunked_cut(group, target, commit_token="dp-apply-2", skip_resident_fragments=True)

    after = _assert_replicas_identical(ranks)
    owner_plan = next(plan for plan in plans if any(fragment.fragment_id == changed for fragment in plan.fragments))
    changed_names = {
        wire.split("::")[2]
        for fragment in owner_plan.fragments
        if fragment.fragment_id == changed
        for wire in fragment.wire_names
    }
    for name in before:
        assert torch.equal(after[name], before[name]) == (name not in changed_names), name
    replica_readers = 0
    for rank in ranks:
        stats = rank._external_full_parameter_apply_stats
        owns_changed = rank.topology == owner_plan.topology
        replicates_changed = not owns_changed and any(name in changed_names for name, _parameter in rank.named)
        replica_readers += replicates_changed
        assert stats["prepare"]["read_fragments"] == (1 if owns_changed else 0)
        # A rank replicating a member of the changed fragment reads that one fragment, no other.
        assert (
            stats["prepare"]["replica_read_fragments"]
            == stats["commit"]["replica_copied_fragments"]
            == replicates_changed
        )
        assert stats["prepare"]["replica_resident_fragments"] == stats["commit"]["replica_resident_fragments"]
    assert replica_readers >= 1

    # The owners can still materialize unchanged deferred fragments; replicas are not asked.
    third_export = await export_chunked_cut(group, 2, 0, max_chunk_bytes=16, defer_payload=True)
    stored = await materialize_chunked_fragments(group, third_export, [1, 2])
    assert sorted(stored) == [1, 2]


async def test_diverged_replica_is_refused_when_checked(cluster, monkeypatch):
    group, ranks, ray = cluster(2, 2)
    await _install(group)
    first = await export_chunked_cut(group, 0, 0, max_chunk_bytes=16)
    await apply_chunked_cut(group, _target(first, ray, 1, _merged), commit_token="dp-apply-1")
    for rank in ranks:
        rank.local_step(rollout_id=1, delta=0.25)
    await record_local_step(group, base_policy_version=1, rollout_id=1)

    monkeypatch.setenv(VERIFY_REPLICAS_ENV, "1")
    checked = await export_chunked_cut(group, 1, 1, max_chunk_bytes=16, defer_payload=True)
    assert all(rank._external_full_parameter_apply_stats["replica_export"]["verified_chunks"] > 0 for rank in ranks)

    # Rank 1 holds a replica of a dense parameter rank 0 owns; make it drift.
    # The check hashes only chunks that lie wholly inside rank 1's replicas;
    # the largest replica spans at least one such 4-scalar chunk.
    manifests = await shard_manifests(group)
    replica = max(
        manifests[1].replicated_holdings(), key=lambda holding: (holding.shape[0] * holding.shape[-1], holding.name)
    ).name
    ranks[1].masters()[replica].add_(1.0)
    with pytest.raises(RuntimeError, match="replica ranks refused the full-parameter export note"):
        await export_chunked_cut(group, 1, 1, max_chunk_bytes=16, defer_payload=True)
    assert checked.policy_version == 1


async def test_inconsistent_ownership_is_refused(cluster):
    group, _ranks, _ray = cluster(2, 2)
    manifests = await shard_manifests(group)
    holding = manifests[1].replicated_holdings()[0]
    # A rank that names itself the owner of a parameter without listing its spec.
    wrong = replace(holding, owner_rank=1)
    with pytest.raises(ValueError, match="owned holdings"):
        replace(
            manifests[1],
            holdings=tuple(sorted(wrong if item.name == holding.name else item for item in manifests[1].holdings)),
        )
    # A replica-group member that does not hold the owner's parameter.
    dropped = replace(
        manifests[1], holdings=tuple(item for item in manifests[1].holdings if item.name != holding.name)
    )
    with pytest.raises(RuntimeError, match="does not hold it with the same owner"):
        replica_owner_manifests((manifests[0], dropped))


async def test_apply_without_replica_states_is_refused_on_replica_ranks(cluster):
    group, _ranks, ray = cluster(2, 2)
    await _install(group)
    exported = await export_chunked_cut(group, 0, 0, max_chunk_bytes=16)
    # A transport that forgot the routing must not let replicas silently skip.
    setattr(group, transport.REPLICA_OWNERS_ATTRIBUTE, {})
    with pytest.raises(RuntimeError, match="rejected full-parameter prepare"):
        await apply_chunked_cut(group, _target(exported, ray, 1, _merged), commit_token="no-routing")
