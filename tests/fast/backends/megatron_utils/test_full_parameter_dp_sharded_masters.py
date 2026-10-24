"""Full-parameter sync with sharded FP32 masters (distributed optimizer, DP>1), on CPU.

Two fake clusters of the same model and topology run side by side in one
process, behind fake trainer groups whose ``run_plugin`` calls the real
per-rank entry point for each rank in turn:

* ``whole``: the non-distributed optimizer, a whole FP32 master per parameter
  on every rank (the data-parallel replica path);
* ``sharded``: a stand-in for Megatron's distributed optimizer.  Every replica
  group (dense: all ranks; experts: the ranks with the same EP rank) keeps its
  parameters in one flat buffer per dtype, padded to a multiple of the group
  size and split contiguously: each rank steps the FP32 ``main_param`` of its
  range (``shard_float16_groups`` views of the model), a native-FP32 parameter
  through a ``shard_fp32_groups`` view, and a step all-gathers the model.

The tests prove that the sharded cluster has the same layout, exports the same
bytes and applies the same bits as the whole one, and that the cases it cannot
serve are refused.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_sharded_masters as sharded
from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH, full_parameter_rank_call
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
    iter_full_parameter_fragment_parts,
    materialize_chunked_fragments,
    optimizer_states,
    record_local_step,
    shard_manifests,
    store_full_parameter_fragment_payload,
)

pytestmark = pytest.mark.asyncio

_LAYOUT_HASH = "9" * 64
_EXPERTS_PER_RANK = 2
_DENSE = {
    "embedding.word_embeddings.weight": (6, 3),
    "decoder.layers.0.self_attention.linear_qkv.weight": (4, 3),
    "decoder.layers.0.mlp.router.weight": (4, 3),
    "decoder.layers.0.input_layernorm.weight": (3,),
    "decoder.layers.0.mixer.A_log": (5,),  # native FP32
    "decoder.final_layernorm.weight": (3,),
    "output_layer.weight": (5, 3),
}
_NATIVE_FP32 = {"decoder.layers.0.mixer.A_log"}
_EXPERT_SHAPES = {"linear_fc1": (4, 3), "linear_fc2": (3, 2)}
_SPLIT_NUMEL = 16  # a parameter above this many scalars spans two fragments
_CHUNK_BYTES = 16


def _initial_value(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    seed = sum(ord(character) * (index + 1) for index, character in enumerate(name)) % 997
    return (torch.arange(math.prod(shape), dtype=torch.float32) * 0.1 + seed / 7.0).reshape(shape)


def _expert_names(ep_rank: int) -> dict[str, tuple[int, ...]]:
    names = {}
    for local in range(_EXPERTS_PER_RANK):
        expert = ep_rank * _EXPERTS_PER_RANK + local
        for projection, shape in _EXPERT_SHAPES.items():
            names[f"decoder.layers.0.mlp.experts.local_experts.{expert}.{projection}.weight"] = shape
    return names


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name in _NATIVE_FP32 else torch.bfloat16


class _Rank:
    """One fake Megatron rank; ``sharded`` picks the optimizer layout."""

    def __init__(self, rank: int, *, world: int, ep: int, sharded_masters: bool) -> None:
        self.rank = rank
        self.role = "actor"
        self.topology = FullParameterTopology(0, 1, 0, 1, rank % ep, ep, 0, 1, rank, world)
        self.replication = FullParameterReplication(
            global_rank=rank,
            dense_ranks=tuple(range(world)),
            expert_ranks=tuple(other for other in range(world) if other % ep == rank % ep),
        )
        self.groups = {
            "dense": (self.replication.dense_ranks, dict(_DENSE)),
            "expert": (self.replication.expert_ranks, _expert_names(rank % ep)),
        }
        self.args = SimpleNamespace(
            lora_rank=0, lora_adapter_path=None, num_steps_per_rollout=1, global_batch_size=1, start_rollout_id=None
        )
        self.opt_param_scheduler = SimpleNamespace(num_steps=0)
        self.weights_backuper = SimpleNamespace(backup=lambda role: None)
        self._last_rollout_id = 0
        self.sharded_masters = sharded_masters
        self.ranges: dict[str, tuple[int, int]] = {}
        if sharded_masters:
            self.named, self.optimizer = self._distributed_optimizer()
        else:
            self.named, self.optimizer = self._whole_optimizer()

    def _whole_optimizer(self):
        named, owned = [], []
        for _group, (_ranks, shapes) in self.groups.items():
            for name, shape in shapes.items():
                parameter = torch.nn.Parameter(_initial_value(name, shape).to(_dtype(name)))
                if name in _NATIVE_FP32:
                    owned.append(parameter)
                else:
                    parameter.main_param = _initial_value(name, shape)
                    owned.append(parameter.main_param)
                if ".experts." in name:
                    parameter.allreduce = False
                named.append((name, parameter))
                self.ranges[name] = (0, parameter.numel())
        return tuple(named), SimpleNamespace(optimizer=_base(owned))

    def _distributed_optimizer(self):
        named, wrappers = [], []
        for group, (ranks, shapes) in self.groups.items():
            ordered = sorted(shapes.items())
            wrapper = SimpleNamespace(
                model_float16_groups=[[]], shard_float16_groups=[[]], model_fp32_groups=[[]], shard_fp32_groups=[[]]
            )
            owned = []
            for dtype in (torch.bfloat16, torch.float32):
                members = [(name, shape) for name, shape in ordered if _dtype(name) == dtype]
                total = sum(math.prod(shape) for _name, shape in members)
                shard = -(-total // len(ranks))  # the bucket padded to a multiple of the group
                low = ranks.index(self.rank) * shard
                buffer = torch.zeros(shard * len(ranks), dtype=dtype)
                offset = 0
                for name, shape in members:
                    numel = math.prod(shape)
                    parameter = torch.nn.Parameter(buffer[offset : offset + numel].view(shape))
                    parameter.data.copy_(_initial_value(name, shape).to(dtype))
                    if group == "expert":
                        parameter.allreduce = False
                    start, stop = max(low, offset) - offset, min(low + shard, offset + numel) - offset
                    self.ranges[name] = (start, stop) if start < stop else (0, 0)
                    if start < stop:
                        view = parameter.detach().view(-1)[start:stop]
                        if dtype == torch.float32:
                            wrapper.model_fp32_groups[0].append(parameter)
                            wrapper.shard_fp32_groups[0].append(view)
                            owned.append(view)
                        else:
                            parameter.main_param = _initial_value(name, shape).view(-1)[start:stop].clone()
                            wrapper.model_float16_groups[0].append(parameter)
                            wrapper.shard_float16_groups[0].append(view)
                            owned.append(parameter.main_param)
                    elif dtype == torch.bfloat16:
                        parameter.main_param = None
                    if dtype == torch.bfloat16:
                        parameter.main_param_sharded = True
                    named.append((name, parameter))
                    offset += numel
            wrapper.optimizer = _base(owned)
            wrappers.append(wrapper)
        return tuple(named), SimpleNamespace(chained_optimizers=wrappers)

    def local_range_values(self) -> dict[str, torch.Tensor]:
        """The FP32 values this rank's optimizer steps, by name."""

        values = {}
        for name, parameter in self.named:
            start, stop = self.ranges[name]
            if start == stop:
                continue
            if name in _NATIVE_FP32:
                values[name] = parameter.detach().view(-1)[start:stop]
            elif getattr(parameter, "main_param_sharded", False):
                values[name] = parameter.main_param
            else:
                values[name] = parameter.main_param.view(-1)
        return values

    def model_bits(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone().view(torch.int32 if name in _NATIVE_FP32 else torch.int16)
            for name, parameter in self.named
        }


def _base(owned):
    return SimpleNamespace(
        param_groups=[{"params": owned}],
        state={tensor: {"exp_avg": torch.ones_like(tensor), "exp_avg_sq": torch.ones_like(tensor)} for tensor in owned},
    )


def _full_masters(ranks) -> dict[str, torch.Tensor]:
    """Every parameter's whole FP32 master: put together from the ranks'
    ranges (sharded), or the replicas' identical copies (whole)."""

    full: dict[str, torch.Tensor] = {}
    for rank in ranks:
        for name, values in rank.local_range_values().items():
            numel = dict(rank.named)[name].numel()
            if not rank.sharded_masters:
                assert name not in full or torch.equal(full[name].view(torch.int32), values.view(torch.int32)), name
                full[name] = values.detach().clone()
                continue
            target = full.setdefault(name, torch.full((numel,), float("nan")))
            start, stop = rank.ranges[name]
            assert torch.isnan(target[start:stop]).all(), f"{name} has two FP32 masters"
            target[start:stop] = values
    assert all(not torch.isnan(value).any() for value in full.values())
    return full


def _local_step(ranks, rollout_id: int, delta: float) -> None:
    """One identical optimizer step on every rank, then the model all-gather."""

    for rank in ranks:
        for _name, values in rank.local_range_values().items():
            values.add_(delta)
    full = _full_masters(ranks)
    for rank in ranks:
        for name, parameter in rank.named:
            parameter.data.copy_(full[name].view(parameter.shape).to(parameter.dtype))
        rank.opt_param_scheduler.num_steps += 1
        rank._last_rollout_id = rollout_id


class _FakeRay:
    def __init__(self) -> None:
        self.objects = {}

    def put(self, value):
        reference = object()
        self.objects[reference] = value.detach().clone()
        return reference

    def get(self, reference):
        return self.objects[reference]


class _Group:
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
    """``make(world, ep, sharded_masters)`` -> (group, ranks, fake ray)."""

    current = {}
    rays = {}
    monkeypatch.setattr(fps, "full_parameter_topology", lambda: current["rank"].topology)
    monkeypatch.setattr(fps, "full_parameter_replication", lambda: current["rank"].replication)
    monkeypatch.setattr(fps, "_named_actor_parameters", lambda actor: actor.named)
    # Megatron's main-to-model copy of the non-distributed optimizer (whole masters).
    monkeypatch.setattr(
        fps,
        "_copy_masters_to_model",
        lambda actor: [
            parameter.data.copy_(parameter.main_param.to(parameter.dtype))
            for name, parameter in actor.named
            if name not in _NATIVE_FP32
        ],
    )
    monkeypatch.setattr(
        fps, "_resolve_ray_module", lambda module: module if module is not None else rays[id(current["rank"].cluster)]
    )
    monkeypatch.setattr(
        transport, "_resolve_ray_module", lambda module: module if module is not None else rays[id(current["group"])]
    )

    def make(world: int, ep: int, *, sharded_masters: bool):
        ranks = [_Rank(rank, world=world, ep=ep, sharded_masters=sharded_masters) for rank in range(world)]
        group = _Group(ranks, current)
        rays[id(group)] = _FakeRay()
        for rank in ranks:
            rank.cluster = group
        current["group"] = group
        return group, ranks, rays[id(group)]

    def use(group) -> None:
        current["group"] = group

    make.use = use
    return make


def _plans(manifests) -> tuple[FullParameterOwnerFragmentPlan, ...]:
    """Each rank's specs in canonical order: fragments of at most two whole
    parameters; a parameter above ``_SPLIT_NUMEL`` split in half."""

    plans, next_id = [], 0
    for manifest in sorted(manifests, key=lambda value: value.topology):
        groups, pending = [], []
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
    shards = []
    for shard in cut.shards:
        fragments, refs = list(shard.fragments), [list(row) for row in shard.chunk_refs]
        for index, descriptor in enumerate(shard.fragments):
            values = values_for(descriptor)
            if values is not None:
                fragments[index], refs[index] = store_full_parameter_fragment_payload(
                    cut, descriptor.fragment_id, values.numpy().tobytes(), ray_module=ray
                )
        shards.append(
            replace(shard, policy_version=policy_version, local_step_generation=0, fragments=tuple(fragments), chunk_refs=refs)
        )
    return FullParameterChunkedCut(
        policy_version=policy_version,
        local_step_generation=0,
        parameter_layout_hash=cut.parameter_layout_hash,
        shards=tuple(shards),
    )


def _merged(descriptor) -> torch.Tensor:
    return torch.arange(descriptor.numel, dtype=torch.float32) * -0.37 + 10.3 * (descriptor.fragment_id + 1)


async def _pair(cluster, world: int, ep: int):
    """Both clusters with their plans installed: ``{"whole": ..., "sharded": ...}``."""

    pair = {}
    for name in ("whole", "sharded"):
        group, ranks, ray = cluster(world, ep, sharded_masters=name == "sharded")
        manifests = await shard_manifests(group)
        assert await install_fragment_plans(group, _plans(manifests)) > 0
        pair[name] = SimpleNamespace(group=group, ranks=ranks, ray=ray, manifests=manifests)
    return pair


async def _step_and_record(cluster, pair, *, rollout_id: int, base: int, delta: float) -> None:
    for side in pair.values():
        cluster.use(side.group)
        _local_step(side.ranks, rollout_id, delta)
        await record_local_step(side.group, base_policy_version=base, rollout_id=rollout_id)


def _fragment_bytes(cut, ray, descriptors=None) -> dict[int, bytes]:
    """Every fragment's stored payload, read through ``descriptors`` (a lazy
    cut's content descriptors) or the cut's own rows."""

    if descriptors is not None:
        cut = replace(
            cut,
            shards=tuple(
                replace(shard, fragments=tuple(descriptors[row.fragment_id] for row in shard.fragments))
                for shard in cut.shards
            ),
        )
    return {
        row.fragment_id: b"".join(
            bytes(part) for part in iter_full_parameter_fragment_parts(cut, row.fragment_id, ray_module=ray)
        )
        for shard in cut.shards
        for row in shard.fragments
    }


def _assert_same_bits(pair) -> None:
    """Same whole FP32 masters and the same model bits on every rank of both clusters."""

    whole, sharded_full = _full_masters(pair["whole"].ranks), _full_masters(pair["sharded"].ranks)
    assert whole.keys() == sharded_full.keys()
    for name in whole:
        assert torch.equal(whole[name].view(torch.int32), sharded_full[name].view(torch.int32)), name
    for whole_rank, sharded_rank in zip(pair["whole"].ranks, pair["sharded"].ranks, strict=True):
        whole_bits, sharded_bits = whole_rank.model_bits(), sharded_rank.model_bits()
        assert whole_bits.keys() == sharded_bits.keys()
        for name in whole_bits:
            assert torch.equal(whole_bits[name], sharded_bits[name]), name


@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2), (4, 1)])
async def test_sharded_masters_keep_the_layout_and_plans(cluster, world, ep):
    pair = await _pair(cluster, world, ep)
    for whole, sharded_manifest in zip(pair["whole"].manifests, pair["sharded"].manifests, strict=True):
        assert sharded_manifest.sharded_masters and not whole.sharded_masters
        assert replace(sharded_manifest, sharded_masters=False) == whole
    assert [plan.plan_hash for plan in _plans(pair["whole"].manifests)] == [
        plan.plan_hash for plan in _plans(pair["sharded"].manifests)
    ]
    # Some parameter really is split across ranks, and some rank holds none of one.
    ranks = pair["sharded"].ranks
    assert any(0 < stop - start < dict(rank.named)[name].numel() for rank in ranks for name, (start, stop) in rank.ranges.items())
    assert any(start == stop for rank in ranks for start, stop in rank.ranges.values())


async def test_dp1_distributed_optimizer_keeps_whole_masters(cluster):
    group, _ranks, _ray = cluster(1, 1, sharded_masters=True)
    (manifest,) = await shard_manifests(group)
    whole_group, _whole_ranks, _whole_ray = cluster(1, 1, sharded_masters=False)
    (whole,) = await shard_manifests(whole_group)
    assert not manifest.sharded_masters
    assert manifest == whole


@pytest.mark.parametrize("mode", ["full", "deferred", "lazy"])
@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2)])
async def test_exports_are_byte_identical_to_whole_masters(cluster, mode, world, ep):
    pair = await _pair(cluster, world, ep)
    await _step_and_record(cluster, pair, rollout_id=1, base=0, delta=0.375)
    flags = {"full": {}, "deferred": {"defer_payload": True}, "lazy": {"defer_payload": True, "lazy": True}}[mode]
    stored = {}
    for name, side in pair.items():
        cluster.use(side.group)
        cut = await export_chunked_cut(side.group, 0, 1, max_chunk_bytes=_CHUNK_BYTES, **flags)
        fragment_ids = [descriptor.fragment_id for shard in cut.shards for descriptor in shard.fragments]
        descriptors = None
        if mode != "full":
            materialized = await materialize_chunked_fragments(
                side.group, cut, fragment_ids, **({"return_descriptors": True} if mode == "lazy" else {})
            )
            for shard in cut.shards:
                for descriptor, refs in zip(shard.fragments, shard.chunk_refs, strict=True):
                    row = materialized[descriptor.fragment_id]
                    refs[:] = row[1] if mode == "lazy" else row
            if mode == "lazy":
                descriptors = {fragment_id: row[0] for fragment_id, row in materialized.items()}
        stored[name] = (cut, _fragment_bytes(cut, side.ray, descriptors), descriptors)
    (whole_cut, whole_bytes, whole_lazy), (sharded_cut, sharded_bytes, sharded_lazy) = stored["whole"], stored["sharded"]
    assert whole_bytes == sharded_bytes
    if mode == "lazy":
        assert whole_lazy == sharded_lazy  # the same content descriptors
    else:
        assert [shard.fragments for shard in whole_cut.shards] == [shard.fragments for shard in sharded_cut.shards]
    contributed = [rank._external_full_parameter_apply_stats["contribute"] for rank in pair["sharded"].ranks]
    assert sum(stats["scalars"] for stats in contributed) > 0


@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2), (4, 1)])
async def test_applies_are_bit_identical_to_whole_masters(cluster, world, ep):
    pair = await _pair(cluster, world, ep)
    proofs_before = {}
    for side in pair.values():
        cluster.use(side.group)
        exported = await export_chunked_cut(side.group, 0, 0, max_chunk_bytes=_CHUNK_BYTES)
        await apply_chunked_cut(side.group, _target(exported, side.ray, 1, _merged), commit_token="apply-1")
    _assert_same_bits(pair)

    # A local step, a deferred export, and a skip-resident apply that changes one fragment.
    await _step_and_record(cluster, pair, rollout_id=1, base=1, delta=-0.625)
    for name, side in pair.items():
        cluster.use(side.group)
        proofs_before[name] = await optimizer_states(side.group)
        deferred = await export_chunked_cut(side.group, 1, 1, max_chunk_bytes=_CHUNK_BYTES, defer_payload=True)
        target = _target(deferred, side.ray, 2, lambda d: _merged(d) * 3 if d.fragment_id == 0 else None)
        await apply_chunked_cut(side.group, target, commit_token="apply-2", skip_resident_fragments=True)
    _assert_same_bits(pair)
    for whole_rank, sharded_rank in zip(pair["whole"].ranks, pair["sharded"].ranks, strict=True):
        for phase, key in (("prepare", "read_fragments"), ("prepare", "replica_read_fragments"), ("commit", "replica_copied_fragments")):
            assert (
                whole_rank._external_full_parameter_apply_stats[phase][key]
                == sharded_rank._external_full_parameter_apply_stats[phase][key]
            )
        assert sharded_rank._external_full_parameter_apply_stats["commit"]["sharded"]

    # Adam state survives on every rank, and the model matches each local range.
    cluster.use(pair["sharded"].group)
    proofs = await optimizer_states(pair["sharded"].group)
    for proof, before, rank in zip(proofs, proofs_before["sharded"], pair["sharded"].ranks, strict=True):
        assert proof.installed_policy_version == 2
        assert proof.selected_state_sha256 == before.selected_state_sha256
        assert proof.model_master_parameter_count == len(rank.named)

    # The next boundary exports the same bytes again.
    await _step_and_record(cluster, pair, rollout_id=2, base=2, delta=0.25)
    cuts = {}
    for name, side in pair.items():
        cluster.use(side.group)
        cuts[name] = await export_chunked_cut(side.group, 2, 1, max_chunk_bytes=_CHUNK_BYTES)
    assert [shard.fragments for shard in cuts["whole"].shards] == [shard.fragments for shard in cuts["sharded"].shards]


async def test_an_export_without_every_contribution_is_refused(cluster, monkeypatch):
    group, ranks, _ray = cluster(4, 2, sharded_masters=True)
    await install_fragment_plans(group, _plans(await shard_manifests(group)))
    real = transport._master_contributions

    async def drop_one_contributor(*args, **kwargs):
        by_owner = await real(*args, **kwargs)
        return {owner: tuple(c for c in values if c.contributor != ranks[1].topology) for owner, values in by_owner.items()}

    monkeypatch.setattr(transport, "_master_contributions", drop_one_contributor)
    with pytest.raises(RuntimeError, match="incomplete chunked cut") as error:
        await export_chunked_cut(group, 0, 0, max_chunk_bytes=_CHUNK_BYTES)
    assert "covered exactly once" in str(error.value.__cause__)


async def test_materialize_after_the_ranges_changed_is_refused(cluster):
    group, ranks, _ray = cluster(4, 2, sharded_masters=True)
    await install_fragment_plans(group, _plans(await shard_manifests(group)))
    cut = await export_chunked_cut(group, 0, 0, max_chunk_bytes=_CHUNK_BYTES, defer_payload=True)
    _local_step(ranks, rollout_id=1, delta=1.0)
    await record_local_step(group, base_policy_version=0, rollout_id=1)
    every = [row.fragment_id for shard in cut.shards for row in shard.fragments]
    with pytest.raises(RuntimeError, match="refused to contribute"):
        await materialize_chunked_fragments(group, cut, every)


async def test_contributions_from_another_optimizer_state_are_refused(cluster):
    group, ranks, _ray = cluster(2, 2, sharded_masters=True)
    await install_fragment_plans(group, _plans(await shard_manifests(group)))
    contributions = await transport._master_contributions(group, {rank.topology: None for rank in ranks})
    _local_step(ranks, rollout_id=1, delta=1.0)
    await record_local_step(group, base_policy_version=0, rollout_id=1)
    owner = ranks[0]
    group._current["rank"] = owner
    with pytest.raises(RuntimeError, match="another optimizer state"):
        fps.export_full_parameter_chunked_shard(
            owner, policy_version=0, local_step_generation=1, contributions=contributions[owner.topology]
        )


def _ranks_for(world: int, ep: int):
    return [_Rank(rank, world=world, ep=ep, sharded_masters=True) for rank in range(world)]


def test_offload_precision_aware_and_param_gather_are_refused(monkeypatch):
    rank = _ranks_for(2, 2)[0]
    owned = fps._optimizer_owned_parameter_ids(rank.optimizer)
    named = tuple(sorted(rank.named))
    assert sharded.master_ranges(named, rank.replication, owned) is not None

    offloaded = fps._OptimizerOwnedParameters(owned, distributed_views=owned.distributed_views, hybrid_copies=True)
    with pytest.raises(RuntimeError, match="no optimizer CPU offload"):
        sharded.master_ranges(named, rank.replication, offloaded)

    precision_aware = {key: (view, view, None) for key, (view, _owned, _fp32) in owned.distributed_views.items()}
    without_main = fps._OptimizerOwnedParameters(owned, distributed_views=precision_aware)
    with pytest.raises(RuntimeError, match="no precision-aware optimizer"):
        sharded.master_ranges(named, rank.replication, without_main)

    with pytest.raises(RuntimeError, match="overlap_param_gather"):
        sharded.refuse_unsupported_args(SimpleNamespace(overlap_param_gather=True))
    sharded.refuse_unsupported_args(SimpleNamespace(overlap_param_gather=False))


async def test_a_group_mixing_whole_and_sharded_masters_is_refused(cluster):
    group, ranks, _ray = cluster(2, 2, sharded_masters=True)
    manifests = await shard_manifests(group)
    mixed = (manifests[0], replace(manifests[1], sharded_masters=False))
    with pytest.raises(RuntimeError, match="disagree on sharded"):
        transport._sharded_master_topologies(mixed)
