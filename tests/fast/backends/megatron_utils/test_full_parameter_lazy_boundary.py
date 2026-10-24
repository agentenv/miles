"""A per-fragment round boundary: lazy export, labelled rows, partial commit.

A lazy export reads no master and gives every fragment the real chunk
geometry with label hashes.  Materializing a labelled fragment yields exactly
the content descriptor and bytes of the full export, and checks its model
parameters against their masters.  An apply on labelled rows, and a commit
that finishes over the written parameters only, must leave the masters, the
model and the hybrid optimizer copies bit-identical to the whole-model path.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    commit_prepared_full_parameter_shard,
    export_full_parameter_chunked_shard,
    finalize_full_parameter_commit,
    full_parameter_shard_manifest,
    install_full_parameter_fragment_plan,
    is_deferred_full_parameter_chunk,
    materialize_full_parameter_fragments,
    prepare_full_parameter_chunked_shard,
    record_full_parameter_local_step,
)
from miles.ray.full_parameter_transport import (
    FULL_PARAMETER_TRANSPORT_FEATURES,
    FullParameterChunkedCut,
    export_chunked_cut,
    iter_full_parameter_authoritative_fragment_delta_parts,
    materialize_chunked_fragments,
    record_local_step,
    store_full_parameter_fragment_payload,
)
from miles.utils.tensor_backper import TensorBackuper
from tests.fast.backends.megatron_utils.test_full_parameter_deferred_export import _PluginGroup
from tests.fast.backends.megatron_utils.test_full_parameter_resident_apply import (
    _apply,
    _CountingRay,
    _master_bits,
    _model_bits,
    _split_actor,
    _target,
)
from tests.fast.backends.megatron_utils.test_full_parameter_state import _distopt_actor


def _export(actor, ray, *, lazy=False, defer=True):
    return export_full_parameter_chunked_shard(actor, policy_version=0, local_step_generation=0,
                                               max_chunk_bytes=8, ray_module=ray, defer_payload=defer, lazy=lazy)


def _geometry(descriptor):
    return [(c.chunk_index, c.flat_offset, c.numel) for c in descriptor.chunks]


# -- lazy export --------------------------------------------------------------


def test_lazy_export_reads_no_master_and_labels_every_fragment(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    full = _export(actor, _CountingRay(), defer=False)
    named[0][1].main_param[0, 1] = float("nan")  # no master read: not seen
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)

    assert ray.objects == {}
    assert all(is_deferred_full_parameter_chunk(ref) for refs in lazy.chunk_refs for ref in refs)
    for labelled, content in zip(lazy.fragments, full.fragments, strict=True):
        assert _geometry(labelled) == _geometry(content) and labelled.numel == content.numel
        assert labelled.payload_hash != content.payload_hash
        assert {c.payload_hash for c in labelled.chunks}.isdisjoint(c.payload_hash for c in content.chunks)
    record = actor._external_full_parameter_resident
    assert record.lazy_fragment_ids == {0, 1, 2}
    assert actor._external_full_parameter_apply_stats["export"]["lazy"] is True
    # Every lazy export names a new export of the masters.
    assert _export(actor, _CountingRay(), lazy=True).fragments != lazy.fragments


def test_a_lazy_export_must_defer_its_payload(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    with pytest.raises(ValueError, match="must defer"):
        _export(actor, _CountingRay(), lazy=True, defer=False)


# -- materialize --------------------------------------------------------------


def test_materializing_a_labelled_fragment_gives_the_full_export_descriptor_and_bytes(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    full_ray = _CountingRay()
    full = _export(actor, full_ray, defer=False)
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)
    record = actor._external_full_parameter_resident

    with pytest.raises(RuntimeError, match="return_descriptors=True"):
        materialize_full_parameter_fragments(actor, state=lazy, fragment_ids=[1], ray_module=ray)
    stored = materialize_full_parameter_fragments(actor, state=lazy, fragment_ids=[2, 1], ray_module=ray,
                                                  return_descriptors=True)

    assert [row[0] for row in stored] == [1, 2]
    for fragment_id, refs, descriptor in stored:
        assert descriptor == full.fragments[fragment_id]
        want = [full_ray.objects[ref] for ref in full.chunk_refs[fragment_id]]
        assert all(torch.equal(ray.objects[ref], value) for ref, value in zip(refs, want, strict=True))
    assert actor._external_full_parameter_resident is record
    stats = actor._external_full_parameter_apply_stats["materialize"]
    assert stats["lazy_fragments"] == 2 and stats["model_checked_parameters"] == 1  # both halves of proj


def test_materializing_a_labelled_fragment_checks_values_and_the_model(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    norm, proj = named[1][1], named[0][1]
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)
    proj.main_param.view(-1)[4] = float("inf")  # fragment 2, after the export
    with pytest.raises(ValueError, match="NaN or Inf"):
        materialize_full_parameter_fragments(actor, state=lazy, fragment_ids=[2], ray_module=ray,
                                             return_descriptors=True)

    actor, named = _split_actor(monkeypatch)
    norm = named[1][1]
    lazy = _export(actor, ray, lazy=True)
    norm.data[0] += 1.0  # the model of fragment 0 drifted from its master
    with pytest.raises(RuntimeError, match="differs from its optimizer master"):
        materialize_full_parameter_fragments(actor, state=lazy, fragment_ids=[0], ray_module=ray,
                                             return_descriptors=True)


def test_a_labelled_fragment_is_unusable_once_the_record_is_gone(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)
    fps.forget_full_parameter_resident_fragments(actor)  # a local step, restore or commit start
    with pytest.raises(RuntimeError, match="no longer held"):
        materialize_full_parameter_fragments(actor, state=lazy, fragment_ids=[0], ray_module=ray,
                                             return_descriptors=True)
    target = _target(lazy, ray, 1, {0: [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="lazy full-parameter fragments \\[1, 2\\] are not held"):
        prepare_full_parameter_chunked_shard(actor, target, commit_token="t", ray_module=ray, skip_resident=True)


# -- apply on labelled rows ---------------------------------------------------


def _materialize_into(actor, state, ray, fragment_ids):
    rows = materialize_full_parameter_fragments(actor, state=state, fragment_ids=fragment_ids, ray_module=ray,
                                                return_descriptors=True)
    for fragment_id, refs, _descriptor in rows:
        state.chunk_refs[fragment_id][:] = refs
    return {fragment_id: descriptor for fragment_id, _refs, descriptor in rows}


@pytest.mark.parametrize("switch_off", [False, True])
def test_apply_on_labelled_rows_is_bit_identical_to_the_full_path(monkeypatch, switch_off):
    replaced = {1: [30.5, -1.25, 1e-30]}
    actor, full_named = _split_actor(monkeypatch)
    full_ray = _CountingRay()
    full = _export(actor, full_ray, defer=False)
    _apply(actor, _target(full, full_ray, 1, replaced), full_ray, "full", skip=True)
    want_masters, want_model = _master_bits(full_named), _model_bits(full_named)

    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)
    _materialize_into(actor, lazy, ray, [2])  # a labelled row with stored bytes
    if switch_off:
        monkeypatch.setenv("MILES_FULL_PARAMETER_SKIP_RESIDENT", "0")
    count, gets = _apply(actor, _target(lazy, ray, 1, replaced), ray, "lazy", skip=True)

    assert count == len(named)
    assert gets == 4  # only the replaced fragment: labelled rows are never read
    assert all(torch.equal(a, b) for a, b in zip(_master_bits(named), want_masters, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(_model_bits(named), want_model, strict=True))
    # The commit keeps the untouched rows labelled, so they stay usable.
    record = actor._external_full_parameter_resident
    assert record.lazy_fragment_ids == {0, 2}
    committed = _target(lazy, ray, 1, replaced)
    stored = _materialize_into(actor, committed, ray, [0])
    assert stored[0].payload_hash != committed.fragments[0].payload_hash


def test_delta_reads_a_labelled_row_through_its_content_descriptor(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    lazy = _export(actor, ray, lazy=True)
    content = _materialize_into(actor, lazy, ray, [0])[0]
    local = FullParameterChunkedCut(policy_version=0, local_step_generation=1,
                                    parameter_layout_hash=lazy.parameter_layout_hash,
                                    shards=(fps.replace(lazy, local_step_generation=1),))
    anchor_cut = FullParameterChunkedCut(policy_version=0, local_step_generation=0,
                                         parameter_layout_hash=lazy.parameter_layout_hash, shards=(lazy,))
    anchor, anchor_refs = store_full_parameter_fragment_payload(
        anchor_cut, 0, torch.tensor([1.0, 2.0]).numpy().tobytes(), ray_module=ray)

    def delta(**override):
        return b"".join(bytes(part) for part in iter_full_parameter_authoritative_fragment_delta_parts(
            local, 0, authoritative_parameter_layout_hash=lazy.parameter_layout_hash,
            authoritative_topology=lazy.topology, authoritative_plan_hash=lazy.plan_hash,
            authoritative_descriptor=anchor, authoritative_refs=anchor_refs, ray_module=ray, **override))

    got = torch.frombuffer(bytearray(delta(local_descriptor=content)), dtype=torch.float32)
    assert torch.equal(got, torch.tensor([7.0, 8.0]) - torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="hash"):
        delta()  # the label is not a content hash
    with pytest.raises(ValueError, match="does not match its row"):
        delta(local_descriptor=lazy.fragments[1])


# -- partial commit -----------------------------------------------------------


def _forbid_whole_model(monkeypatch):
    def refuse(_actor):
        raise AssertionError("the partial commit used the whole-model copy")

    monkeypatch.setattr(fps, "_copy_masters_to_model", refuse)


def test_a_commit_finishes_over_the_written_parameters_only(monkeypatch):
    replaced = {0: [3.5, -4.25]}  # final_norm only
    actor, full_named = _split_actor(monkeypatch)
    monkeypatch.setenv("MILES_FULL_PARAMETER_PARTIAL_COMMIT", "0")
    full_ray = _CountingRay()
    full = _export(actor, full_ray, defer=False)
    _apply(actor, _target(full, full_ray, 1, replaced), full_ray, "whole", skip=True)
    assert actor._external_full_parameter_apply_stats["commit"]["partial"] is False
    want_masters, want_model = _master_bits(full_named), _model_bits(full_named)

    monkeypatch.delenv("MILES_FULL_PARAMETER_PARTIAL_COMMIT")
    actor, named = _split_actor(monkeypatch)
    _forbid_whole_model(monkeypatch)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=False)
    _apply(actor, _target(exported, ray, 1, replaced), ray, "partial", skip=True)

    assert all(torch.equal(a, b) for a, b in zip(_master_bits(named), want_masters, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(_model_bits(named), want_model, strict=True))
    stats = actor._external_full_parameter_apply_stats["commit"]
    assert stats["partial"] is True and stats["written_parameters"] == 1 and stats["backup"] == "all"
    assert {"propagate_seconds", "model_copy_seconds", "check_seconds", "backup_seconds"} <= set(stats)


def test_writing_every_parameter_finishes_over_the_whole_model(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=False)
    _apply(actor, _target(exported, ray, 1, {0: [1.0, 2.0], 1: [3.0, 4.0, 5.0]}), ray, "all", skip=True)
    stats = actor._external_full_parameter_apply_stats["commit"]
    assert stats["partial"] is False and stats["partial_refusal"] == "every parameter written"


def test_a_partial_commit_still_checks_the_written_model(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    monkeypatch.setattr(fps, "_copy_masters_to_written_parameters", lambda rows, owned: None)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=False)
    with pytest.raises(RuntimeError, match="differs from its optimizer master"):
        _apply(actor, _target(exported, ray, 1, {0: [1.0, 2.0]}), ray, "missed", skip=True)


def _one_fragment_per_parameter(actor):
    manifest = full_parameter_shard_manifest(actor)
    plan = FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="9" * 64,
        fragments=tuple(FullParameterFragmentPlan(index, (spec.wire_name,), spec.numel)
                        for index, spec in enumerate(manifest.specs)),
    )
    install_full_parameter_fragment_plan(actor, plan)
    return manifest


def _distopt_commit(monkeypatch, mode, *, partial):
    if not partial:
        monkeypatch.setenv("MILES_FULL_PARAMETER_PARTIAL_COMMIT", "0")
    actor, named, inner, _state = _distopt_actor(monkeypatch, mode=mode)
    if partial:
        _forbid_whole_model(monkeypatch)
    manifest = _one_fragment_per_parameter(actor)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    proj = next(i for i, spec in enumerate(manifest.specs) if spec.name.endswith("proj.weight"))
    native = next(i for i, spec in enumerate(manifest.specs) if spec.name.endswith("A_log"))
    replaced = {proj: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], native: [-0.75, 0.125]}
    _apply(actor, _target(exported, ray, 1, replaced), ray, f"{mode}-{partial}", skip=True)
    monkeypatch.delenv("MILES_FULL_PARAMETER_PARTIAL_COMMIT", raising=False)
    stats = actor._external_full_parameter_apply_stats["commit"]
    assert stats["partial"] is partial
    model = [parameter.detach().clone().float() for _name, parameter in named]
    copies = [copy.detach().clone().float() for copy in inner.values()]
    owned = [key.detach().clone().float() for key in inner]
    return model, copies, owned


@pytest.mark.parametrize("mode", ["main", "main-offload", "precision-aware"])
def test_distributed_optimizer_partial_commit_matches_the_whole_model(monkeypatch, mode):
    want = _distopt_commit(monkeypatch, mode, partial=False)
    got = _distopt_commit(monkeypatch, mode, partial=True)
    for got_side, want_side in zip(got, want, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(got_side, want_side, strict=True))
    _model, copies, owned = got
    # Every hybrid copy agrees with the tensor the base optimizer steps.
    assert all(torch.equal(a.to(b.dtype), b) or torch.equal(a, b.float()) for a, b in zip(owned, copies, strict=True))


def test_an_uncovered_hybrid_copy_forces_the_whole_model(monkeypatch):
    actor, _named, inner, _state = _distopt_actor(monkeypatch, mode="main-offload")
    stray = torch.zeros(3)
    inner[stray] = torch.ones(3)  # a stepped tensor no named parameter owns
    _one_fragment_per_parameter(actor)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    _apply(actor, _target(exported, ray, 1, {0: [1.0, 2.0]}), ray, "stray", skip=True)
    stats = actor._external_full_parameter_apply_stats["commit"]
    assert stats["partial"] is False and "no named parameter" in stats["partial_refusal"]


def test_backup_follows_the_actor_switch_and_refreshes_written_tensors(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=False)
    actor._enable_weight_backup = False
    _apply(actor, _target(exported, ray, 1, {0: [1.0, 2.0]}), ray, "off", skip=True)
    assert actor.weights_backuper.calls == []
    assert actor._external_full_parameter_apply_stats["commit"]["backup"] == "off"

    actor, named = _split_actor(monkeypatch)
    backuper = TensorBackuper.create(source_getter=lambda: named)
    backuper._backups["actor"] = {name: parameter.detach().clone() for name, parameter in named}
    actor.weights_backuper = backuper
    exported = _export(actor, ray, defer=False)
    _apply(actor, _target(exported, ray, 1, {0: [1.0, 2.0]}), ray, "written", skip=True)
    assert actor._external_full_parameter_apply_stats["commit"]["backup"] == "written"
    for name, parameter in named:
        assert torch.equal(backuper.get("actor")[name], parameter.detach())
    # No complete backup yet: the caller must take a full one.
    assert TensorBackuper.create(source_getter=lambda: named).backup_tensors("actor", {id(named[0][1])}) == 0


# -- local-step receipt and transport -------------------------------------------


def test_a_receipt_without_the_model_check_skips_only_that_check(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    actor._last_rollout_id = 0
    actor.opt_param_scheduler.num_steps = 1
    named[0][1].data.view(-1)[0] += 1.0  # model drifted from its master
    with pytest.raises(RuntimeError, match="differs from its optimizer master"):
        record_full_parameter_local_step(actor, base_policy_version=0, rollout_id=0)
    receipt = record_full_parameter_local_step(actor, base_policy_version=0, rollout_id=0, check_model=False)
    assert receipt.local_step_generation == 1


def test_the_transport_carries_the_lazy_options_through_the_plugin(monkeypatch):
    assert "lazy_export" in FULL_PARAMETER_TRANSPORT_FEATURES
    actor, _named = _split_actor(monkeypatch)
    actor.role = "actor"
    ray = _CountingRay()
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: ray)
    group = _PluginGroup(actor)
    full = _export(actor, _CountingRay(), defer=False)

    async def run():
        cut = await export_chunked_cut(group, 0, 0, max_chunk_bytes=8, defer_payload=True, lazy=True)
        stored = await materialize_chunked_fragments(group, cut, [1], return_descriptors=True)
        actor._last_rollout_id = 0
        actor.opt_param_scheduler.num_steps = 1
        await record_local_step(group, base_policy_version=0, rollout_id=0, check_model=False)
        return stored

    stored = asyncio.run(run())
    assert stored[1][0] == full.fragments[1] and len(stored[1][1]) == len(full.fragments[1].chunks)
    calls = dict(group.calls)
    assert calls["export_full_parameter_chunked_shard"]["lazy"] is True
    assert calls["materialize_full_parameter_fragments"]["return_descriptors"] is True
    assert calls["record_full_parameter_local_step"]["check_model"] is False
