"""A deferred export, on-demand materialization and one apply of many fragments.

``defer_payload=True`` must give the same descriptors (chunk hashes, fragment
digests, NaN refusal) as the full export while storing no bytes in Ray.  A
fragment it later materializes must hold exactly the bytes the full export
stores.  An apply of a target built from the deferred cut must leave the
masters and the model bit-identical to the full path, and one apply of k
fragments must equal k applies of one fragment each.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH, full_parameter_rank_call
from miles.backends.megatron_utils.full_parameter_state import (
    DEFERRED_FULL_PARAMETER_CHUNK,
    export_full_parameter_chunked_shard,
    forget_full_parameter_resident_fragments,
    full_parameter_optimizer_state,
    is_deferred_full_parameter_chunk,
    materialize_full_parameter_fragments,
    prepare_full_parameter_chunked_shard,
)
from miles.ray.full_parameter_transport import (
    FULL_PARAMETER_TRANSPORT_FEATURES,
    FullParameterChunkedCut,
    export_chunked_cut,
    iter_full_parameter_fragment_parts,
    materialize_chunked_fragments,
    optimizer_states,
)
from tests.fast.backends.megatron_utils.test_full_parameter_resident_apply import (
    _apply,
    _CountingRay,
    _master_bits,
    _model_bits,
    _split_actor,
    _target,
)


def _export(actor, ray, *, defer, policy_version=0, generation=0):
    return export_full_parameter_chunked_shard(actor, policy_version=policy_version,
                                               local_step_generation=generation, max_chunk_bytes=8,
                                               ray_module=ray, defer_payload=defer)


def _values(state, ray):
    """Each fragment's stored float bits, in chunk order."""
    return {descriptor.fragment_id: [ray.objects[ref].view(torch.int32) for ref in refs]
            for descriptor, refs in zip(state.fragments, state.chunk_refs, strict=True)}


@pytest.mark.parametrize("block, threads", [(1, 1), (2, 4), (1 << 21, 8)])
def test_deferred_export_has_the_full_descriptors_and_stores_nothing(monkeypatch, block, threads):
    monkeypatch.setattr(fps, "_SCAN_BLOCK_NUMEL", block)
    monkeypatch.setenv("MILES_FULL_PARAMETER_HASH_THREADS", str(threads))
    actor, _named = _split_actor(monkeypatch)
    full = _export(actor, _CountingRay(), defer=False)
    full_record = actor._external_full_parameter_resident

    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)

    assert deferred.fragments == full.fragments  # every chunk hash and fragment digest
    assert ray.objects == {}
    assert all(is_deferred_full_parameter_chunk(ref) for refs in deferred.chunk_refs for ref in refs)
    assert [len(refs) for refs in deferred.chunk_refs] == [len(refs) for refs in full.chunk_refs]
    assert actor._external_full_parameter_resident.payload_hashes == full_record.payload_hashes
    assert actor._external_full_parameter_apply_stats["export"]["deferred"] is True


def test_deferred_export_still_refuses_a_non_finite_master(monkeypatch):
    monkeypatch.setattr(fps, "_SCAN_BLOCK_NUMEL", 1)
    actor, named = _split_actor(monkeypatch)
    named[0][1].main_param[0, 1] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        _export(actor, _CountingRay(), defer=True)


def test_the_deferred_marker_survives_pickling_as_itself():
    import pickle

    assert pickle.loads(pickle.dumps(DEFERRED_FULL_PARAMETER_CHUNK)) is DEFERRED_FULL_PARAMETER_CHUNK


def test_materialize_stores_exactly_the_bytes_of_the_full_export(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    full_ray = _CountingRay()
    full = _export(actor, full_ray, defer=False)
    want = _values(full, full_ray)

    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)
    record = actor._external_full_parameter_resident
    stored = materialize_full_parameter_fragments(actor, state=deferred, fragment_ids=[2, 0], ray_module=ray)

    assert [fragment_id for fragment_id, _refs in stored] == [0, 2]
    for fragment_id, refs in stored:
        got = [ray.objects[ref].view(torch.int32) for ref in refs]
        assert len(got) == len(want[fragment_id])
        assert all(torch.equal(a, b) for a, b in zip(got, want[fragment_id], strict=True))
    # Nothing about the masters' bookkeeping moved.
    assert actor._external_full_parameter_resident is record


def test_materialize_refuses_after_the_masters_changed(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)
    # A commit start, a restore or a local step drops the record.
    forget_full_parameter_resident_fragments(actor)
    with pytest.raises(RuntimeError, match="no longer held"):
        materialize_full_parameter_fragments(actor, state=deferred, fragment_ids=[1], ray_module=ray)

    deferred = _export(actor, ray, defer=True)
    named[0][1].main_param.view(-1)[4] += 1.0  # fragment 2, written without telling the record
    with pytest.raises(RuntimeError, match="bytes changed since its deferred export"):
        materialize_full_parameter_fragments(actor, state=deferred, fragment_ids=[1, 2], ray_module=ray)


def _deferred_rows_target(deferred, ray, policy_version, replaced):
    """The role stream's mixed target: deferred rows, ``replaced`` fragments stored."""
    return _target(deferred, ray, policy_version, replaced)


def test_apply_of_a_target_built_on_a_deferred_cut_is_bit_identical(monkeypatch):
    replaced = {1: [30.5, -1.25, 1e-30]}
    actor, full_named = _split_actor(monkeypatch)
    full_ray = _CountingRay()
    full = _export(actor, full_ray, defer=False)
    _apply(actor, _target(full, full_ray, 1, replaced), full_ray, "full", skip=True)
    want_masters, want_model = _master_bits(full_named), _model_bits(full_named)

    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)
    count, gets = _apply(actor, _deferred_rows_target(deferred, ray, 1, replaced), ray, "deferred", skip=True)

    assert count == len(named)
    assert gets == 4  # only the replaced fragment's 2 chunks, verified then copied
    assert all(torch.equal(a, b) for a, b in zip(_master_bits(named), want_masters, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(_model_bits(named), want_model, strict=True))
    # The commit records the deferred rows as resident: the next apply skips them too.
    assert set(actor._external_full_parameter_resident.payload_hashes) == {0, 1, 2}


def test_a_deferred_fragment_is_resident_even_with_the_skip_switched_off(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)
    target = _deferred_rows_target(deferred, ray, 1, {0: [1.0, 2.0]})
    monkeypatch.setenv("MILES_FULL_PARAMETER_SKIP_RESIDENT", "0")
    _count, gets = _apply(actor, target, ray, "switch-off", skip=True)
    assert gets == 2  # fragment 0 (one chunk) only; the deferred rows have no bytes to read


def test_prepare_refuses_a_deferred_fragment_the_masters_no_longer_hold(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    deferred = _export(actor, ray, defer=True)
    target = _deferred_rows_target(deferred, ray, 1, {0: [1.0, 2.0]})
    forget_full_parameter_resident_fragments(actor)
    with pytest.raises(RuntimeError, match=r"deferred or lazy full-parameter fragments \[1, 2\] are not held"):
        prepare_full_parameter_chunked_shard(actor, target, commit_token="t", ray_module=ray, skip_resident=True)


def test_one_apply_of_two_fragments_equals_two_applies_of_one(monkeypatch):
    first, second = {1: [30.5, -1.25, 1e-30]}, {2: [-0.0, 3.75, 2.5e38]}
    actor, seq_named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=True)
    one = _target(exported, ray, 1, first)
    _apply(actor, one, ray, "seq-1", skip=True)
    _apply(actor, _target(one, ray, 2, second), ray, "seq-2", skip=True)
    want_masters, want_model = _master_bits(seq_named), _model_bits(seq_named)

    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = _export(actor, ray, defer=True)
    count, gets = _apply(actor, _target(exported, ray, 1, {**first, **second}), ray, "batch", skip=True)

    assert count == len(named)
    assert gets == 8  # both replaced fragments (2 chunks each), verified then copied
    assert all(torch.equal(a, b) for a, b in zip(_master_bits(named), want_masters, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(_model_bits(named), want_model, strict=True))
    assert actor.weights_backuper.calls == ["actor"]


def test_optimizer_state_without_the_model_check_reports_the_same_proof(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    actor._last_rollout_id = 0
    checked = full_parameter_optimizer_state(actor)
    assert full_parameter_optimizer_state(actor, check_model=False) == checked
    named[0][1].data.view(-1)[0] += 1.0  # the model drifted from its master
    with pytest.raises(RuntimeError, match="differs from its optimizer master"):
        full_parameter_optimizer_state(actor)
    assert full_parameter_optimizer_state(actor, check_model=False) == checked


class _PluginGroup:
    """One real rank behind the real plugin entry point."""

    def __init__(self, actor):
        self.actor = actor
        self.calls = []

    async def run_plugin(self, fn_path, kwargs):
        assert fn_path == PLUGIN_PATH
        self.calls.append((kwargs["operation"], dict(kwargs["kwargs"])))
        return [full_parameter_rank_call(self.actor, **kwargs)]


def test_the_transport_defers_and_materializes_through_the_plugin(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    actor.role = "actor"
    ray = _CountingRay()
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: ray)
    group = _PluginGroup(actor)
    assert {"skip_resident", "deferred_export", "optimizer_state_check_model"} <= FULL_PARAMETER_TRANSPORT_FEATURES

    async def run():
        cut = await export_chunked_cut(group, 0, 0, max_chunk_bytes=8, defer_payload=True)
        assert ray.objects == {}
        with pytest.raises(RuntimeError, match="deferred"):
            next(iter_full_parameter_fragment_parts(cut, 1, ray_module=ray))
        stored = await materialize_chunked_fragments(group, cut, [1])
        assert list(stored) == [1]
        (state,) = cut.shards
        state.chunk_refs[1][:] = stored[1]
        parts = b"".join(bytes(part) for part in iter_full_parameter_fragment_parts(cut, 1, ray_module=ray))
        assert len(parts) == 3 * 4
        actor._last_rollout_id = 0
        assert await optimizer_states(group, check_model=False) == await optimizer_states(group)
        return cut

    cut = asyncio.run(run())
    assert isinstance(cut, FullParameterChunkedCut)
    operations = [operation for operation, _kwargs in group.calls]
    assert operations.count("materialize_full_parameter_fragments") == 1
    export_kwargs = next(kwargs for operation, kwargs in group.calls if operation == "export_full_parameter_chunked_shard")
    assert export_kwargs["defer_payload"] is True
    assert [kwargs.get("check_model") for operation, kwargs in group.calls
            if operation == "full_parameter_optimizer_state"] == [False, None]


def test_a_plain_export_keeps_its_kwargs(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    actor.role = "actor"
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: _CountingRay())
    group = _PluginGroup(actor)
    asyncio.run(export_chunked_cut(group, 0, 0, max_chunk_bytes=8))
    (operation, kwargs), = [call for call in group.calls if call[0] == "export_full_parameter_chunked_shard"]
    assert "defer_payload" not in kwargs
    assert replace  # keep the import for helpers that build variants
