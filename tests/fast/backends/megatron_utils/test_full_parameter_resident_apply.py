"""A chunked apply reads and copies only the fragments the masters do not hold.

``skip_resident=True`` must leave the masters, the model and the commit count
bit-identical to the full apply, while reading (``ray.get``) only the
replaced fragments' chunks.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_state import (
    commit_prepared_full_parameter_shard,
    export_full_parameter_chunked_shard,
    finalize_full_parameter_commit,
    forget_full_parameter_resident_fragments,
    install_full_parameter_fragment_plan,
    prepare_full_parameter_chunked_shard,
)
from miles.ray.full_parameter_transport import FullParameterChunkedCut, store_full_parameter_fragment_payload
from tests.fast.backends.megatron_utils.test_full_parameter_state import _chunked_actor, _split_plan


class _CountingRay:
    def __init__(self):
        self.objects = {}
        self.gets = 0

    def put(self, value):
        reference = object()
        self.objects[reference] = value.clone()
        return reference

    def get(self, reference):
        self.gets += 1
        return self.objects[reference]


def _split_actor(monkeypatch):
    actor, named, _state, manifest, _plan = _chunked_actor(monkeypatch)
    delattr(actor, "_external_full_parameter_fragment_plan")
    plan = _split_plan(manifest)
    assert install_full_parameter_fragment_plan(actor, plan) == 3
    return actor, named


def _master_bits(named):
    return [parameter.main_param.detach().clone().view(torch.int32) for _name, parameter in named]


def _model_bits(named):
    return [parameter.detach().clone().view(torch.int16) for _name, parameter in named]


def _target(exported, ray, policy_version, replaced):
    """``exported`` promoted to ``policy_version`` with ``replaced`` {fragment: values}."""

    cut = FullParameterChunkedCut(policy_version=exported.policy_version,
                                  local_step_generation=exported.local_step_generation,
                                  parameter_layout_hash=exported.parameter_layout_hash, shards=(exported,))
    fragments, refs = list(exported.fragments), [list(row) for row in exported.chunk_refs]
    for index, descriptor in enumerate(exported.fragments):
        values = replaced.get(descriptor.fragment_id)
        if values is not None:
            payload = torch.tensor(values, dtype=torch.float32).numpy().tobytes()
            fragments[index], refs[index] = store_full_parameter_fragment_payload(
                cut, descriptor.fragment_id, payload, ray_module=ray)
    return replace(exported, policy_version=policy_version, local_step_generation=0,
                   fragments=tuple(fragments), chunk_refs=refs)


def _apply(actor, target, ray, token, *, skip):
    before = ray.gets
    count = prepare_full_parameter_chunked_shard(actor, target, commit_token=token, ray_module=ray,
                                                 skip_resident=skip)
    assert commit_prepared_full_parameter_shard(actor, commit_token=token, ray_module=ray) == count
    assert finalize_full_parameter_commit(actor, commit_token=token) == count
    return count, ray.gets - before


def _two_applies(monkeypatch, *, skip):
    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    first = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    counts = [_apply(actor, first, ray, "apply-1", skip=skip)]
    stats = [dict(actor._external_full_parameter_apply_stats)]
    # The next apply starts from the committed target, as the role stream does.
    second = _target(first, ray, 2, {2: [-0.0, 3.75, 2.5e38]})
    counts.append(_apply(actor, second, ray, "apply-2", skip=skip))
    stats.append(dict(actor._external_full_parameter_apply_stats))
    return actor, named, counts, stats


def test_resident_skip_is_bit_identical_and_reads_only_replaced_fragments(monkeypatch):
    _actor, full_named, full_counts, _ = _two_applies(monkeypatch, skip=False)
    full_masters, full_model = _master_bits(full_named), _model_bits(full_named)
    actor, named, counts, stats = _two_applies(monkeypatch, skip=True)

    for got, want in zip(_master_bits(named), full_masters, strict=True):
        assert torch.equal(got, want)
    for got, want in zip(_model_bits(named), full_model, strict=True):
        assert torch.equal(got, want)
    assert [count for count, _ in counts] == [count for count, _ in full_counts] == [len(named)] * 2
    # Fragments of 2, 3 and 3 scalars in 8-byte chunks: 1 + 2 + 2 chunks.
    # Full apply reads every chunk twice (prepare verifies, commit copies).
    assert [gets for _, gets in full_counts] == [10, 10]
    # Skip: only the replaced fragment's 2 chunks, twice.
    assert [gets for _, gets in counts] == [4, 4]
    for one, replaced in zip(stats, (1, 2), strict=True):
        assert one["prepare"]["resident_fragments"] == 2 and one["prepare"]["read_fragments"] == 1
        assert one["commit"]["copied_fragments"] == 1 and one["commit"]["resident_fragments"] == 2
    assert actor.weights_backuper.calls == ["actor", "actor"]


def test_skip_needs_the_live_record(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {0: [1.0, 2.0]})
    want = _master_bits(named)
    want[1] = torch.tensor([1.0, 2.0]).view(torch.int32)

    # Something wrote the masters outside the export/commit path and said so:
    # every fragment is read and copied again, restoring the target bytes.
    named[1][1].main_param.fill_(99.0)  # final_norm: fragment 0
    forget_full_parameter_resident_fragments(actor)
    _count, gets = _apply(actor, target, ray, "after-forget", skip=True)
    assert gets == 10
    for got, expected in zip(_master_bits(named), want, strict=True):
        assert torch.equal(got, expected)


def test_new_local_generation_disables_the_skip(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {0: [1.0, 2.0]})
    record = actor._external_full_parameter_resident
    assert record is not None and record.local_step_generation == 0
    # A record from another generation does not describe the masters.
    actor._external_full_parameter_resident = replace(record, local_step_generation=5)
    _count, gets = _apply(actor, target, ray, "other-generation", skip=True)
    assert gets == 10


def test_commit_refuses_a_record_replaced_after_prepare(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {0: [1.0, 2.0]})
    prepare_full_parameter_chunked_shard(actor, target, commit_token="t", ray_module=ray, skip_resident=True)
    forget_full_parameter_resident_fragments(actor)
    with pytest.raises(RuntimeError, match="resident fragments changed"):
        commit_prepared_full_parameter_shard(actor, commit_token="t", ray_module=ray)


def test_retried_commit_after_a_partial_copy_keeps_the_skip_exact(monkeypatch):
    _actor, full_named, _counts, _ = _two_applies(monkeypatch, skip=False)
    want = _master_bits(full_named)

    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    first = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    prepare_full_parameter_chunked_shard(actor, first, commit_token="r", ray_module=ray, skip_resident=True)
    original = fps._propagate_master_writes
    calls = 0

    def fail_once(target_actor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure after the copy")
        return original(target_actor)

    monkeypatch.setattr(fps, "_propagate_master_writes", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        commit_prepared_full_parameter_shard(actor, commit_token="r", ray_module=ray)
    assert actor._external_full_parameter_resident is None
    commit_prepared_full_parameter_shard(actor, commit_token="r", ray_module=ray)
    finalize_full_parameter_commit(actor, commit_token="r")
    second = _target(first, ray, 2, {2: [-0.0, 3.75, 2.5e38]})
    _apply(actor, second, ray, "r2", skip=True)
    for got, expected in zip(_master_bits(named), want, strict=True):
        assert torch.equal(got, expected)


def test_parallel_export_hashes_match_the_serial_definition(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    for descriptor, refs in zip(exported.fragments, exported.chunk_refs, strict=True):
        digest = fps._fragment_hasher(descriptor.fragment_id, exported.parameter_layout_hash, exported.plan_hash)
        for chunk, ref in zip(descriptor.chunks, refs, strict=True):
            value = ray.objects[ref]
            assert chunk.payload_hash == fps._chunk_hash(
                descriptor.fragment_id, chunk.chunk_index, chunk.flat_offset, value,
                parameter_layout_hash=exported.parameter_layout_hash, plan_hash=exported.plan_hash)
            digest.update(fps._tensor_bytes(value))
        assert descriptor.payload_hash == digest.hexdigest()


def test_export_still_refuses_a_non_finite_master(monkeypatch):
    actor, named = _split_actor(monkeypatch)
    named[0][1].main_param[0, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=_CountingRay())
