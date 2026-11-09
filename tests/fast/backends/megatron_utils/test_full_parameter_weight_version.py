"""The published weight version (model companion) under the full-parameter transport.

The companion ``weight_version`` is the one number the engines are told at a
weight update.  A DiLoCo apply writes new weights, so the finalize of a global
cut stamps the version the orchestrator chose on every rank; a local-step
receipt reports the version the step left behind.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.full_parameter_state import (
    commit_prepared_full_parameter_shard,
    export_full_parameter_chunked_shard,
    finalize_full_parameter_commit,
    model_weight_version,
    prepare_full_parameter_chunked_shard,
    record_full_parameter_local_step,
    set_model_weight_version,
)
from miles.backends.training_utils.model_companion import ModelCompanion
from tests.fast.backends.megatron_utils.test_full_parameter_state import (
    _chunked_actor,
    _copy_chunk_refs,
    _FakeRay,
)


def _chunk(version: int) -> torch.nn.Module:
    chunk = torch.nn.Module()
    chunk.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0,)))
    chunk.model_companion.weight_version.fill_(version)
    return chunk


def _with_companions(actor, *versions: int):
    actor.model = [_chunk(version) for version in versions]
    refreshed: list[set[int]] = []
    actor.weights_backuper = SimpleNamespace(
        calls=[],
        backup=lambda role: actor.weights_backuper.calls.append(role),
        backup_tensors=lambda tag, ids: refreshed.append(set(ids)) or len(ids),
    )
    return refreshed


def test_bare_actor_has_no_weight_version(monkeypatch):
    actor, *_ = _chunked_actor(monkeypatch)
    assert model_weight_version(actor) is None
    with pytest.raises(RuntimeError, match="no companion"):
        set_model_weight_version(actor, 1)


def test_set_is_monotone_across_chunks_and_refreshes_the_backup(monkeypatch):
    actor, *_ = _chunked_actor(monkeypatch)
    refreshed = _with_companions(actor, 3, 3)

    assert set_model_weight_version(actor, 7) == 7
    assert model_weight_version(actor) == 7
    assert [int(chunk.model_companion.weight_version) for chunk in actor.model] == [7, 7]
    assert refreshed == [{id(chunk.model_companion.weight_version) for chunk in actor.model}]

    assert set_model_weight_version(actor, 7) == 7  # idempotent, no second refresh
    assert refreshed[1:] == []
    with pytest.raises(RuntimeError, match="below the installed"):
        set_model_weight_version(actor, 6)
    with pytest.raises(ValueError):
        set_model_weight_version(actor, -1)

    actor._enable_weight_backup = False
    assert set_model_weight_version(actor, 8) == 8
    assert refreshed[1:] == []  # backup off: nothing to refresh


def test_diverged_companions_are_refused(monkeypatch):
    actor, *_ = _chunked_actor(monkeypatch)
    _with_companions(actor, 1, 2)
    with pytest.raises(RuntimeError, match="diverged"):
        model_weight_version(actor)


def test_finalize_stamps_the_weight_version_once_per_cut(monkeypatch):
    actor, named, _state, _manifest, _plan = _chunked_actor(monkeypatch)
    _with_companions(actor, 4)
    ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=16, ray_module=ray)
    target = replace(exported, policy_version=1, local_step_generation=0, chunk_refs=_copy_chunk_refs(exported))

    assert prepare_full_parameter_chunked_shard(actor, target, commit_token="c", ray_module=ray) == len(named)
    assert commit_prepared_full_parameter_shard(actor, commit_token="c", ray_module=ray) == len(named)
    assert model_weight_version(actor) == 4  # commit alone leaves the published version
    assert finalize_full_parameter_commit(actor, commit_token="c", weight_version=9) == len(named)
    assert model_weight_version(actor) == 9
    # An idempotent re-finalize with the same version is a no-op; a lower one is refused.
    assert finalize_full_parameter_commit(actor, commit_token="c", weight_version=9) == len(named)
    with pytest.raises(RuntimeError, match="below the installed"):
        finalize_full_parameter_commit(actor, commit_token="c", weight_version=8)
    assert finalize_full_parameter_commit(actor, commit_token="c") == len(named)  # None: untouched
    assert model_weight_version(actor) == 9


def test_local_step_receipt_reports_the_published_version(monkeypatch):
    actor, *_ = _chunked_actor(monkeypatch)
    _with_companions(actor, 5)
    actor._last_rollout_id = 0
    actor.opt_param_scheduler.num_steps = 1
    receipt = record_full_parameter_local_step(actor, base_policy_version=0, rollout_id=0)
    assert receipt.weight_version == 5

    bare, *_ = _chunked_actor(monkeypatch)
    bare._last_rollout_id = 0
    bare.opt_param_scheduler.num_steps = 1
    assert record_full_parameter_local_step(bare, base_policy_version=0, rollout_id=0).weight_version is None
