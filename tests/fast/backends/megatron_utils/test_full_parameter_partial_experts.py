"""Full-parameter transport over a partially trainable model (K3.1 partial experts).

Only the selected experts (``.mlp.experts.trainable.``) train; everything else
is frozen by ``freeze_all_but_trainable_experts`` and never enters the
optimizer.  The transport must see exactly the trainable parameters: the
manifest and the fragment plan name only them, an export carries only their
masters, and an apply leaves every frozen parameter bit-identical.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    commit_prepared_full_parameter_shard,
    export_full_parameter_chunked_shard,
    finalize_full_parameter_commit,
    full_parameter_partition_report,
    full_parameter_rank_state,
    full_parameter_shard_manifest,
    initialize_full_parameter_tracking,
    install_full_parameter_fragment_plan,
    prepare_full_parameter_chunked_shard,
)
from miles.ray.full_parameter_transport import FullParameterChunkedCut, store_full_parameter_fragment_payload
from miles_plugins.models.kimi_k3.partial_experts import (
    freeze_all_but_trainable_experts,
    is_trainable_expert_param_name,
)
from tests.fast.backends.megatron_utils.test_full_parameter_state import _backend, _FakeRay, topology
from tests.fast.miles_plugins.models.kimi_k3.test_partial_experts_freeze_export import _fake_model

PREFIX = "module.module."


class _Optimizer:
    """A non-distributed Megatron-shaped optimizer over the trainable parameters only."""

    def __init__(self, trainable: list[torch.nn.Parameter]) -> None:
        for parameter in trainable:
            parameter.main_param = parameter.detach().float().clone()
        masters = [parameter.main_param for parameter in trainable]
        self._pairs = [(parameter, parameter.main_param) for parameter in trainable]
        self.optimizer = SimpleNamespace(
            param_groups=[{"params": masters}],
            state={master: {"exp_avg": torch.zeros_like(master), "exp_avg_sq": torch.zeros_like(master)} for master in masters},
        )

    def _copy_main_params_to_model_params(self) -> None:
        for parameter, master in self._pairs:
            parameter.data.copy_(master.to(dtype=parameter.dtype))


@pytest.fixture
def actor(monkeypatch):
    model = _fake_model()
    for parameter in model.parameters():
        parameter.data.normal_()
    trainable_count, frozen_count = freeze_all_but_trainable_experts(model)
    assert trainable_count and frozen_count
    trainable = [parameter for _name, parameter in model.named_parameters() if parameter.requires_grad]

    # The transport reads the model through Megatron's global naming; here that is the
    # plain parameter tree with the usual module prefix.
    stub = types.ModuleType("miles.backends.megatron_utils.named_weights")
    stub.named_params_and_buffers = lambda args, chunks, convert_to_global_name=True: (
        (PREFIX + name, parameter) for chunk in chunks for name, parameter in chunk.named_parameters()
    )
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.named_weights", stub)
    monkeypatch.setattr("miles.backends.megatron_utils.full_parameter_state.full_parameter_topology", topology)
    monkeypatch.setattr("miles.backends.megatron_utils.full_parameter_state._tensor_parallel_ranks", lambda: ((0, 1), 1))

    actor = SimpleNamespace(
        role="actor",
        model=[model],
        args=SimpleNamespace(
            full_parameter_lora_hybrid=False,
            rematerialize_param_from_master_weight=False,
            stream_optimizer_state_to_disk=False,
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            start_rollout_id=None,
            backend=_backend(global_batch_size=1),
        ),
        optimizer=_Optimizer(trainable),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        weights_backuper=SimpleNamespace(backup=lambda tag: None, backup_tensors=lambda tag, ids: 0),
        _enable_weight_backup=True,
        _last_rollout_id=0,
    )
    initialize_full_parameter_tracking(actor, 0)
    return actor


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def test_manifest_and_partition_report_name_only_the_trainable_experts(actor):
    model = actor.model[0]
    expected = {PREFIX + name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert expected and all(is_trainable_expert_param_name(name) for name in expected)

    manifest = full_parameter_shard_manifest(actor, role="actor")
    assert {spec.name for spec in manifest.specs} == expected
    assert not any(".experts.frozen." in spec.name or "router" in spec.name for spec in manifest.specs)
    assert sum(spec.numel for spec in manifest.specs) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    report = full_parameter_partition_report(actor, role="actor")
    assert {partition.name for partition in report.parameters} == expected


def test_apply_writes_the_trainable_experts_and_leaves_frozen_parameters_bit_identical(actor):
    model = actor.model[0]
    manifest = full_parameter_shard_manifest(actor, role="actor")
    names = tuple(spec.wire_name for spec in manifest.specs)
    plan = FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="9" * 64,
        fragments=(FullParameterFragmentPlan(fragment_id=0, wire_names=names, numel=sum(spec.numel for spec in manifest.specs)),),
    )
    assert install_full_parameter_fragment_plan(actor, plan, role="actor") == 1
    assert {name for fragment in full_parameter_rank_state(actor).fragment_plan.fragments for name in fragment.wire_names} == set(names)

    ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=64, ray_module=ray, role="actor")
    exported_numel = sum(ray.get(reference).numel() for references in exported.chunk_refs for reference in references)
    assert exported_numel == sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # A merged policy: every exported value plus one, stored as a fresh authoritative fragment.
    cut = FullParameterChunkedCut(
        policy_version=0, local_step_generation=0, parameter_layout_hash=exported.parameter_layout_hash, shards=(exported,)
    )
    merged = torch.cat([ray.get(reference) for reference in exported.chunk_refs[0]]) + 1.0
    descriptor, refs = store_full_parameter_fragment_payload(cut, 0, merged.numpy().tobytes(), ray_module=ray)
    target = replace(exported, policy_version=1, local_step_generation=0, fragments=(descriptor,), chunk_refs=[refs])
    before = _snapshot(model)
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}

    count = len(names)
    assert prepare_full_parameter_chunked_shard(actor, target, commit_token="c1", ray_module=ray, role="actor") == count
    assert commit_prepared_full_parameter_shard(actor, commit_token="c1", ray_module=ray, role="actor") == count
    assert finalize_full_parameter_commit(actor, commit_token="c1") == count

    after = _snapshot(model)
    for name in frozen:
        assert torch.equal(after[name], before[name]), f"frozen {name} changed"
        assert not model.get_parameter(name).requires_grad
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(after[name], (before[name].float() + 1.0).to(after[name].dtype)), name
            assert torch.equal(parameter.main_param, before[name].float() + 1.0)
    assert full_parameter_rank_state(actor).policy_version == 1
