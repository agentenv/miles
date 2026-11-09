from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FULL_PARAMETER_MAX_CHUNK_BYTES,
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    FullParameterShardState,
    FullParameterTopology,
    abort_prepared_full_parameter_shard,
    apply_full_parameter_shard,
    apply_full_parameter_shard_values,
    commit_prepared_full_parameter_shard,
    export_full_parameter_chunked_shard,
    export_full_parameter_shard,
    finalize_full_parameter_commit,
    full_parameter_initial_policy_version,
    full_parameter_optimizer_state,
    full_parameter_shard_manifest,
    initialize_full_parameter_tracking,
    install_full_parameter_fragment_plan,
    make_full_parameter_shard_state,
    prepare_full_parameter_chunked_shard,
    rebase_full_parameter_scheduler_anchor,
    record_full_parameter_local_step,
    validate_full_parameter_shard_values,
)


def topology(*, tp_rank=0, tp_size=2, dp_rank=0, dp_size=1):
    return FullParameterTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        cp_rank=0,
        cp_size=1,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )


def parameters():
    first = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    first.main_param = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    second.main_param = torch.tensor([7.0, 8.0], dtype=torch.float32)
    return (
        ("module.decoder.layers.0.proj.weight", first),
        ("module.decoder.final_norm.weight", second),
    )


def test_export_is_fp32_canonical_and_topology_qualified():
    state = make_full_parameter_shard_state(
        policy_version=3,
        role="actor",
        topology=topology(),
        named_parameters=parameters(),
    )

    assert state.policy_version == 3
    assert all(value.device.type == "cpu" for value in state.tensors.values())
    assert all(value.dtype == torch.float32 for value in state.tensors.values())
    assert all("actor::tp0-of-2." in name for name in state.tensors)
    assert all(spec.shard_id == topology().shard_id for spec in state.specs)


def test_apply_changes_masters_but_preserves_optimizer_state_objects():
    named = parameters()
    state = make_full_parameter_shard_state(
        policy_version=3,
        role="actor",
        topology=topology(),
        named_parameters=named,
    )
    changed = {name: value + 0.5 for name, value in state.tensors.items()}
    incoming = FullParameterShardState(
        4,
        state.topology,
        state.layout_hash,
        state.specs,
        changed,
    )
    optimizer_state = {parameter.main_param: {"exp_avg": torch.ones_like(parameter.main_param)} for _name, parameter in named}
    before_ids = {id(parameter): id(value["exp_avg"]) for parameter, value in optimizer_state.items()}
    before_masters = {name: parameter.main_param.clone() for name, parameter in named}

    assert (
        validate_full_parameter_shard_values(
            incoming,
            role="actor",
            topology=topology(),
            named_parameters=named,
        )
        == 2
    )
    assert all(torch.equal(parameter.main_param, before_masters[name]) for name, parameter in named)

    applied = apply_full_parameter_shard_values(
        incoming,
        role="actor",
        topology=topology(),
        named_parameters=named,
    )

    assert applied == 2
    parameters_by_name = {name: parameter for name, parameter in named}
    for spec in state.specs:
        assert torch.equal(
            parameters_by_name[spec.name].main_param,
            changed[spec.wire_name],
        )
    assert {id(parameter): id(value["exp_avg"]) for parameter, value in optimizer_state.items()} == before_ids


def test_apply_rejects_topology_layout_and_value_drift_before_copy():
    named = parameters()
    state = make_full_parameter_shard_state(
        policy_version=3,
        role="actor",
        topology=topology(),
        named_parameters=named,
    )
    before = [parameter.main_param.clone() for _name, parameter in named]

    with pytest.raises(RuntimeError, match="topology"):
        apply_full_parameter_shard_values(
            state,
            role="actor",
            topology=topology(tp_rank=1),
            named_parameters=named,
        )
    malformed = replace(state, tensors={**state.tensors, "extra": torch.ones(1)})
    with pytest.raises(ValueError, match="incomplete"):
        apply_full_parameter_shard_values(
            malformed,
            role="actor",
            topology=topology(),
            named_parameters=named,
        )
    assert all(torch.equal(parameter.main_param, value) for (_name, parameter), value in zip(named, before, strict=True))


def test_initial_profile_rejects_dp_sharded_or_missing_optimizer_masters():
    with pytest.raises(RuntimeError, match="DP=1"):
        make_full_parameter_shard_state(
            policy_version=0,
            role="actor",
            topology=topology(dp_size=2),
            named_parameters=parameters(),
        )
    named = parameters()
    del named[0][1].main_param
    with pytest.raises(RuntimeError, match="no FP32 optimizer master"):
        make_full_parameter_shard_state(
            policy_version=0,
            role="actor",
            topology=topology(),
            named_parameters=named,
        )


def test_native_fp32_parameter_must_be_owned_by_optimizer():
    native = torch.nn.Parameter(torch.tensor([0.25, 0.5], dtype=torch.float32))
    named = (("module.decoder.layers.0.A_log", native),)

    with pytest.raises(RuntimeError, match="no FP32 optimizer master"):
        make_full_parameter_shard_state(
            policy_version=0,
            role="actor",
            topology=topology(),
            named_parameters=named,
        )

    state = make_full_parameter_shard_state(
        policy_version=0,
        role="actor",
        topology=topology(),
        named_parameters=named,
        optimizer_owned_parameter_ids=frozenset({id(native)}),
    )
    incoming = replace(
        state,
        policy_version=1,
        tensors={name: value + 0.125 for name, value in state.tensors.items()},
    )
    assert (
        apply_full_parameter_shard_values(
            incoming,
            role="actor",
            topology=topology(),
            named_parameters=named,
            optimizer_owned_parameter_ids=frozenset({id(native)}),
        )
        == 1
    )
    assert torch.equal(native, torch.tensor([0.375, 0.625], dtype=torch.float32))


def test_optimizer_state_proof_binds_progress_and_survives_parameter_apply(
    monkeypatch,
):
    named = parameters()
    for _name, parameter in named:
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    masters = [parameter.main_param for _name, parameter in named]
    states = {
        master: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.full_like(master, 0.25),
            "exp_avg_sq": torch.full_like(master, 0.5),
        }
        for master in masters
    }
    base = SimpleNamespace(
        param_groups=[{"params": masters}],
        state=states,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            global_batch_size=1,
        ),
        optimizer=SimpleNamespace(optimizer=base),
        opt_param_scheduler=SimpleNamespace(num_steps=1),
        _external_full_policy_version=0,
        _external_full_local_step_generation=1,
        _last_rollout_id=0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state.full_parameter_topology",
        topology,
    )

    before = full_parameter_optimizer_state(actor)
    for _name, parameter in named:
        parameter.main_param.add_(0.5)
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    actor._external_full_policy_version = 1
    after_apply = full_parameter_optimizer_state(actor)

    assert before.selected_state_sha256 == after_apply.selected_state_sha256
    assert before.populated_parameter_count == after_apply.populated_parameter_count == 2
    assert before.optimizer_state_tensor_count == after_apply.optimizer_state_tensor_count == 6
    assert before.optimizer_state_scalar_count == after_apply.optimizer_state_scalar_count
    assert before.installed_policy_version == 0
    assert before.local_step_generation == 1
    assert after_apply.installed_policy_version == 1

    selected_master = min(masters, key=torch.Tensor.numel)
    states[selected_master]["exp_avg"].add_(1.0)
    actor._last_rollout_id = 1
    actor.opt_param_scheduler.num_steps = 2
    after_next_step = full_parameter_optimizer_state(actor)
    assert after_next_step.selected_state_sha256 != after_apply.selected_state_sha256
    assert after_next_step.last_rollout_id == 1
    assert after_next_step.scheduler_num_steps == 2


def test_local_step_generation_keeps_global_base_until_complete_apply(monkeypatch):
    named = parameters()
    for _name, parameter in named:
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    masters = [parameter.main_param for _name, parameter in named]
    base = SimpleNamespace(param_groups=[{"params": masters}], state={})
    backups = []
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            global_batch_size=1,
        ),
        optimizer=SimpleNamespace(optimizer=base),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state.full_parameter_topology",
        topology,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._copy_masters_to_model",
        lambda _actor: None,
    )

    initialize_full_parameter_tracking(actor, 0)
    anchor = export_full_parameter_shard(
        actor,
        policy_version=0,
        local_step_generation=0,
    )
    actor._last_rollout_id = 0
    actor.opt_param_scheduler.num_steps = 1
    receipt = record_full_parameter_local_step(
        actor,
        base_policy_version=0,
        rollout_id=0,
    )
    assert (
        record_full_parameter_local_step(
            actor,
            base_policy_version=0,
            rollout_id=0,
        )
        == receipt
    )

    assert receipt.base_policy_version == 0
    assert receipt.local_step_generation == 1
    assert actor._external_full_policy_version == 0
    with pytest.raises(RuntimeError, match="generation is stale"):
        export_full_parameter_shard(
            actor,
            policy_version=0,
            local_step_generation=0,
        )
    local = export_full_parameter_shard(
        actor,
        policy_version=0,
        local_step_generation=1,
    )
    assert local.policy_version == 0
    assert local.local_step_generation == 1

    global_target = replace(
        local,
        policy_version=1,
        local_step_generation=0,
    )
    assert apply_full_parameter_shard(actor, global_target) == 2
    assert actor._external_full_policy_version == 1
    assert actor._external_full_local_step_generation == 0
    assert actor._external_full_exported_local_step_generation == 0
    assert actor._external_full_scheduler_anchor_steps == 1
    assert backups == ["actor"]
    assert anchor.policy_version == 0

    actor._last_rollout_id = 1
    actor.opt_param_scheduler.num_steps = 2
    record_full_parameter_local_step(
        actor,
        base_policy_version=1,
        rollout_id=1,
    )
    with pytest.raises(RuntimeError, match="unexported"):
        apply_full_parameter_shard(
            actor,
            replace(global_target, policy_version=2),
        )


@pytest.mark.parametrize(
    ("role", "expected_optimizer_steps"),
    (("actor", 2), ("critic", 6)),
)
def test_local_step_receipt_uses_role_aware_optimizer_progress(
    monkeypatch,
    role,
    expected_optimizer_steps,
):
    named = parameters()
    for _name, parameter in named:
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    masters = [parameter.main_param for _name, parameter in named]
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=2,
            num_critic_epochs=3,
            global_batch_size=4,
        ),
        optimizer=SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=[{"params": masters}],
                state={},
            )
        ),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        _last_rollout_id=0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state.full_parameter_topology",
        topology,
    )

    initialize_full_parameter_tracking(actor, 0)
    actor.opt_param_scheduler.num_steps = expected_optimizer_steps * 4
    receipt = record_full_parameter_local_step(
        actor,
        base_policy_version=0,
        rollout_id=0,
        role=role,
    )

    assert receipt.role == role
    assert receipt.optimizer_steps == expected_optimizer_steps
    assert receipt.scheduler_start_steps == 0
    assert receipt.scheduler_end_steps == expected_optimizer_steps * 4

    exported = export_full_parameter_shard(
        actor,
        policy_version=0,
        local_step_generation=1,
        role=role,
    )
    assert {spec.role for spec in exported.specs} == {role}


def test_local_step_receipt_rejects_normal_without_scheduler_progress(monkeypatch):
    named = parameters()
    masters = [parameter.main_param for _name, parameter in named]
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            global_batch_size=1,
        ),
        optimizer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": masters}], state={})),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        _last_rollout_id=0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    initialize_full_parameter_tracking(actor, 0)

    with pytest.raises(RuntimeError, match="exact optimizer progress"):
        record_full_parameter_local_step(
            actor,
            base_policy_version=0,
            rollout_id=0,
        )
    assert actor._external_full_local_step_generation == 0


def test_export_rejects_optimizer_progress_without_a_receipt(monkeypatch):
    actor, _named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    actor.args.num_steps_per_rollout = 1
    actor.args.global_batch_size = 1
    actor.opt_param_scheduler.num_steps = 1

    with pytest.raises(RuntimeError, match="has no local-step receipt"):
        export_full_parameter_chunked_shard(
            actor,
            policy_version=0,
            local_step_generation=0,
            max_chunk_bytes=16,
            ray_module=_FakeRay(),
        )


def test_tracking_initialization_is_atomic_when_scheduler_state_is_missing():
    actor = SimpleNamespace(opt_param_scheduler=SimpleNamespace(num_steps=None))

    with pytest.raises(RuntimeError, match="scheduler progress"):
        initialize_full_parameter_tracking(actor, 0)

    assert not hasattr(actor, "_external_full_policy_version")


def test_initial_policy_version_follows_effective_train_loop_start():
    assert (
        full_parameter_initial_policy_version(
            SimpleNamespace(start_rollout_id=0),
            0,
        )
        == 0
    )
    assert (
        full_parameter_initial_policy_version(
            SimpleNamespace(start_rollout_id=None),
            4,
        )
        == 5
    )
    assert (
        full_parameter_initial_policy_version(
            SimpleNamespace(start_rollout_id=None),
            -1,
        )
        == 0
    )
    with pytest.raises(RuntimeError, match="configured rollout start"):
        full_parameter_initial_policy_version(
            SimpleNamespace(start_rollout_id=True),
            0,
        )


def test_local_step_receipt_does_not_advance_when_topology_proof_fails(
    monkeypatch,
):
    named = parameters()
    for _name, parameter in named:
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    masters = [parameter.main_param for _name, parameter in named]
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            global_batch_size=1,
        ),
        optimizer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": masters}], state={})),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        _last_rollout_id=0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    initialize_full_parameter_tracking(actor, 0)
    actor.opt_param_scheduler.num_steps = 1
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state.full_parameter_topology",
        lambda: (_ for _ in ()).throw(RuntimeError("topology unavailable")),
    )

    with pytest.raises(RuntimeError, match="topology unavailable"):
        record_full_parameter_local_step(
            actor,
            base_policy_version=0,
            rollout_id=0,
        )

    assert actor._external_full_local_step_generation == 0
    assert actor._external_full_last_local_rollout_id is None


class _FakeRay:
    def __init__(self):
        self.objects = {}

    def put(self, value):
        reference = object()
        self.objects[reference] = value.clone()
        return reference

    def get(self, reference):
        return self.objects[reference]


def _chunked_actor(monkeypatch):
    named = parameters()
    for _name, parameter in named:
        parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))
    masters = [parameter.main_param for _name, parameter in named]
    optimizer_state = {
        master: {
            "exp_avg": torch.ones_like(master),
            "exp_avg_sq": torch.full_like(master, 2.0),
        }
        for master in masters
    }
    actor = SimpleNamespace(
        args=SimpleNamespace(
            lora_rank=0,
            lora_adapter_path=None,
            num_steps_per_rollout=1,
            global_batch_size=1,
        ),
        optimizer=SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=[{"params": masters}],
                state=optimizer_state,
            )
        ),
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        weights_backuper=SimpleNamespace(calls=[], backup=lambda role: actor.weights_backuper.calls.append(role)),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state.full_parameter_topology",
        topology,
    )

    def copy_masters_to_model(_actor):
        for _name, parameter in named:
            parameter.data.copy_(parameter.main_param.to(dtype=parameter.dtype))

    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._copy_masters_to_model",
        copy_masters_to_model,
    )
    initialize_full_parameter_tracking(actor, 0)
    manifest = full_parameter_shard_manifest(actor)
    names = tuple(spec.wire_name for spec in manifest.specs)
    plan = FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="9" * 64,
        fragments=(
            FullParameterFragmentPlan(
                fragment_id=0,
                wire_names=names,
                numel=sum(spec.numel for spec in manifest.specs),
            ),
        ),
    )
    assert install_full_parameter_fragment_plan(actor, plan) == 1
    return actor, named, optimizer_state, manifest, plan


def _copy_chunk_refs(state):
    return [list(references) for references in state.chunk_refs]


def test_chunked_export_splits_tensors_and_commit_preserves_optimizer(monkeypatch):
    actor, named, optimizer_state, manifest, plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    expected = torch.cat([parameter.main_param.detach().reshape(-1).clone() for spec in manifest.specs for name, parameter in named if name == spec.name])
    before_optimizer_ids = {id(parameter): tuple(id(value) for value in state.values()) for parameter, state in optimizer_state.items()}

    exported = export_full_parameter_chunked_shard(
        actor,
        policy_version=0,
        max_chunk_bytes=16,
        ray_module=fake_ray,
    )

    assert exported.plan_hash == plan.plan_hash
    assert len(exported.fragments) == 1
    assert [chunk.numel for chunk in exported.fragments[0].chunks] == [4, 4]
    assert len(exported.chunk_refs) == 1
    assert torch.equal(
        torch.cat([fake_ray.get(reference) for reference in exported.chunk_refs[0]]),
        expected,
    )

    for _name, parameter in named:
        parameter.main_param.zero_()
        parameter.data.zero_()
    target = replace(
        exported,
        policy_version=1,
        local_step_generation=0,
        chunk_refs=_copy_chunk_refs(exported),
    )
    before_prepare = [parameter.main_param.clone() for _name, parameter in named]
    assert prepare_full_parameter_chunked_shard(
        actor,
        target,
        commit_token="commit-1",
        ray_module=fake_ray,
    ) == len(named)
    assert all(torch.equal(parameter.main_param, before) for (_name, parameter), before in zip(named, before_prepare, strict=True))

    assert commit_prepared_full_parameter_shard(
        actor,
        commit_token="commit-1",
        ray_module=fake_ray,
    ) == len(named)
    actual = torch.cat([parameter.main_param.detach().reshape(-1) for spec in manifest.specs for name, parameter in named if name == spec.name])
    assert torch.equal(actual, expected)
    assert actor._external_full_policy_version == 1
    assert actor._external_full_local_step_generation == 0
    assert actor.weights_backuper.calls == ["actor"]
    assert {id(parameter): tuple(id(value) for value in state.values()) for parameter, state in optimizer_state.items()} == before_optimizer_ids
    with pytest.raises(RuntimeError, match="unfinalized"):
        record_full_parameter_local_step(
            actor,
            base_policy_version=1,
            rollout_id=0,
        )
    assert finalize_full_parameter_commit(
        actor,
        commit_token="commit-1",
    ) == len(named)
    assert finalize_full_parameter_commit(
        actor,
        commit_token="commit-1",
    ) == len(named)


def test_chunked_partial_commit_retries_same_token_from_immutable_refs(monkeypatch):
    actor, named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(
        actor,
        policy_version=0,
        max_chunk_bytes=16,
        ray_module=fake_ray,
    )
    target = replace(
        exported,
        policy_version=1,
        local_step_generation=0,
        chunk_refs=_copy_chunk_refs(exported),
    )
    prepare_full_parameter_chunked_shard(
        actor,
        target,
        commit_token="commit-retry",
        ray_module=fake_ray,
    )
    with pytest.raises(RuntimeError, match="prepared"):
        export_full_parameter_chunked_shard(
            actor,
            policy_version=0,
            max_chunk_bytes=16,
            ray_module=fake_ray,
        )
    original = __import__(
        "miles.backends.megatron_utils.full_parameter_state",
        fromlist=["_copy_fragment_chunks_to_masters"],
    )._copy_fragment_chunks_to_masters
    attempts = 0

    def fail_after_copy(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        result = original(*args, **kwargs)
        if attempts == 1:
            raise RuntimeError("injected response loss after copy")
        return result

    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._copy_fragment_chunks_to_masters",
        fail_after_copy,
    )
    with pytest.raises(RuntimeError, match="injected"):
        commit_prepared_full_parameter_shard(
            actor,
            commit_token="commit-retry",
            ray_module=fake_ray,
        )
    assert hasattr(actor, "_external_full_parameter_prepared_cut")
    with pytest.raises(RuntimeError, match="committing"):
        abort_prepared_full_parameter_shard(
            actor,
            commit_token="commit-retry",
        )
    assert commit_prepared_full_parameter_shard(
        actor,
        commit_token="commit-retry",
        ray_module=fake_ray,
    ) == len(named)
    with pytest.raises(RuntimeError, match="unfinalized"):
        export_full_parameter_chunked_shard(
            actor,
            policy_version=1,
            max_chunk_bytes=16,
            ray_module=fake_ray,
        )
    changed_target = replace(
        target,
        fragments=(replace(target.fragments[0], payload_hash="f" * 64),),
    )
    with pytest.raises(RuntimeError, match="reconciled"):
        prepare_full_parameter_chunked_shard(
            actor,
            changed_target,
            commit_token="commit-retry",
            ray_module=fake_ray,
        )
    assert commit_prepared_full_parameter_shard(
        actor,
        commit_token="commit-retry",
        ray_module=fake_ray,
    ) == len(named)
    with pytest.raises(RuntimeError, match="cannot be aborted"):
        abort_prepared_full_parameter_shard(
            actor,
            commit_token="commit-retry",
        )
    assert finalize_full_parameter_commit(
        actor,
        commit_token="commit-retry",
    ) == len(named)
    with pytest.raises(RuntimeError, match="stale"):
        prepare_full_parameter_chunked_shard(
            actor,
            changed_target,
            commit_token="commit-retry",
            ray_module=fake_ray,
        )


def test_chunked_prepare_rejects_corrupt_late_chunk_before_mutation(monkeypatch):
    actor, named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(
        actor,
        policy_version=0,
        max_chunk_bytes=16,
        ray_module=fake_ray,
    )
    target = replace(
        exported,
        policy_version=1,
        local_step_generation=0,
        chunk_refs=_copy_chunk_refs(exported),
    )
    fake_ray.objects[target.chunk_refs[-1][-1]][0] = float("nan")
    before = [parameter.main_param.clone() for _name, parameter in named]

    with pytest.raises(ValueError, match="NaN or Inf"):
        prepare_full_parameter_chunked_shard(
            actor,
            target,
            commit_token="commit-corrupt",
            ray_module=fake_ray,
        )

    assert not hasattr(actor, "_external_full_parameter_prepared_cut")
    assert all(torch.equal(parameter.main_param, value) for (_name, parameter), value in zip(named, before, strict=True))


def test_chunked_prepared_context_and_tokens_fail_closed(monkeypatch):
    actor, _named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(
        actor,
        policy_version=0,
        max_chunk_bytes=16,
        ray_module=fake_ray,
    )
    target = replace(
        exported,
        policy_version=1,
        local_step_generation=0,
        chunk_refs=_copy_chunk_refs(exported),
    )
    prepare_full_parameter_chunked_shard(
        actor,
        target,
        commit_token="commit-first",
        ray_module=fake_ray,
    )

    with pytest.raises(RuntimeError, match="stale"):
        prepare_full_parameter_chunked_shard(
            actor,
            target,
            commit_token="commit-second",
            ray_module=fake_ray,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        commit_prepared_full_parameter_shard(
            actor,
            commit_token="commit-wrong",
            ray_module=fake_ray,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        abort_prepared_full_parameter_shard(
            actor,
            commit_token="commit-wrong",
        )
    assert abort_prepared_full_parameter_shard(
        actor,
        commit_token="commit-first",
    )
    assert not abort_prepared_full_parameter_shard(
        actor,
        commit_token="commit-first",
    )
    with pytest.raises(ValueError, match="token"):
        prepare_full_parameter_chunked_shard(
            actor,
            target,
            commit_token="",
            ray_module=fake_ray,
        )


def test_chunked_commit_rejects_boundary_drift_before_mutation(monkeypatch):
    actor, named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(
        actor,
        policy_version=0,
        max_chunk_bytes=16,
        ray_module=fake_ray,
    )
    target = replace(
        exported,
        policy_version=1,
        local_step_generation=0,
        chunk_refs=_copy_chunk_refs(exported),
    )
    prepare_full_parameter_chunked_shard(
        actor,
        target,
        commit_token="commit-drift",
        ray_module=fake_ray,
    )
    before = [parameter.main_param.clone() for _name, parameter in named]

    actor.opt_param_scheduler.num_steps += 1
    with pytest.raises(RuntimeError, match="became stale"):
        commit_prepared_full_parameter_shard(
            actor,
            commit_token="commit-drift",
            ray_module=fake_ray,
        )
    assert all(torch.equal(parameter.main_param, value) for (_name, parameter), value in zip(named, before, strict=True))
    with pytest.raises(RuntimeError, match="blocks boundary mutation"):
        record_full_parameter_local_step(
            actor,
            base_policy_version=0,
            rollout_id=0,
        )
    assert abort_prepared_full_parameter_shard(
        actor,
        commit_token="commit-drift",
    )


def test_owner_plan_requires_canonical_exact_spec_coverage(monkeypatch):
    actor, _named, _optimizer_state, manifest, _plan = _chunked_actor(monkeypatch)
    other, _named, _optimizer_state, other_manifest, _plan = _chunked_actor(monkeypatch)
    delattr(other, "_external_full_parameter_fragment_plan")
    names = tuple(spec.wire_name for spec in manifest.specs)
    missing = FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="8" * 64,
        fragments=(FullParameterFragmentPlan(0, names[:1], manifest.specs[0].numel),),
    )
    with pytest.raises(RuntimeError, match="exactly cover"):
        install_full_parameter_fragment_plan(other, missing)

    reversed_plan = FullParameterOwnerFragmentPlan(
        topology=other_manifest.topology,
        parameter_layout_hash="8" * 64,
        fragments=(
            FullParameterFragmentPlan(
                0,
                tuple(reversed(names)),
                sum(spec.numel for spec in other_manifest.specs),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="canonical spec order"):
        install_full_parameter_fragment_plan(other, reversed_plan)

    split_reordered = FullParameterOwnerFragmentPlan(
        topology=other_manifest.topology,
        parameter_layout_hash="8" * 64,
        fragments=(
            FullParameterFragmentPlan(
                0,
                names[1:],
                sum(spec.numel for spec in other_manifest.specs[1:]),
            ),
            FullParameterFragmentPlan(
                1,
                names[:1],
                other_manifest.specs[0].numel,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="canonical spec order"):
        install_full_parameter_fragment_plan(other, split_reordered)
    assert actor._external_full_parameter_fragment_plan is not None


def test_chunked_export_rejects_unbounded_ray_objects(monkeypatch):
    actor, _named, _optimizer_state, _manifest, _plan = _chunked_actor(monkeypatch)
    with pytest.raises(ValueError, match="1 GiB"):
        export_full_parameter_chunked_shard(
            actor,
            policy_version=0,
            max_chunk_bytes=FULL_PARAMETER_MAX_CHUNK_BYTES,
            ray_module=_FakeRay(),
        )


def _manifest_actor(**args):
    named = parameters()
    masters = [parameter.main_param for _name, parameter in named]
    actor = SimpleNamespace(
        args=SimpleNamespace(lora_rank=0, lora_adapter_path=None, **args),
        optimizer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": masters}], state={})),
    )
    return actor, named


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"lora_rank": 8}, "without --full-parameter-lora-hybrid"),
        ({"full_parameter_lora_hybrid": True}, "requires a LoRA actor"),
        (
            {"lora_rank": 8, "full_parameter_lora_hybrid": True, "lora_base_cpu_backup": True},
            "incompatible with --lora-base-cpu-backup",
        ),
    ],
)
def test_lora_actor_needs_the_hybrid_profile(monkeypatch, args, message):
    actor, named = _manifest_actor()
    vars(actor.args).update(args)
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr("miles.backends.megatron_utils.full_parameter_state.full_parameter_topology", topology)
    with pytest.raises(RuntimeError, match=message):
        full_parameter_shard_manifest(actor)


def test_lora_hybrid_actor_streams_every_trainable_parameter(monkeypatch):
    actor, named = _manifest_actor(full_parameter_lora_hybrid=True, lora_base_cpu_backup=False)
    actor.args.lora_rank = 8
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr("miles.backends.megatron_utils.full_parameter_state.full_parameter_topology", topology)

    manifest = full_parameter_shard_manifest(actor)

    assert [spec.name for spec in manifest.specs] == sorted(name for name, _parameter in named)



def _stub_megatron_argument_imports(monkeypatch):
    import sys
    import types

    for name, attributes in (
        ("megatron.core.tokenizers.utils.build_tokenizer", {"vocab_size_with_padding": None}),
        ("megatron.training.arguments", {"parse_args": None, "validate_args": None}),
    ):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attributes))
    monkeypatch.delitem(sys.modules, "miles.backends.megatron_utils.arguments", raising=False)


def _megatron_default_args(**overrides):
    """Every field ``set_default_megatron_args`` reads, for a non-DSV4 Adam run."""

    values = dict(
        optimizer="adam", fp16=False, seq_length=128, multi_latent_attention=False, rope_type="rope",
        vocab_size=0, padded_vocab_size=None, tokenizer_model="tok", tokenizer_type="HuggingFaceTokenizer",
        hf_checkpoint="/models/x", spec=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(("hybrid", "distributed"), [(False, True), (True, False)])
def test_lora_hybrid_keeps_complete_fp32_masters(monkeypatch, hybrid, distributed):
    _stub_megatron_argument_imports(monkeypatch)
    from miles.backends.megatron_utils.arguments import set_default_megatron_args

    args = _megatron_default_args(full_parameter_lora_hybrid=hybrid)
    set_default_megatron_args(args)
    assert args.use_distributed_optimizer is distributed


# Megatron distributed optimizer at DP=1 (+ HybridDeviceOptimizer CPU offload).
# The stubs mirror the attributes Megatron sets (distrib_optimizer.py,
# cpu_offloading/hybrid_optimizer.py): ``main_param`` is the flat FP32 range the
# base optimizer steps and ``main_param_sharded`` is True; native-FP32
# parameters are stepped through ``shard_fp32_groups`` views; under the
# precision-aware optimizer ``main_param`` is None and the base optimizer owns
# the BF16 ``shard_float16_groups`` view while the hybrid optimizer steps an FP32
# inner copy (``param_to_inner_param``).

_PROJ = "module.decoder.layers.0.proj.weight"
_NORM = "module.decoder.final_norm.weight"
_A_LOG = "module.decoder.layers.0.A_log"


def _distopt_actor(monkeypatch, *, mode: str):
    """``mode``: "main" (FP32 main_param ranges), "main-offload" (+ host copies),
    "precision-aware" (BF16 owned views, FP32 inner copies on the host)."""

    first = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    native = torch.nn.Parameter(torch.tensor([0.25, 0.5], dtype=torch.float32))
    initial = {id(first): torch.arange(6, dtype=torch.float32), id(second): torch.tensor([7.0, 8.0])}
    named = ((_PROJ, first), (_NORM, second), (_A_LOG, native))
    native_shard = native.detach().view(-1)
    bf16_views = []
    inner = {}
    for parameter in (first, second):
        parameter.data.copy_(initial[id(parameter)].view(parameter.shape).to(torch.bfloat16))
        parameter.main_param_sharded = True
        if mode == "precision-aware":
            parameter.main_param = None
            view = parameter.detach().view(-1)
            bf16_views.append(view)
            inner[view] = initial[id(parameter)].clone()
        else:
            parameter.main_param = initial[id(parameter)].clone()
            if mode == "main-offload":
                inner[parameter.main_param] = parameter.main_param.clone()
    if mode != "main":
        inner[native_shard] = native_shard.clone()
    owned = [native_shard, *(bf16_views or [first.main_param, second.main_param])]
    state = {
        tensor: {"exp_avg": torch.ones(tensor.numel()), "exp_avg_sq": torch.full((tensor.numel(),), 2.0)}
        for tensor in owned
    }
    if mode == "precision-aware":
        # HybridDeviceOptimizer._sync_sub_optimizers_state_to_hdo (param_update_in_fp32).
        for view in bf16_views:
            state[view]["master_param"] = inner[view]
    base = SimpleNamespace(param_groups=[{"params": owned}], state=state)
    if inner:
        base.param_to_inner_param = inner
    wrapper = SimpleNamespace(
        optimizer=base,
        model_fp32_groups=[[native]],
        shard_fp32_groups=[[native_shard]],
        model_float16_groups=[[first, second]],
        shard_float16_groups=[bf16_views or [first.detach().view(-1), second.detach().view(-1)]],
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(lora_rank=0, lora_adapter_path=None, num_steps_per_rollout=1, global_batch_size=1),
        optimizer=wrapper,
        opt_param_scheduler=SimpleNamespace(num_steps=0),
        weights_backuper=SimpleNamespace(backup=lambda role: None),
        _last_rollout_id=0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._named_actor_parameters",
        lambda _actor: named,
    )
    monkeypatch.setattr("miles.backends.megatron_utils.full_parameter_state.full_parameter_topology", topology)

    def copy_masters_to_model(_actor):
        # Megatron: the distributed optimizer copies main ranges into the param
        # buffer; under the precision-aware optimizer it returns early.
        if mode != "precision-aware":
            for parameter in (first, second):
                parameter.data.copy_(parameter.main_param.view(parameter.shape).to(torch.bfloat16))

    monkeypatch.setattr(
        "miles.backends.megatron_utils.full_parameter_state._copy_masters_to_model",
        copy_masters_to_model,
    )
    initialize_full_parameter_tracking(actor, 0)
    return actor, named, inner, state


def _assert_inner_copies_agree(inner):
    for owned, copy in inner.items():
        assert torch.equal(owned.to(torch.float32), copy.to(owned.dtype).to(torch.float32))


_MODES = ["main", "main-offload", "precision-aware"]


@pytest.mark.parametrize("mode", _MODES)
def test_distributed_optimizer_dp1_masters_round_trip(monkeypatch, mode):
    actor, named, inner, state = _distopt_actor(monkeypatch, mode=mode)
    manifest = full_parameter_shard_manifest(actor)
    assert {spec.name: spec.shape for spec in manifest.specs} == {_PROJ: (2, 3), _NORM: (2,), _A_LOG: (2,)}
    exported = export_full_parameter_shard(actor, policy_version=0)
    by_name = {name.rsplit("::", 1)[-1]: value for name, value in exported.tensors.items()}
    assert torch.equal(by_name[_PROJ], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    moments = {id(tensor): {key: value.clone() for key, value in values.items()} for tensor, values in state.items()}
    actor.opt_param_scheduler.num_steps = 1
    proof_before = full_parameter_optimizer_state(actor)

    incoming = replace(
        exported,
        policy_version=1,
        tensors={name: value + 0.5 for name, value in exported.tensors.items()},
    )
    assert apply_full_parameter_shard(actor, incoming) == 3

    first, _second, native = (parameter for _name, parameter in named)
    expected = (torch.arange(6, dtype=torch.float32) + 0.5).reshape(2, 3)
    assert torch.equal(first, expected.to(torch.bfloat16))
    assert torch.equal(native, torch.tensor([0.75, 1.0]))
    exported_after = export_full_parameter_shard(actor, policy_version=1)
    assert torch.equal({n.rsplit("::", 1)[-1]: v for n, v in exported_after.tensors.items()}[_PROJ], expected)
    # The FP32 copy the hybrid optimizer steps follows the overwrite.
    _assert_inner_copies_agree(inner)
    # Adam moments keep their objects and values.
    for tensor, values in state.items():
        assert all(torch.equal(values[key], moments[id(tensor)][key]) for key in values if key != "master_param")
    proof = full_parameter_optimizer_state(actor)
    # The apply proof sees Adam state only, not the hybrid master mirror.
    assert proof.selected_state_sha256 == proof_before.selected_state_sha256
    assert proof.optimizer_state_tensor_count == proof_before.optimizer_state_tensor_count == 6
    assert proof.populated_parameter_count == 3
    assert proof.model_master_parameter_count == 3


@pytest.mark.parametrize("mode", _MODES)
def test_distributed_optimizer_chunked_commit_writes_the_stepped_copies(monkeypatch, mode):
    actor, named, inner, _state = _distopt_actor(monkeypatch, mode=mode)
    manifest = full_parameter_shard_manifest(actor)
    names = tuple(spec.wire_name for spec in manifest.specs)
    plan = FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="9" * 64,
        fragments=(FullParameterFragmentPlan(fragment_id=0, wire_names=names, numel=sum(spec.numel for spec in manifest.specs)),),
    )
    install_full_parameter_fragment_plan(actor, plan)
    fake_ray = _FakeRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=16, ray_module=fake_ray)
    # Drift both sides of every pair so the commit must restore them.
    for _name, parameter in named:
        parameter.data.zero_()
        if getattr(parameter, "main_param", None) is not None:
            parameter.main_param.zero_()
    for copy in inner.values():
        copy.fill_(-1.0)
    target = replace(exported, policy_version=1, local_step_generation=0, chunk_refs=_copy_chunk_refs(exported))
    prepare_full_parameter_chunked_shard(actor, target, commit_token="c", ray_module=fake_ray)
    assert commit_prepared_full_parameter_shard(actor, commit_token="c", ray_module=fake_ray) == 3
    assert torch.equal(named[0][1], torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16))
    assert torch.equal(named[2][1], torch.tensor([0.25, 0.5]))
    _assert_inner_copies_agree(inner)


def test_distributed_optimizer_partial_or_missing_master_is_refused():
    parameter = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    parameter.main_param = torch.zeros(3, dtype=torch.float32)  # DP=2: half the range
    parameter.main_param_sharded = True
    with pytest.raises(RuntimeError, match="partial DP shard"):
        make_full_parameter_shard_state(
            policy_version=0,
            role="actor",
            topology=topology(),
            named_parameters=(("module.w", parameter),),
        )
    parameter.main_param = None  # a rank owning none of it, or precision-aware without offload
    with pytest.raises(RuntimeError, match="DP=1"):
        make_full_parameter_shard_state(
            policy_version=0,
            role="actor",
            topology=topology(),
            named_parameters=(("module.w", parameter),),
        )


def _split_plan(manifest, *, halves=((0, 3), (3, 6))):
    """One fragment per member; the 6-scalar matrix spans two fragments."""

    members = []
    for spec in manifest.specs:
        ranges = halves if spec.numel == 6 else ((0, spec.numel),)
        members.extend((spec.wire_name, start, stop) for start, stop in ranges)
    return FullParameterOwnerFragmentPlan(
        topology=manifest.topology,
        parameter_layout_hash="7" * 64,
        fragments=tuple(
            FullParameterFragmentPlan(index, (name,), stop - start, ((start, stop),))
            for index, (name, start, stop) in enumerate(members)
        ),
    )


def test_plan_hash_of_whole_parameter_plans_is_unchanged():
    plan = FullParameterOwnerFragmentPlan(
        topology=topology(),
        parameter_layout_hash="9" * 64,
        fragments=(FullParameterFragmentPlan(0, ("actor::s::a", "actor::s::b"), 5),),
    )
    legacy = {
        "schema": 1,
        "topology": asdict(plan.topology),
        "parameter_layout_hash": plan.parameter_layout_hash,
        "fragments": [asdict(fragment) for fragment in plan.fragments],
    }
    del legacy["fragments"][0]["flat_ranges"]
    assert plan.plan_hash == hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_split_parameter_round_trips_through_chunked_commit(monkeypatch):
    actor, named, _optimizer_state, manifest, _plan = _chunked_actor(monkeypatch)
    delattr(actor, "_external_full_parameter_fragment_plan")
    plan = _split_plan(manifest)
    assert install_full_parameter_fragment_plan(actor, plan) == 3
    fake_ray = _FakeRay()
    expected = torch.cat([parameter.main_param.detach().reshape(-1).clone() for spec in manifest.specs for name, parameter in named if name == spec.name])
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=fake_ray)
    assert [fragment.numel for fragment in exported.fragments] == [fragment.numel for fragment in plan.fragments]
    flat = torch.cat([fake_ray.get(reference) for references in exported.chunk_refs for reference in references])
    assert torch.equal(flat, expected)
    for _name, parameter in named:
        parameter.main_param.zero_()
        parameter.data.zero_()
    target = replace(exported, policy_version=1, local_step_generation=0, chunk_refs=_copy_chunk_refs(exported))
    prepare_full_parameter_chunked_shard(actor, target, commit_token="split", ray_module=fake_ray)
    # The split parameter counts once.
    assert commit_prepared_full_parameter_shard(actor, commit_token="split", ray_module=fake_ray) == len(named)
    actual = torch.cat([parameter.main_param.detach().reshape(-1) for spec in manifest.specs for name, parameter in named if name == spec.name])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("halves", [((0, 3), (4, 6)), ((0, 3), (3, 5)), ((3, 6), (0, 3))])
def test_split_parameter_ranges_must_tile_in_order(monkeypatch, halves):
    actor, _named, _optimizer_state, manifest, _plan = _chunked_actor(monkeypatch)
    delattr(actor, "_external_full_parameter_fragment_plan")
    with pytest.raises(RuntimeError, match="tile|exactly cover|canonical"):
        install_full_parameter_fragment_plan(actor, _split_plan(manifest, halves=halves))


def test_flat_ranges_must_match_wire_names():
    with pytest.raises(ValueError, match="flat ranges"):
        FullParameterFragmentPlan(0, ("actor::s::a",), 3, ((0, 2),))
    with pytest.raises(ValueError, match="flat ranges"):
        FullParameterFragmentPlan(0, ("actor::s::a",), 3, ((0, 2), (2, 3)))


def _restore_donor_scheduler(actor, num_steps):
    # A peer snapshot restore loads the donor's LR scheduler progress.
    actor.opt_param_scheduler.num_steps = num_steps


def _chunked_target(actor, fake_ray, policy_version):
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=16, ray_module=fake_ray)
    return replace(exported, policy_version=policy_version, local_step_generation=0, chunk_refs=_copy_chunk_refs(exported))


def test_restored_scheduler_progress_without_rebase_has_no_receipt(monkeypatch):
    actor, _named, _state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    target = _chunked_target(actor, fake_ray, 1)
    _restore_donor_scheduler(actor, 7)
    with pytest.raises(RuntimeError, match="no local-step receipt"):
        prepare_full_parameter_chunked_shard(actor, target, commit_token="commit-1", ray_module=fake_ray)


def test_rebase_after_restore_lets_first_apply_through(monkeypatch):
    actor, named, _state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    target = _chunked_target(actor, fake_ray, 1)
    _restore_donor_scheduler(actor, 7)
    assert rebase_full_parameter_scheduler_anchor(actor, reason="peer snapshot restore") == 7
    assert prepare_full_parameter_chunked_shard(actor, target, commit_token="commit-1", ray_module=fake_ray) == len(named)
    assert commit_prepared_full_parameter_shard(actor, commit_token="commit-1", ray_module=fake_ray) == len(named)
    assert finalize_full_parameter_commit(actor, commit_token="commit-1") == len(named)
    assert actor._external_full_scheduler_anchor_steps == 7
    # Later progress still needs a receipt.
    actor.opt_param_scheduler.num_steps = 8
    with pytest.raises(RuntimeError, match="no local-step receipt"):
        export_full_parameter_chunked_shard(actor, policy_version=1, max_chunk_bytes=16, ray_module=fake_ray)


def test_rebase_refuses_pending_local_progress_or_commit(monkeypatch):
    actor, _named, _state, _manifest, _plan = _chunked_actor(monkeypatch)
    fake_ray = _FakeRay()
    with pytest.raises(ValueError, match="explicit reason"):
        rebase_full_parameter_scheduler_anchor(actor, reason=" ")
    target = _chunked_target(actor, fake_ray, 1)
    prepare_full_parameter_chunked_shard(actor, target, commit_token="commit-1", ray_module=fake_ray)
    with pytest.raises(RuntimeError, match="prepared full-parameter cut"):
        rebase_full_parameter_scheduler_anchor(actor, reason="peer snapshot restore")
    abort_prepared_full_parameter_shard(actor, commit_token="commit-1")
    actor._external_full_local_step_generation = 1
    with pytest.raises(RuntimeError, match="discard local-step progress"):
        rebase_full_parameter_scheduler_anchor(actor, reason="peer snapshot restore")
