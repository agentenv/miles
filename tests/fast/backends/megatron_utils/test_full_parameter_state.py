from __future__ import annotations

from dataclasses import replace
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
        "miles.backends.megatron_utils.trainable_state._copy_masters_to_model",
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
        "miles.backends.megatron_utils.trainable_state._copy_masters_to_model",
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
