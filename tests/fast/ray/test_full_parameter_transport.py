from __future__ import annotations

from dataclasses import replace

import pytest

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterCommitStatus,
    FullParameterFragmentDescriptor,
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    FullParameterTopology,
)
from miles.ray.full_parameter_transport import (
    FullParameterCommitInDoubtError,
    FullParameterChunkedCut,
    apply_chunked_cut,
    assemble_full_parameter_chunked_target,
    full_parameter_commit_token,
    install_fragment_plans,
    promote_full_parameter_chunked_cut,
    release_full_parameter_chunked_cut,
)

pytestmark = pytest.mark.asyncio


def _topology(rank: int) -> FullParameterTopology:
    return FullParameterTopology(
        tp_rank=rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        cp_rank=0,
        cp_size=1,
        dp_rank=0,
        dp_size=1,
    )


def _shard(
    rank: int,
    fragment_id: int,
    *,
    policy_version: int,
    local_step_generation: int,
) -> FullParameterChunkedShardState:
    chunk = FullParameterChunkDescriptor(0, 0, 4, str(rank + 1) * 64)
    fragment = FullParameterFragmentDescriptor(
        fragment_id,
        4,
        str(rank + 3) * 64,
        (chunk,),
    )
    return FullParameterChunkedShardState(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        topology=_topology(rank),
        parameter_layout_hash="9" * 64,
        plan_hash=str(rank + 5) * 64,
        fragments=(fragment,),
        chunk_refs=[[object()]],
    )


def _cut(*, policy_version: int, local_step_generation: int) -> FullParameterChunkedCut:
    return FullParameterChunkedCut(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        parameter_layout_hash="9" * 64,
        shards=(
            _shard(
                0,
                0,
                policy_version=policy_version,
                local_step_generation=local_step_generation,
            ),
            _shard(
                1,
                1,
                policy_version=policy_version,
                local_step_generation=local_step_generation,
            ),
        ),
    )


class _Remote:
    def __init__(self, callback):
        self.callback = callback

    async def remote(self, *args):
        return self.callback(*args)


class _Actor:
    def __init__(
        self,
        rank,
        events,
        *,
        prepare_error=None,
        prepare_count=1,
        commit_count=1,
        commit_errors=(),
        finalize_errors=(),
        abort_race_commit=False,
    ):
        self.rank = rank
        self.events = events
        self.phase = "ABSENT"
        self.full_parameter_topology = _Remote(lambda: _topology(rank))
        commit_errors = list(commit_errors)
        finalize_errors = list(finalize_errors)

        def prepare(state, token):
            self.events.append(("prepare", rank, token, state.topology.tp_rank))
            if self.phase != "ABSENT":
                return prepare_count
            if prepare_error is not None:
                raise prepare_error
            self.phase = "PREPARED"
            return prepare_count

        def commit(token):
            self.events.append(("commit", rank, token))
            if self.phase in {"COMMITTED", "FINALIZED"}:
                return commit_count
            self.phase = "COMMITTING"
            if commit_errors:
                error = commit_errors.pop(0)
                if error is not None:
                    raise error
            self.phase = "COMMITTED"
            return commit_count

        def finalize(token):
            self.events.append(("finalize", rank, token))
            if self.phase == "FINALIZED":
                return commit_count
            if finalize_errors:
                error = finalize_errors.pop(0)
                if error is not None:
                    raise error
            self.phase = "FINALIZED"
            return commit_count

        def abort(token):
            self.events.append(("abort", rank, token))
            if abort_race_commit and self.phase == "PREPARED":
                self.phase = "COMMITTING"
            if self.phase in {"COMMITTING", "COMMITTED", "FINALIZED"}:
                raise RuntimeError("cannot abort")
            self.phase = "ABSENT"
            return prepare_error is None

        def status(token, policy_version, state_identity_hash):
            self.events.append(("status", rank, token, self.phase))
            return FullParameterCommitStatus(
                token,
                self.phase,
                policy_version,
                0 if self.phase == "ABSENT" else prepare_count,
                state_identity_hash,
            )

        def install(plan):
            self.events.append(("install", rank, plan.topology.tp_rank))
            return len(plan.fragments)

        self.prepare_full_parameter_chunked_shard = _Remote(prepare)
        self.commit_prepared_full_parameter_shard = _Remote(commit)
        self.finalize_full_parameter_commit = _Remote(finalize)
        self.abort_prepared_full_parameter_shard = _Remote(abort)
        self.full_parameter_commit_status = _Remote(status)
        self.install_full_parameter_fragment_plan = _Remote(install)


class _Group:
    def __init__(self, actors):
        self._actor_handles = list(actors)

    async def _broadcast(self, name, *args):
        return [await getattr(actor, name).remote(*args) for actor in self._actor_handles]


async def test_two_phase_apply_commits_only_after_every_prepare():
    events = []
    group = _Group([_Actor(0, events), _Actor(1, events)])
    target = _cut(policy_version=1, local_step_generation=0)

    assert await apply_chunked_cut(group, target, commit_token="cut-1") == 2

    prepare_positions = [index for index, event in enumerate(events) if event[0] == "prepare"]
    commit_positions = [index for index, event in enumerate(events) if event[0] == "commit"]
    assert len(prepare_positions) == len(commit_positions) == 2
    assert max(prepare_positions) < min(commit_positions)
    assert len([event for event in events if event[0] == "finalize"]) == 2
    assert not [event for event in events if event[0] == "abort"]


async def test_default_commit_token_is_stable_for_the_same_cut():
    target = _cut(policy_version=1, local_step_generation=0)
    token_sets = []
    for _ in range(2):
        events = []
        group = _Group([_Actor(0, events), _Actor(1, events)])
        assert await apply_chunked_cut(group, target) == 2
        token_sets.append({event[2] for event in events if event[0] == "prepare"})
    assert token_sets[0] == token_sets[1]
    assert next(iter(token_sets[0])).startswith("fullparam:")


async def test_prepare_failure_aborts_all_and_launches_no_commit():
    events = []
    group = _Group(
        [
            _Actor(0, events),
            _Actor(1, events, prepare_error=ValueError("bad chunk")),
        ]
    )

    with pytest.raises(RuntimeError, match="prepare"):
        await apply_chunked_cut(
            group,
            _cut(policy_version=1, local_step_generation=0),
            commit_token="cut-fail",
        )

    assert len([event for event in events if event[0] == "prepare"]) == 2
    assert len([event for event in events if event[0] == "abort"]) == 2
    assert not [event for event in events if event[0] == "commit"]


async def test_prepare_abort_race_is_classified_in_doubt():
    events = []
    group = _Group(
        [
            _Actor(0, events, abort_race_commit=True),
            _Actor(1, events, prepare_error=ValueError("bad chunk")),
        ]
    )

    with pytest.raises(FullParameterCommitInDoubtError) as captured:
        await apply_chunked_cut(
            group,
            _cut(policy_version=1, local_step_generation=0),
            commit_token="cut-abort-race",
        )
    assert captured.value.phase == "prepare-abort-race"


async def test_explicit_empty_commit_token_fails_closed_before_prepare():
    events = []
    group = _Group([_Actor(0, events), _Actor(1, events)])

    with pytest.raises(ValueError, match="token"):
        await apply_chunked_cut(
            group,
            _cut(policy_version=1, local_step_generation=0),
            commit_token="",
        )

    assert not events


async def test_commit_counts_must_match_prepare_on_each_rank():
    events = []
    group = _Group(
        [
            _Actor(0, events, prepare_count=1, commit_count=2),
            _Actor(1, events, prepare_count=2, commit_count=1),
        ]
    )

    with pytest.raises(FullParameterCommitInDoubtError, match="count-reconciliation"):
        await apply_chunked_cut(
            group,
            _cut(policy_version=1, local_step_generation=0),
            commit_token="cut-counts",
        )

    assert len([event for event in events if event[0] == "prepare"]) == 2
    assert len([event for event in events if event[0] == "commit"]) == 2
    assert not [event for event in events if event[0] == "abort"]


async def test_commit_retries_every_rank_with_same_token_before_finalize():
    events = []
    group = _Group(
        [
            _Actor(0, events),
            _Actor(
                1,
                events,
                commit_errors=(RuntimeError("late chunk copy failed"), None),
            ),
        ]
    )

    target = _cut(policy_version=1, local_step_generation=0)
    expected_token = full_parameter_commit_token(target, "cut-retry")
    assert (
        await apply_chunked_cut(
            group,
            target,
            commit_token="cut-retry",
        )
        == 2
    )
    assert [event[:3] for event in events if event[0] == "commit"] == [
        ("commit", 0, expected_token),
        ("commit", 1, expected_token),
        ("commit", 0, expected_token),
        ("commit", 1, expected_token),
    ]
    assert len([event for event in events if event[0] == "finalize"]) == 2


async def test_finalize_response_loss_is_retried_idempotently():
    events = []
    group = _Group(
        [
            _Actor(0, events),
            _Actor(
                1,
                events,
                finalize_errors=(RuntimeError("response lost"), None),
            ),
        ]
    )

    assert (
        await apply_chunked_cut(
            group,
            _cut(policy_version=1, local_step_generation=0),
            commit_token="cut-finalize-retry",
        )
        == 2
    )
    assert len([event for event in events if event[0] == "finalize"]) == 4


async def test_permanent_commit_failure_is_in_doubt_and_never_aborts():
    events = []
    group = _Group(
        [
            _Actor(0, events),
            _Actor(
                1,
                events,
                commit_errors=(
                    RuntimeError("actor unavailable"),
                    RuntimeError("actor unavailable"),
                ),
            ),
        ]
    )

    target = _cut(policy_version=1, local_step_generation=0)
    expected_token = full_parameter_commit_token(target, "cut-in-doubt")
    with pytest.raises(FullParameterCommitInDoubtError) as captured:
        await apply_chunked_cut(
            group,
            target,
            commit_token="cut-in-doubt",
            max_commit_attempts=2,
        )
    assert captured.value.commit_token == expected_token
    assert captured.value.phase == "commit"
    assert not [event for event in events if event[0] == "abort"]
    assert not [event for event in events if event[0] == "finalize"]


async def test_new_controller_reconciles_same_token_after_partial_rank_commit():
    events = []
    actors = [
        _Actor(0, events),
        _Actor(
            1,
            events,
            commit_errors=(RuntimeError("late copy failure"), None),
        ),
    ]
    group = _Group(actors)
    target = _cut(policy_version=1, local_step_generation=0)
    with pytest.raises(FullParameterCommitInDoubtError):
        await apply_chunked_cut(
            group,
            target,
            commit_token="cut-controller-restart",
            max_commit_attempts=1,
        )
    assert [actor.phase for actor in actors] == ["COMMITTED", "COMMITTING"]
    assert (
        await apply_chunked_cut(
            group,
            target,
            commit_token="cut-controller-restart",
        )
        == 2
    )
    assert [actor.phase for actor in actors] == ["FINALIZED", "FINALIZED"]
    assert not [event for event in events if event[0] == "abort"]


async def test_promote_reuses_refs_and_release_is_caller_local():
    local = _cut(policy_version=0, local_step_generation=1)
    promoted = promote_full_parameter_chunked_cut(local, 1)

    assert promoted.policy_version == 1
    assert promoted.local_step_generation == 0
    for local_shard, promoted_shard in zip(
        local.shards,
        promoted.shards,
        strict=True,
    ):
        assert local_shard.chunk_refs is not promoted_shard.chunk_refs
        assert local_shard.chunk_refs[0][0] is promoted_shard.chunk_refs[0][0]

    assert release_full_parameter_chunked_cut(promoted) == 2
    assert all(reference is not None for shard in local.shards for references in shard.chunk_refs for reference in references)
    assert release_full_parameter_chunked_cut(promoted) == 0
    assert release_full_parameter_chunked_cut(local) == 2


async def test_assemble_target_maps_authoritative_refs_to_fragment_owners():
    reference = _cut(policy_version=0, local_step_generation=1)
    alternate_shards = []
    for shard in reference.shards:
        descriptor = replace(
            shard.fragments[0],
            payload_hash="a" * 64,
            chunks=(replace(shard.fragments[0].chunks[0], payload_hash="b" * 64),),
        )
        alternate_shards.append(
            replace(
                shard,
                fragments=(descriptor,),
                chunk_refs=[[object()]],
            )
        )
    authoritative = {
        0: (
            reference.shards[0].fragments[0],
            reference.shards[0].chunk_refs[0],
        ),
        1: (
            alternate_shards[1].fragments[0],
            alternate_shards[1].chunk_refs[0],
        ),
    }

    target = assemble_full_parameter_chunked_target(
        reference,
        target_policy_version=1,
        authoritative_fragments=authoritative,
    )

    assert target.policy_version == 1
    assert target.local_step_generation == 0
    assert target.shards[0].fragments[0] is authoritative[0][0]
    assert target.shards[1].fragments[0] is authoritative[1][0]
    for shard in target.shards:
        fragment_id = shard.fragments[0].fragment_id
        assert shard.chunk_refs[0] is not authoritative[fragment_id][1]
        assert shard.chunk_refs[0][0] is authoritative[fragment_id][1][0]

    with pytest.raises(ValueError, match="exactly cover"):
        assemble_full_parameter_chunked_target(
            reference,
            target_policy_version=1,
            authoritative_fragments={0: authoritative[0]},
        )

    with pytest.raises(ValueError, match="IDs are malformed"):
        assemble_full_parameter_chunked_target(
            reference,
            target_policy_version=1,
            authoritative_fragments={
                False: authoritative[0],
                1: authoritative[1],
            },
        )

    with pytest.raises(ValueError, match="owner-plan identity"):
        assemble_full_parameter_chunked_target(
            reference,
            target_policy_version=1,
            authoritative_fragments={
                0: authoritative[1],
                1: authoritative[1],
            },
        )


async def test_group_plan_install_requires_complete_global_fragment_sweep():
    events = []
    group = _Group([_Actor(0, events), _Actor(1, events)])
    plans = tuple(
        FullParameterOwnerFragmentPlan(
            topology=_topology(rank),
            parameter_layout_hash="9" * 64,
            fragments=(
                FullParameterFragmentPlan(
                    fragment_id=rank,
                    wire_names=(f"actor::tp{rank}-of-2::model.weight",),
                    numel=4,
                ),
            ),
        )
        for rank in range(2)
    )
    assert await install_fragment_plans(group, plans) == 2
    assert len([event for event in events if event[0] == "install"]) == 2

    invalid = (
        plans[0],
        replace(
            plans[1],
            fragments=(replace(plans[1].fragments[0], fragment_id=2),),
        ),
    )
    with pytest.raises(ValueError, match="complete fragment sweep"):
        await install_fragment_plans(group, invalid)
