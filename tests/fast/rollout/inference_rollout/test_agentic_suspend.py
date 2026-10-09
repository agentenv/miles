"""--agentic-suspend-between-turns: the cut-off suspends agentic trajectories
between two model turns (never inside a tool call) and the next rollout
continues them (agentenv fork; yeto agentic-rollout-utilization 5.1)."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_common, inference_rollout_train


class FakeAgent:
    """An agent loop: model turn -> tool call -> (gate) -> model turn -> done.

    ``gate`` is the agent's model-turn gate (what the ``suspend``/``resume``
    hooks close/open): it is only checked before a model request, so a tool
    call that is running when the gate closes completes and its result is
    written into the conversation first."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.gate.set()
        self.events: list[str] = []
        self.tool_started = asyncio.Event()
        self.finish_tool = asyncio.Event()
        self.conversation: list[str] = []

    async def model_turn(self, n: int) -> None:
        if not self.gate.is_set():
            self.events.append(f"parked_before_turn{n}")
            await self.gate.wait()
        self.events.append(f"turn{n}")
        self.conversation.append(f"assistant{n}")

    async def run(self) -> list[SimpleNamespace]:
        await self.model_turn(1)
        self.tool_started.set()
        self.events.append("tool_start")
        await self.finish_tool.wait()
        self.conversation.append("tool_result")
        self.events.append("tool_written")
        await self.model_turn(2)
        return [SimpleNamespace(response="done", response_length=7, metadata={})]

    async def suspend(self, _args) -> None:
        self.events.append("suspend_hook")
        self.gate.clear()

    async def resume(self, _args) -> None:
        self.events.append("resume_hook")
        self.gate.set()


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch):
    posted: list[str] = []

    async def fake_get(url: str):
        if url.endswith("/get_load"):
            return [{"num_reqs": 0, "num_waiting_reqs": 0}]
        return {"urls": ["http://w"]}

    async def fake_post(url: str, payload: dict) -> None:
        posted.append(url)

    monkeypatch.setattr(inference_rollout_train, "get", fake_get)
    monkeypatch.setattr(inference_rollout_train, "post", fake_post)
    monkeypatch.setattr(inference_rollout_train, "SUSPEND_ENGINE_ABORT_INTERVAL_S", 0.0)
    return posted


def _hook(monkeypatch: pytest.MonkeyPatch, agent: FakeAgent, calls: list[str], posted: list[str]):
    async def fake_hook(args, name: str):
        calls.append(f"{name}@{len(posted)}")  # engine aborts already posted when the hook ran
        await getattr(agent, name)(args)

    monkeypatch.setattr(inference_rollout_train, "call_agent_hook", fake_hook)


def _state(**kw) -> SimpleNamespace:
    args = Namespace(
        agentic_suspend_between_turns=True,
        agentic_suspend_max_rounds=1,
        partial_rollout=False,
        sglang_router_ip="router",
        sglang_router_port=30000,
        use_miles_router=True,
        **kw,
    )
    return SimpleNamespace(args=args, aborted=False)


async def test_cutoff_during_a_tool_call_waits_for_the_result_then_parks(monkeypatch, engine) -> None:
    agent = FakeAgent()
    calls: list[str] = []
    _hook(monkeypatch, agent, calls, engine)
    state = _state()
    task = asyncio.create_task(agent.run())
    await agent.tool_started.wait()  # the cut-off lands while the tool runs

    stats = await inference_rollout_train.suspend(state, {task}, rollout_id=3)

    # the gate closed BEFORE the engines aborted the in-flight model turns
    assert calls == ["suspend@0"] and len(engine) == inference_rollout_train.SUSPEND_ENGINE_ABORT_REPEATS
    assert not task.done() and not task.cancelled()  # the tool call is never interrupted
    assert stats["suspended_groups"] == 1 and stats["over_age_cancelled_groups"] == 0
    assert stats["engines_idle_confirmed"] == 1
    assert state.args.rollout_suspend_stats == stats
    agent.finish_tool.set()  # the tool returns during the suspension
    for _ in range(5):
        await asyncio.sleep(0)
    # its result is written back, then the trajectory waits before the next model turn
    assert agent.events == ["turn1", "tool_start", "suspend_hook", "tool_written", "parked_before_turn2"]
    assert agent.conversation == ["assistant1", "tool_result"]
    assert not task.done()

    adopted, started = await inference_rollout_train.resume(state, rollout_id=4)
    assert adopted == {task} and started == {task: 3}
    assert state.args.rollout_resume_stats == {"resumed_groups": 1, "resumed_done_groups": 0}
    group = await asyncio.wait_for(task, 1.0)
    assert agent.events[-2:] == ["resume_hook", "turn2"]
    assert group[0].response == "done"


async def test_a_group_past_the_round_window_is_cancelled_and_counted(monkeypatch, engine) -> None:
    agent = FakeAgent()
    _hook(monkeypatch, agent, [], engine)
    state = _state()
    old = asyncio.create_task(agent.run())
    await agent.tool_started.wait()
    fresh = asyncio.create_task(asyncio.Event().wait())
    await inference_rollout_train.suspend(state, {old}, rollout_id=3)  # old started at 3
    await inference_rollout_train.resume(state, rollout_id=4)

    stats = await inference_rollout_train.suspend(state, {old, fresh}, rollout_id=4)

    # old: started 3, next trainable rollout 5 -> 2 rounds > max 1: cancelled
    assert old.cancelled() and not fresh.done()
    assert stats["suspended_groups"] == 1 and stats["over_age_cancelled_groups"] == 1
    assert stats["over_age_unknown_groups"] == 1  # cancelled while running: tokens unknown
    assert inference_rollout_train._suspended(state) == {fresh: 4}
    fresh.cancel()


async def test_a_group_finished_during_the_suspension_is_kept_for_the_next_rollout(monkeypatch, engine) -> None:
    agent = FakeAgent()
    _hook(monkeypatch, agent, [], engine)
    state = _state()

    async def finished():
        return [SimpleNamespace(response="x", response_length=5, metadata={})]

    done = asyncio.create_task(finished())
    await done
    stats = await inference_rollout_train.suspend(state, {done}, rollout_id=0)
    assert stats["suspended_groups"] == 1 and stats["suspended_done_groups"] == 1
    adopted, started = await inference_rollout_train.resume(state, rollout_id=1)
    assert adopted == {done} and started == {done: 0}
    assert state.args.rollout_resume_stats["resumed_done_groups"] == 1


def test_carried_samples_are_stamped_with_their_start_rollout() -> None:
    sample = SimpleNamespace(metadata={})
    nested = SimpleNamespace(metadata={"start_rollout_id": 1})
    inference_rollout_train._stamp_start_rollout([sample, [nested]], 2)
    assert sample.metadata["start_rollout_id"] == 2
    assert nested.metadata["start_rollout_id"] == 1  # an earlier stamp wins


async def test_samples_not_started_wait_for_the_next_rollout_instead_of_aborting(monkeypatch) -> None:
    started: list[int] = []

    async def fake_generate(input):
        started.append(input.sample.index)
        input.sample.status = inference_rollout_common.Sample.Status.COMPLETED
        input.sample.reward = 1.0
        return SimpleNamespace(samples=input.sample)

    state = SimpleNamespace(
        args=Namespace(partial_rollout=False, group_rm=False),
        aborted=False,
        generate_fn_semaphore=asyncio.Semaphore(4),
        generate_function=fake_generate,
    )
    state.hold_new_samples = lambda: inference_rollout_common.GenerateState.hold_new_samples(state)
    state.release_new_samples = lambda: inference_rollout_common.GenerateState.release_new_samples(state)
    monkeypatch.setattr(inference_rollout_common, "load_generate_function", lambda _p: None)
    monkeypatch.setattr(inference_rollout_common, "TrajectoryLifecycle", lambda: SimpleNamespace(sink=None))
    sample = inference_rollout_common.Sample(index=5, prompt="p")
    state.hold_new_samples()
    task = asyncio.create_task(inference_rollout_common.generate_and_rm(state, sample, {}))
    for _ in range(5):
        await asyncio.sleep(0)
    assert started == [] and not task.done()  # held, not ABORTED
    state.release_new_samples()
    out = await asyncio.wait_for(task, 1.0)
    assert started == [5] and out.status == inference_rollout_common.Sample.Status.COMPLETED


async def test_next_rollout_adopts_suspended_groups_tops_up_and_stamps_them(monkeypatch, engine) -> None:
    """Two rollouts end to end through generate_rollout_async (engine/agent faked):
    rollout 0 suspends its unfinished groups; rollout 1 adopts them, tops the
    in-flight set up to the over-sampling size with new groups, and the carried
    group's samples carry start_rollout_id 0."""
    from miles.rollout.filter_hub.base_types import FilterOutput

    calls: list[str] = []

    async def fake_hook(_args, name: str):
        calls.append(name)

    gates: dict[int, asyncio.Event] = {}
    submitted: list[int] = []

    def fake_submit(state, samples, sample_done_callback=None):
        tasks = []
        for group in samples:
            gi = group[0].group_index
            submitted.append(gi)
            gates[gi] = asyncio.Event()

            async def run(group=group, gi=gi):
                await gates[gi].wait()
                for s in group:
                    s.status = s.Status.COMPLETED
                    s.reward = 1.0
                return group

            tasks.append(asyncio.create_task(run()))
        return tasks

    counter = iter(range(100))

    def data_source(n):
        out = []
        for _ in range(n):
            gi = next(counter)
            out.append([inference_rollout_common.Sample(group_index=gi, index=gi * 10 + i, prompt="p",
                                                        metadata={}) for i in range(2)])
        return out

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(inference_rollout_train, "call_agent_hook", fake_hook)
    monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", fake_submit)
    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", noop)
    monkeypatch.setattr(inference_rollout_train, "recompute_samples_rollout_logprobs_via_prefill", noop)
    monkeypatch.setattr(inference_rollout_train, "apply_preput_filters",
                        lambda *_a, **_k: FilterOutput(keep=True, reason=None))
    monkeypatch.setattr(inference_rollout_train, "load_function", lambda _p: None)
    args = Namespace(
        rollout_global_dataset=True, dynamic_sampling_filter_path=None, rollout_batch_size=2,
        over_sampling_batch_size=4, n_samples_per_prompt=2, agentic_suspend_between_turns=True,
        agentic_suspend_max_rounds=1, partial_rollout=False, sglang_router_ip="r", sglang_router_port=1,
        use_miles_router=True, rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
        rollout_submission_granularity=None, reward_key=None, eval_reward_key=None,
    )
    state = SimpleNamespace(args=args, aborted=False, reset=lambda: None, sampling_params={})

    async def release(*gis):
        await asyncio.sleep(0.01)
        for gi in gis:
            gates[gi].set()

    # rollout 0: 4 groups submitted, 0 and 1 finish -> 2 and 3 are suspended
    releaser = asyncio.create_task(release(0, 1))
    out0, aborted0 = await inference_rollout_train.generate_rollout_async(state, 0, data_source)
    await releaser
    assert [g[0].group_index for g in out0.samples] == [0, 1] and aborted0 == []
    assert submitted == [0, 1, 2, 3] and set(inference_rollout_train._suspended(state).values()) == {0}
    assert args.rollout_suspend_stats["suspended_groups"] == 2

    # a group that finished after the target was reached is kept, not dropped
    extra_state = SimpleNamespace(args=Namespace(**vars(args)), aborted=False, reset=lambda: None, sampling_params={})
    submitted.clear()
    releaser = asyncio.create_task(release(100, 101, 102))
    counter_before = counter
    out_x, _ = await inference_rollout_train.generate_rollout_async(
        extra_state, 0, lambda n: [[inference_rollout_common.Sample(group_index=100 + i, index=1000 + 10 * i + j,
                                                                    prompt="p", metadata={}) for j in range(2)]
                                   for i in range(n)])
    await releaser
    kept = inference_rollout_train._suspended(extra_state)
    assert len(out_x.samples) == 2 and len(kept) == 2
    assert sum(1 for t in kept if t.done()) == extra_state.args.rollout_suspend_stats["suspended_done_groups"]
    for task in list(kept):
        task.cancel()
    submitted[:] = [0, 1, 2, 3]
    assert counter is counter_before

    # rollout 1: adopts 2 and 3, tops up with 4 and 5; 3 (carried) and 4 finish first
    releaser = asyncio.create_task(release(3, 4))
    out1, _ = await inference_rollout_train.generate_rollout_async(state, 1, data_source)
    await releaser
    assert submitted == [0, 1, 2, 3, 4, 5]
    assert [g[0].group_index for g in out1.samples] == [3, 4]
    carried = next(g for g in out1.samples if g[0].group_index == 3)
    fresh = next(g for g in out1.samples if g[0].group_index == 4)
    assert all(s.metadata["start_rollout_id"] == 0 for s in carried)
    assert all("start_rollout_id" not in s.metadata for s in fresh)
    assert args.rollout_resume_stats == {"resumed_groups": 2, "resumed_done_groups": 0}
    # cut-off of rollout 1: group 2 (started 0) would be 2 rounds old -> cancelled
    assert args.rollout_suspend_stats["over_age_cancelled_groups"] == 1
    assert set(inference_rollout_train._suspended(state).values()) == {1}  # group 5 kept
    assert calls == ["resume", "suspend"] * 3
    for task in list(inference_rollout_train._suspended(state)):
        task.cancel()


async def test_suspend_aborts_again_until_the_engines_are_idle(monkeypatch) -> None:
    loads = [2, 1, 0]
    posted: list[str] = []

    async def fake_get(url: str):
        if url.endswith("/get_load"):
            return [{"num_reqs": loads.pop(0), "num_waiting_reqs": 0}]
        return {"urls": ["http://w"]}

    async def fake_post(url: str, payload: dict) -> None:
        posted.append(url)

    async def fake_hook(_args, _name):
        return None

    monkeypatch.setattr(inference_rollout_train, "get", fake_get)
    monkeypatch.setattr(inference_rollout_train, "post", fake_post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_hook", fake_hook)
    monkeypatch.setattr(inference_rollout_train, "SUSPEND_ENGINE_ABORT_INTERVAL_S", 0.0)
    stats = await inference_rollout_train.suspend(_state(), set(), rollout_id=0)
    # 3 initial aborts + one more per busy load reading
    assert len(posted) == inference_rollout_train.SUSPEND_ENGINE_ABORT_REPEATS + 2
    assert stats["engines_idle_confirmed"] == 1 and loads == []


async def test_the_last_rollout_aborts_instead_of_suspending(monkeypatch, engine) -> None:
    called: list[str] = []

    async def fake_hook(_args, name):
        called.append(name)

    async def fake_abort(state, pendings, rollout_id):
        called.append(f"abort:{len(pendings)}")
        return []

    async def noop(*_a, **_k):
        return None

    from miles.rollout.filter_hub.base_types import FilterOutput

    gates = []

    def fake_submit(state, samples, sample_done_callback=None):
        tasks = []
        for i, group in enumerate(samples):
            ev = asyncio.Event()
            gates.append(ev)
            if len(gates) <= 2:
                ev.set()

            async def run(group=group, ev=ev):
                await ev.wait()
                for s in group:
                    s.status = s.Status.COMPLETED
                    s.reward = 1.0
                return group

            tasks.append(asyncio.create_task(run()))
        return tasks

    monkeypatch.setattr(inference_rollout_train, "call_agent_hook", fake_hook)
    monkeypatch.setattr(inference_rollout_train, "abort", fake_abort)
    monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", fake_submit)
    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", noop)
    monkeypatch.setattr(inference_rollout_train, "recompute_samples_rollout_logprobs_via_prefill", noop)
    monkeypatch.setattr(inference_rollout_train, "apply_preput_filters",
                        lambda *_a, **_k: FilterOutput(keep=True, reason=None))
    monkeypatch.setattr(inference_rollout_train, "load_function", lambda _p: None)
    args = Namespace(
        rollout_global_dataset=True, dynamic_sampling_filter_path=None, rollout_batch_size=2,
        over_sampling_batch_size=4, n_samples_per_prompt=1, agentic_suspend_between_turns=True,
        agentic_suspend_max_rounds=1, partial_rollout=False, num_rollout=6, sglang_router_ip="r", sglang_router_port=1,
        rollout_submission_granularity=None, reward_key=None, eval_reward_key=None,
        rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
    )
    state = SimpleNamespace(args=args, aborted=False, reset=lambda: None, sampling_params={})
    counter = iter(range(100))
    source = lambda n: [[inference_rollout_common.Sample(group_index=(g := next(counter)), index=g, prompt="p",
                                                         metadata={})] for _ in range(n)]
    await inference_rollout_train.generate_rollout_async(state, 5, source)
    assert called == ["resume", "abort:2"] and inference_rollout_train._suspended(state) == {}
    for ev in gates:
        ev.set()
