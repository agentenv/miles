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

    async def fake_get(url: str) -> dict:
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
