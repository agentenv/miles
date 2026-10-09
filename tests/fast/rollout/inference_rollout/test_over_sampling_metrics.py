"""Over-sampling tally in the rollout metrics (yeto rl-algo-supplement 2.4):
submit batch sizes, refill calls, filtered / kept / surplus groups, groups in flight at the end."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

from miles.rollout.filter_hub.base_types import FilterOutput, MetricGatherer
from miles.rollout.inference_rollout import inference_rollout_common, inference_rollout_train

P = "rollout/over_sampling/"


def test_gatherer_reports_nothing_without_a_submission():
    assert not any(k.startswith(P) for k in MetricGatherer().collect())
    assert MetricGatherer().over_sampling_stats() is None


def test_gatherer_counts():
    g = MetricGatherer()
    g.on_submit(4)
    g.on_submit(4)
    g.on_submit(2)
    for _ in range(7):
        g.on_group_completed()
    for _ in range(3):
        g.on_group_filtered()
    for _ in range(3):
        g.on_group_kept()
    g.on_group_surplus()
    g.on_rollout_end(3)
    m = g.collect()
    assert m[P + "submit_calls"] == 3 and m[P + "refill_calls"] == 2
    assert m[P + "submitted_groups"] == 10
    assert m[P + "submit_batch_size_first"] == 4 and m[P + "submit_batch_size_max"] == 4
    assert m[P + "completed_groups"] == 7 and m[P + "filtered_groups"] == 3
    assert m[P + "kept_groups"] == 3 and m[P + "surplus_groups"] == 1
    assert m[P + "inflight_groups_at_end"] == 3
    assert m[P + "resumed_groups"] == 0 and m[P + "failed_groups"] == 0


def test_gatherer_reports_resumed_only_rollout():
    g = MetricGatherer()
    g.on_groups_resumed(4)
    g.on_group_failed()
    m = g.collect()
    assert m[P + "submit_calls"] == 0 and m[P + "refill_calls"] == 0 and m[P + "submitted_groups"] == 0
    assert m[P + "submit_batch_size_first"] == 0 and m[P + "submit_batch_size_max"] == 0
    assert m[P + "resumed_groups"] == 4 and m[P + "failed_groups"] == 1


async def test_generate_rollout_async_reports_the_over_sampling_tally(monkeypatch):
    aborted: list[int] = []

    async def fake_abort(state, pendings, rollout_id):
        aborted.append(len(pendings))
        for t in pendings:
            t.cancel()
        return []

    async def noop(*_a, **_k):
        return None

    def fake_submit(state, samples, sample_done_callback=None):
        async def run(group):
            # filtered groups finish first; kept groups 0 and 3 a bit later; group 6+ never in time
            idx = group[0].group_index
            await asyncio.sleep(0 if idx % 3 else 0.01 if idx < 6 else 10)
            if idx == 1:
                raise RuntimeError("generation failed")
            for s in group:
                s.status = s.Status.COMPLETED
                s.reward = 1.0
            return group

        return [asyncio.create_task(run(group)) for group in samples]

    def keep_even(args, _filter, group):
        idx = group[0].group_index
        return FilterOutput(keep=idx % 3 == 0, reason=None if idx % 3 == 0 else "zero_std")

    monkeypatch.setattr(inference_rollout_train, "abort", fake_abort)
    monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", fake_submit)
    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", noop)
    monkeypatch.setattr(inference_rollout_train, "recompute_samples_rollout_logprobs_via_prefill", noop)
    monkeypatch.setattr(inference_rollout_train, "apply_preput_filters", keep_even)
    monkeypatch.setattr(inference_rollout_train, "load_function", lambda _p: None)
    args = Namespace(
        rollout_global_dataset=True, dynamic_sampling_filter_path=None, rollout_batch_size=2,
        over_sampling_batch_size=3, n_samples_per_prompt=1, agentic_suspend_between_turns=False,
        partial_rollout=False, num_rollout=6, sglang_router_ip="r", sglang_router_port=1,
        rollout_submission_granularity=None, reward_key=None, eval_reward_key=None,
        rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
    )
    state = SimpleNamespace(args=args, aborted=False, reset=lambda: None, sampling_params={})
    counter = iter(range(100))
    source = lambda n: [[inference_rollout_common.Sample(group_index=(g := next(counter)), index=g, prompt="p",
                                                         metadata={})] for _ in range(n)]
    out, _ = await inference_rollout_train.generate_rollout_async(state, 0, source)
    m = out.metrics
    assert len(out.samples) == 2
    assert m[P + "submit_batch_size_first"] == 3
    assert m[P + "submitted_groups"] == 3 * m[P + "submit_calls"]
    assert m[P + "refill_calls"] == m[P + "submit_calls"] - 1 >= 1
    assert m[P + "kept_groups"] == 2
    assert m[P + "filtered_groups"] == m["rollout/dynamic_filter/drop_zero_std"] >= 1  # group 1 failed, not filtered
    assert m[P + "completed_groups"] == m[P + "filtered_groups"] + m[P + "kept_groups"] + m[P + "surplus_groups"]
    assert m[P + "inflight_groups_at_end"] == aborted[0]
    assert m[P + "failed_groups"] == 1 and m[P + "resumed_groups"] == 0
    assert (m[P + "submitted_groups"] + m[P + "resumed_groups"]
            == m[P + "completed_groups"] + m[P + "failed_groups"] + m[P + "inflight_groups_at_end"])
    # the same tally (short keys) is left on args for the all-samples hook
    assert args.rollout_over_sampling_stats == {k[len(P):]: v for k, v in m.items() if k.startswith(P)}
