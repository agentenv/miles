import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput, stamp_kv_cache_namespace
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout import rollout_runtime_stats
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils import dumper_utils
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.function_registry import load_function
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.misc import call_agent_abort_hook, call_agent_hook
from miles.utils.output_store.dropped import record_dropped_samples
from miles.utils.retry_utils import NonRetryableError
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True

    urls = await get_worker_urls(args)
    logger.info(f"Abort request for {urls}")
    await _abort_engines(args, urls=urls)

    # Let the agent integration tear down its in-flight trials so they stop hitting
    # SGLang, instead of running on until their own max_seq_len / timeout.
    await call_agent_abort_hook(args)

    # make sure all the pending tasks are finished
    aborted_samples = []
    # Without partial rollout the drained groups are discarded; tally what was thrown away (groups, samples,
    # generated response tokens) for the all-samples hook. A drained task that raised counts toward groups only
    # (its tokens are unknown) and does not fail the rollout: its group is discarded either way.
    discard = {"groups": 0, "samples": 0, "response_tokens": 0, "unknown_groups": 0}
    for task in asyncio.as_completed(pendings):
        if not args.partial_rollout:
            discard["groups"] += 1
            try:
                group = await task
            except Exception as exc:  # a failed surplus group is still a discarded group
                discard["unknown_groups"] += 1
                logger.warning(f"Discarded aborted group raised {type(exc).__name__}: {exc}")
                continue
            flat = _flatten(group)
            discard["samples"] += len(flat)
            discard["response_tokens"] += sum(sample.response_length for sample in flat)
            record_dropped_samples(group)
            SampleOwnershipRecorder.log_dropped_samples(args=args, samples=flat, reason="aborted")
            continue
        group = await task

        # for partial rollout, collect the partial samples into the data buffer
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    if args.partial_rollout:
        logger.info(f"Collected {sum(len(x) for x in aborted_samples)} partial samples into the data buffer")
    else:
        rollout_runtime_stats.publish("abort_discard", discard)
        logger.info(f"Discarded aborted groups: {discard}")

    return aborted_samples


async def _abort_engines(args: Namespace, *, urls: list[str] | None = None) -> None:
    urls = await get_worker_urls(args) if urls is None else urls
    results = await asyncio.gather(
        *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls], return_exceptions=True
    )
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")


def _flatten(group) -> list[Sample]:
    return [sample for item in group for sample in (item if isinstance(item, list) else [item])]


SUSPEND_ENGINE_ABORT_REPEATS = 3
SUSPEND_ENGINE_ABORT_INTERVAL_S = 0.5
SUSPEND_IDLE_WAIT_S = 30.0


async def _engine_requests(args: Namespace) -> int | None:
    """Requests running or queued on every engine (SGLang /get_load); None when unknown."""
    try:
        total = 0
        for url in await get_worker_urls(args):
            entries = await get(f"{url}/get_load")
            entries = entries if isinstance(entries, list) else [entries]
            total += sum(int(e.get("num_reqs", 0)) for e in entries if isinstance(e, dict))
        return total
    except Exception as exc:  # unknown load is reported, not fatal
        logger.warning(f"suspend: engine load unknown: {exc!r}")
        return None


async def _wait_engines_idle(args: Namespace) -> bool:
    """After the aborts, wait until no request runs on any engine (a request that passed the gate just before
    it closed is aborted again), so the engines can be offloaded for training. False when not confirmed within
    SUSPEND_IDLE_WAIT_S."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SUSPEND_IDLE_WAIT_S
    while True:
        load = await _engine_requests(args)
        if load == 0:
            return True
        if loop.time() >= deadline:
            logger.warning(f"suspend: engines not idle after {SUSPEND_IDLE_WAIT_S}s (load={load})")
            return False
        await _abort_engines(args)
        await asyncio.sleep(SUSPEND_ENGINE_ABORT_INTERVAL_S)


async def suspend(state: GenerateState, pendings: set, rollout_id: int) -> dict[str, int]:
    """Cut-off under --agentic-suspend-between-turns: keep the unfinished groups for the next rollout.

    1. The agents' ``suspend`` hooks close their model-turn gates: a running tool call is not interrupted; when
       it returns, its result is written into the conversation and the next model request waits at the gate.
    2. The engines abort the model turns still in flight (repeated to catch a request that passed the gate just
       before it closed) until they are idle; the agent retries such a turn after ``resume``.
    3. The pending group tasks are kept with the rollout that started them; a group that could no longer be
       trained within ``--agentic-suspend-max-rounds`` rollouts is cancelled (the agent releases its environment),
       recorded as dropped and counted.
    """
    args = state.args
    suspended = state.suspended_groups
    state.hold_new_samples()  # samples not started yet wait for the next rollout
    await call_agent_hook(args, "suspend")
    for attempt in range(SUSPEND_ENGINE_ABORT_REPEATS):
        if attempt:
            await asyncio.sleep(SUSPEND_ENGINE_ABORT_INTERVAL_S)
        await _abort_engines(args)
    quiet = await _wait_engines_idle(args)
    stats = {
        "suspended_groups": 0,
        "suspended_done_groups": 0,
        "over_age_cancelled_groups": 0,
        "over_age_cancelled_samples": 0,
        "over_age_cancelled_tokens": 0,
        "over_age_unknown_groups": 0,
    }
    to_cancel = []
    for task in pendings:
        started = suspended.get(task, rollout_id)
        if (rollout_id + 1) - started > args.agentic_suspend_max_rounds:
            to_cancel.append(task)
            suspended.pop(task, None)
            continue
        suspended[task] = started
        stats["suspended_groups"] += 1
        if task.done():
            stats["suspended_done_groups"] += 1
    for task in to_cancel:
        stats["over_age_cancelled_groups"] += 1
        if task.done() and not task.cancelled() and task.exception() is None:
            flat = _flatten(task.result())
            stats["over_age_cancelled_samples"] += len(flat)
            stats["over_age_cancelled_tokens"] += sum(sample.response_length for sample in flat)
            record_dropped_samples(task.result())
            SampleOwnershipRecorder.log_dropped_samples(args=args, samples=flat, reason="agentic_suspend_over_age")
            continue
        task.cancel()
        stats["over_age_unknown_groups"] += 1
    if to_cancel:
        await asyncio.gather(*to_cancel, return_exceptions=True)
    stats["engines_idle_confirmed"] = int(quiet)
    rollout_runtime_stats.publish("suspend", stats)
    logger.info(f"Suspended agentic groups at rollout {rollout_id}: {stats}")
    return stats


async def resume(state: GenerateState) -> tuple[set, dict]:
    """Start of a rollout under --agentic-suspend-between-turns: reopen the agents' gates and adopt the
    suspended group tasks as pending."""
    await call_agent_hook(state.args, "resume")
    state.release_new_samples()
    adopted = set(state.suspended_groups)
    rollout_runtime_stats.publish(
        "resume",
        {"resumed_groups": len(adopted), "resumed_done_groups": sum(1 for task in adopted if task.done())},
    )
    return adopted, dict(state.suspended_groups)


def _stamp_start_rollout(group, started: int) -> None:
    for sample in _flatten(group):
        if "start_rollout_id" not in sample.metadata:
            sample.metadata["start_rollout_id"] = started


async def get_worker_urls(args: Namespace):
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
    return router_worker_base_urls([worker["url"] for worker in response["workers"]])


def submit_generate_tasks(
    state: GenerateState,
    samples: list[list[Sample]],
    sample_done_callback: Callable[[], None] | None = None,
):
    return [
        asyncio.create_task(
            # submit a group of samples as a single task.
            generate_and_rm_group(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
                sample_done_callback=sample_done_callback,
            )
        )
        for group in samples
    ]


async def generate_rollout_async(
    state: GenerateState,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    *,
    kv_cache_namespace: str | None = None,
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    # default to group level submission for sync/one-step async rollout
    scheduler = make_submission_scheduler(args, default="group")

    pendings = set()
    suspend_mode = args.agentic_suspend_between_turns
    started_at: dict[asyncio.Task, int] = {}
    if suspend_mode:
        pendings, started_at = await resume(state)
        metric_gatherer.on_groups_resumed(len(pendings))
        # Over-sampling keeps --over-sampling-batch-size groups in flight: top the resumed groups up with new ones
        # (the rule below only submits while pending < remaining budget, which resumed groups would block).
        top_up = args.over_sampling_batch_size - len(pendings)
        if pendings and top_up > 0:
            samples = data_source(top_up)
            stamp_kv_cache_namespace(samples, namespace=kv_cache_namespace)
            metric_gatherer.on_submit(len(samples))
            scheduler.on_submit(samples)
            pendings.update(submit_generate_tasks(state, samples, scheduler.sample_done_callback))
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while scheduler.has_capacity(pending_groups=len(pendings), group_budget=target_data_size - len(data)):
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            stamp_kv_cache_namespace(samples, namespace=kv_cache_namespace)
            metric_gatherer.on_submit(len(samples))
            scheduler.on_submit(samples)
            pendings.update(submit_generate_tasks(state, samples, scheduler.sample_done_callback))

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await scheduler.wait_for_progress(pendings)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            if suspend_mode and len(data) >= target_data_size:
                # enough groups already: keep this finished one for the next rollout (it is suspended like the
                # unfinished ones) instead of dropping it
                pendings.add(task)
                continue
            try:
                group: list[Sample] = task.result()
            except NonRetryableError:
                # A broken contract fails every later group the same way; stop instead of spinning.
                raise
            except Exception as e:
                if suspend_mode:
                    started_at.pop(task, None)
                    state.suspended_groups.pop(task, None)
                metric_gatherer.on_group_failed()
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            started = started_at.pop(task, None) if suspend_mode else None
            if started is not None:
                state.suspended_groups.pop(task, None)
                if started < rollout_id:
                    _stamp_start_rollout(group, started)

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            metric_gatherer.on_group_completed()
            metric_gatherer.on_group_before_dynamic_filter(args, group)
            filter_output = apply_preput_filters(args, dynamic_filter, group)
            if not filter_output.keep:
                metric_gatherer.on_group_filtered()
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                record_dropped_samples(group)
                SampleOwnershipRecorder.log_dropped_groups(args=args, before=group, after=[], reason="dynamic_filter")
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                metric_gatherer.on_group_kept()
                pbar.update(args.n_samples_per_prompt)
            else:
                metric_gatherer.on_group_surplus()
                record_dropped_samples(group)
                SampleOwnershipRecorder.log_dropped_groups(args=args, before=group, after=[], reason="oversampling")

    pbar.close()
    metric_gatherer.on_rollout_end(len(pendings))
    if (over_sampling := metric_gatherer.over_sampling_stats()) is not None:
        rollout_runtime_stats.publish("over_sampling", over_sampling)
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
        sample_text_preview(sample),
        str(sample.label)[:100],
        reward_log_summary(sample.reward),
    )

    # there are still some unfinished requests: suspend them between model turns for the next rollout, or abort
    # them. The last rollout of the run suspends nothing (no later rollout would continue it); the agents' abort
    # hooks tear those trajectories down.
    last_rollout = args.num_rollout is not None and args.num_rollout > 0 and rollout_id >= args.num_rollout - 1
    if suspend_mode and not last_rollout:
        await suspend(state, pendings, rollout_id)
        aborted_samples = []
    else:
        aborted_samples = await abort(state, pendings, rollout_id)
        if suspend_mode:
            state.suspended_groups.clear()

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    before_filter = SampleOwnershipRecorder.flatten_samples(data)
    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    kept = {id(sample) for sample in SampleOwnershipRecorder.flatten_samples(data)}
    record_dropped_samples(sample for sample in before_filter if id(sample) not in kept)
    SampleOwnershipRecorder.log_dropped_groups(
        args=args, before=before_filter, after=data, reason="rollout_sample_filter"
    )
    # There can be circumstances where users want to process all samples including filtered ones.
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
