import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils import dumper_utils
from miles.utils.function_registry import load_function
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.misc import call_agent_abort_hook, call_agent_hook
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True

    urls = await get_worker_urls(args)
    logger.info(f"Abort request for {urls}")
    results = await asyncio.gather(
        *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls], return_exceptions=True
    )
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # Let the agent integration tear down its in-flight trials so they stop hitting
    # SGLang, instead of running on until their own max_seq_len / timeout.
    await call_agent_abort_hook(args)

    # make sure all the pending tasks are finished
    aborted_samples = []
    # Without partial rollout the drained groups are discarded; keep a tally of
    # what was thrown away (groups, samples, generated response tokens) on
    # ``args.rollout_abort_discard_stats`` for the all-samples hook (same keys as
    # sglang_rollout.abort). A drained task that raised counts toward groups only
    # (its tokens are unknown) and no longer fails the rollout: its group is
    # discarded either way.
    discard = {"groups": 0, "samples": 0, "response_tokens": 0, "unknown_groups": 0}
    for task in asyncio.as_completed(pendings):
        if not args.partial_rollout:
            discard["groups"] += 1
            try:
                group = await task
                flat = [s for item in group for s in (item if isinstance(item, list) else [item])]
                discard["samples"] += len(flat)
                discard["response_tokens"] += sum(int(getattr(s, "response_length", 0) or 0) for s in flat)
            except BaseException as exc:  # noqa: BLE001 - a failed surplus group is still discarded
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # the rollout itself is being cancelled
                discard["unknown_groups"] += 1
                logger.warning(f"Discarded aborted group raised {type(exc).__name__}: {exc}")
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
        args.rollout_abort_discard_stats = discard
        logger.info(f"Discarded aborted groups: {discard}")

    return aborted_samples


SUSPEND_ENGINE_ABORT_REPEATS = 3
SUSPEND_ENGINE_ABORT_INTERVAL_S = 0.5


def _suspended(state: GenerateState) -> dict:
    """Groups suspended between model turns (--agentic-suspend-between-turns):
    ``{task: start_rollout_id}``; lives on the persistent GenerateState so the
    tasks (running on Miles' background event loop) cross the rollout boundary."""
    suspended = getattr(state, "suspended_groups", None)
    if suspended is None:
        suspended = {}
        state.suspended_groups = suspended
    return suspended


async def _abort_engines(args: Namespace) -> None:
    urls = await get_worker_urls(args)
    results = await asyncio.gather(
        *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls], return_exceptions=True
    )
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")


def _group_stats(group) -> tuple[int, int]:
    flat = [s for item in group for s in (item if isinstance(item, list) else [item])]
    return len(flat), sum(int(getattr(s, "response_length", 0) or 0) for s in flat)


async def suspend(state: GenerateState, pendings: set, rollout_id: int) -> dict:
    """Cut-off under --agentic-suspend-between-turns: keep the unfinished groups.

    1. The agent's ``suspend`` hook closes its model-turn gate: a tool call that
       is running is NOT interrupted; when it returns, its result is written
       into the conversation and the next model request waits at the gate.
    2. The engines abort the model turns still in flight (repeated a few times
       to catch a request that passed the gate just before it closed); the agent
       retries such a turn after ``resume`` (an aborted turn is not recorded).
    3. The pending group tasks are kept (not awaited) with the rollout that
       started them; a group that could no longer be trained within
       ``--agentic-suspend-max-rounds`` rollouts is cancelled (the agent
       releases its environment) and counted.

    Returns the stats also stored on ``args.rollout_suspend_stats``."""
    args = state.args
    suspended = _suspended(state)
    hold = getattr(state, "hold_new_samples", None)
    if callable(hold):
        hold()  # samples not started yet wait for the next rollout
    await call_agent_hook(args, "suspend")
    for attempt in range(SUSPEND_ENGINE_ABORT_REPEATS):
        if attempt:
            await asyncio.sleep(SUSPEND_ENGINE_ABORT_INTERVAL_S)
        await _abort_engines(args)
    max_rounds = int(getattr(args, "agentic_suspend_max_rounds", 1))
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
        if (rollout_id + 1) - started > max_rounds:
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
            samples, tokens = _group_stats(task.result())
            stats["over_age_cancelled_samples"] += samples
            stats["over_age_cancelled_tokens"] += tokens
            continue
        task.cancel()
        stats["over_age_unknown_groups"] += 1
    if to_cancel:
        await asyncio.gather(*to_cancel, return_exceptions=True)
    args.rollout_suspend_stats = stats
    logger.info(f"Suspended agentic groups at rollout {rollout_id}: {stats}")
    return stats


async def resume(state: GenerateState, rollout_id: int) -> tuple[set, dict]:
    """Start of a rollout under --agentic-suspend-between-turns: reopen the
    agent's gate and adopt the suspended group tasks as pending."""
    suspended = _suspended(state)
    await call_agent_hook(state.args, "resume")
    release = getattr(state, "release_new_samples", None)
    if callable(release):
        release()
    adopted = set(suspended)
    state.args.rollout_resume_stats = {
        "resumed_groups": len(adopted),
        "resumed_done_groups": sum(1 for t in adopted if t.done()),
    }
    return adopted, dict(suspended)


def _stamp_start_rollout(group, started: int) -> None:
    for item in group:
        for sample in item if isinstance(item, list) else [item]:
            metadata = getattr(sample, "metadata", None)
            if isinstance(metadata, dict) and "start_rollout_id" not in metadata:
                metadata["start_rollout_id"] = started


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    return router_worker_base_urls(urls)


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
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
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
    suspend_mode = bool(getattr(args, "agentic_suspend_between_turns", False))
    started_at: dict = {}
    if suspend_mode:
        pendings, started_at = await resume(state, rollout_id)
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while scheduler.has_capacity(pending_groups=len(pendings), group_budget=target_data_size - len(data)):
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            scheduler.on_submit(samples)
            pendings.update(submit_generate_tasks(state, samples, scheduler.sample_done_callback))

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await scheduler.wait_for_progress(pendings)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                if suspend_mode:
                    started_at.pop(task, None)
                    _suspended(state).pop(task, None)
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            started = started_at.pop(task, None) if suspend_mode else None
            if started is not None:
                _suspended(state).pop(task, None)
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
            metric_gatherer.on_group_before_dynamic_filter(args, group)
            filter_output = apply_preput_filters(args, dynamic_filter, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
        sample_text_preview(sample),
        str(sample.label)[:100],
        reward_log_summary(sample.reward),
    )

    # there are still some unfinished requests, abort them
    if suspend_mode:
        # keep them: suspended between model turns, continued next rollout
        await suspend(state, pendings, rollout_id)
        aborted_samples = []
    else:
        aborted_samples = await abort(state, pendings, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
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
