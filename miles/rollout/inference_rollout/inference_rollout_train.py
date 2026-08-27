import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils import dumper_utils
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.misc import as_completed_async, call_agent_abort_hook, load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class OnePassRolloutError(RuntimeError):
    """The immutable rollout plan did not complete exactly once."""


def _describe_one_pass_aborted_sample(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    identity = [f"{key}={metadata[key]!r}" for key in ("sample_id", "task_id") if metadata.get(key) is not None]
    if sample.index is not None:
        identity.append(f"index={sample.index!r}")

    failure = metadata.get("agent_failure")
    cause = None
    if isinstance(failure, dict):
        cause = failure.get("diagnostic") or failure.get("error_type")
    if cause is None:
        cause = "unavailable"
    else:
        cause = " ".join(str(cause).split())

    return f"{', '.join(identity) or 'identity=unavailable'}; cause={cause}"


def _validate_one_pass_planned_groups(
    groups: object, *, expected_groups: int, expected_group_size: int
) -> list[list[Sample]]:
    if not isinstance(groups, list) or len(groups) != expected_groups:
        actual = len(groups) if isinstance(groups, list) else type(groups).__name__
        raise OnePassRolloutError(
            f"one-pass data source returned {actual} groups; expected {expected_groups}"
        )
    for group in groups:
        if (
            not isinstance(group, list)
            or len(group) != expected_group_size
            or not all(isinstance(sample, Sample) for sample in group)
        ):
            raise OnePassRolloutError(
                "one-pass data source returned a short or malformed planned group"
            )
    return groups


def _validate_one_pass_generated_group(
    group: object, *, expected_group_size: int
) -> list[Sample] | list[list[Sample]]:
    if not isinstance(group, list) or len(group) != expected_group_size:
        actual = len(group) if isinstance(group, list) else type(group).__name__
        raise OnePassRolloutError(
            f"one-pass generation returned group size {actual}; expected {expected_group_size}"
        )

    flattened: list[Sample] = []
    for trajectory in group:
        if isinstance(trajectory, Sample):
            flattened.append(trajectory)
        elif (
            isinstance(trajectory, list)
            and trajectory
            and all(isinstance(segment, Sample) for segment in trajectory)
        ):
            flattened.extend(trajectory)
        else:
            raise OnePassRolloutError(
                "one-pass generation returned a short or malformed trajectory"
            )
    aborted = [sample for sample in flattened if sample.status == Sample.Status.ABORTED]
    if aborted:
        details = "; ".join(f"[{_describe_one_pass_aborted_sample(sample)}]" for sample in aborted)
        raise OnePassRolloutError(f"one-pass generation returned an aborted trajectory: {details}")
    return group


async def _abort_one_pass_remaining(
    state: GenerateState, tasks: set[asyncio.Task], rollout_id: int
) -> None:
    if not tasks:
        return
    try:
        if not state.aborted:
            await abort(state, tasks, rollout_id)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error(
            "one-pass rollout cleanup failed while aborting remaining work: %r",
            error,
            exc_info=True,
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    args = state.args

    assert not state.aborted
    state.aborted = True

    urls = await get_worker_urls(args)
    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # Let the agent integration tear down its in-flight trials so they stop hitting
    # SGLang, instead of running on until their own max_seq_len / timeout.
    await call_agent_abort_hook(args)

    # make sure all the pending tasks are finished
    aborted_samples = []
    async for group in as_completed_async(pendings):
        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    if args.partial_rollout:
        logger.info(f"Collected {sum(len(x) for x in aborted_samples)} partial samples into the data buffer")

    return aborted_samples


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    return router_worker_base_urls(urls)


def submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]):
    return [
        asyncio.create_task(
            # submit a group of samples as a single task.
            generate_and_rm_group(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in samples
    ]


async def _generate_rollout_one_pass_async(
    state: GenerateState,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    dynamic_filter,
    metric_gatherer: MetricGatherer,
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Execute an immutable rollout plan without replacement or refill."""

    args = state.args
    target_data_size = args.rollout_batch_size
    data: list[list[Sample] | list[list[Sample]]] = []
    pendings: set[asyncio.Task] = set()
    unprocessed_done: set[asyncio.Task] = set()
    do_print = True
    pbar = tqdm(
        total=target_data_size * args.n_samples_per_prompt,
        desc="Rollout generation",
    )

    try:
        planned_groups = _validate_one_pass_planned_groups(
            data_source(target_data_size),
            expected_groups=target_data_size,
            expected_group_size=args.n_samples_per_prompt,
        )
        pendings.update(submit_generate_tasks(state, planned_groups))
        if len(pendings) != target_data_size:
            raise OnePassRolloutError(
                "one-pass rollout did not submit every planned group exactly once"
            )

        while pendings:
            done, pendings = await asyncio.wait(
                pendings, return_when=asyncio.FIRST_COMPLETED
            )
            unprocessed_done = set(done)
            while unprocessed_done:
                task = unprocessed_done.pop()
                try:
                    group = task.result()
                except Exception as error:
                    raise OnePassRolloutError(
                        "one-pass rollout generation task raised an exception"
                    ) from error

                group = _validate_one_pass_generated_group(
                    group,
                    expected_group_size=args.n_samples_per_prompt,
                )
                if do_print:
                    sample = group[0][0] if isinstance(group[0], list) else group[0]
                    logger.info(
                        f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                    )
                    do_print = False

                dynamic_filter_output = call_dynamic_filter(
                    dynamic_filter, args, group
                )
                if not dynamic_filter_output.keep:
                    metric_gatherer.on_dynamic_filter_drop(
                        reason=dynamic_filter_output.reason
                    )
                    raise OnePassRolloutError(
                        "one-pass rollout dynamic filter dropped a planned group"
                    )
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
    except BaseException:
        await _abort_one_pass_remaining(
            state,
            pendings | unprocessed_done,
            rollout_id,
        )
        raise
    finally:
        pbar.close()

    if len(data) != target_data_size:
        raise OnePassRolloutError(
            f"one-pass rollout completed {len(data)} groups; expected {target_data_size}"
        )

    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    data = sorted(
        data,
        key=lambda group: (
            group[0][0].index if isinstance(group[0], list) else group[0].index
        ),
    )
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
        if len(data) != target_data_size:
            raise OnePassRolloutError(
                "one-pass rollout sample filter returned a short result"
            )
    # Argument validation forbids rollout_all_samples_process_path because that
    # callback receives the data source and could refill the immutable plan.
    if args.rollout_all_samples_process_path is not None:
        raise OnePassRolloutError(
            "one-pass rollout forbids all-samples data-source callbacks"
        )

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(
        samples=data, metrics=metric_gatherer.collect()
    ), []


async def generate_rollout_async(
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)

    metric_gatherer = MetricGatherer()

    if getattr(args, "rollout_one_pass_no_replacement", False):
        return await _generate_rollout_one_pass_async(
            state,
            rollout_id,
            data_source,
            dynamic_filter,
            metric_gatherer,
        )

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    pendings = set()
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            pendings.update(submit_generate_tasks(state, samples))

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
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
