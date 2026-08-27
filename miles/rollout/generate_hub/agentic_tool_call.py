"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach unless the function opts
  into the fail-closed result contract via ``_miles_fail_closed_result=True``.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
"""

import argparse
import asyncio
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.merge import merge_samples_by_compaction_segment
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_AGENT_FAILURE_DIAGNOSTIC_MAX_BYTES = 2048
_AGENT_FAILURE_TRUNCATION_MARKER = b"...[diagnostic truncated]"


def _bounded_agent_failure_diagnostic(error: Exception) -> str:
    value = f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
    if len(value) > _AGENT_FAILURE_DIAGNOSTIC_MAX_BYTES:
        value = (
            value[: _AGENT_FAILURE_DIAGNOSTIC_MAX_BYTES - len(_AGENT_FAILURE_TRUNCATION_MARKER)]
            + _AGENT_FAILURE_TRUNCATION_MARKER
        )
    return value.decode("utf-8", errors="ignore")


def _aborted_agent_failure(input: GenerateFnInput, error: Exception, *, records_collected: int) -> GenerateFnOutput:
    sample = deepcopy(input.sample)
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "agent_failure": {
            "schema": "miles.agent-failure.v1",
            "stage": "custom_agent_function",
            "error_type": type(error).__name__[:128],
            "diagnostic": _bounded_agent_failure_diagnostic(error),
            "records_collected": records_collected,
        },
    }
    return GenerateFnOutput(samples=sample)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert getattr(input.args, "session_server_ip", None) and getattr(input.args, "session_server_ports", None), (
        "agentic_tool_call.generate requires session_server_ip/session_server_ports. "
        "Pass --use-session-server to start the session server."
    )
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    metadata = {**metadata, "session_server_id": tracer.session_server_id}

    agent_metadata = None
    agent_failure: Exception | None = None
    collect_timed_out = False
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            base_url=tracer.base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        if agent_metadata is None and getattr(custom_agent_function, "_miles_fail_closed_result", False) is True:
            raise RuntimeError("fail-closed custom agent returned null without a trustworthy outcome")
        if agent_metadata is not None and not isinstance(agent_metadata, dict):
            raise TypeError("custom agent result must be an object or null")
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except Exception as e:
        agent_failure = e
        logger.warning(
            "%s Agent function failed: %s",
            log_prefix,
            _bounded_agent_failure_diagnostic(e),
        )

    finally:
        # Collect even if the agent failed.
        logger.debug(f"{log_prefix} Calling collect_samples...")
        try:
            result = await tracer.collect_samples(
                input.sample,
                max_seq_len=max_seq_len,
            )
        except asyncio.TimeoutError:
            collect_timed_out = True
            logger.warning(f"{log_prefix} Timed out collecting samples", exc_info=True)
        else:
            logger.debug(
                f"{log_prefix} collect_samples done: {len(result.samples)} samples, "
                f"total_time={time.monotonic()-t_start:.1f}s"
            )

    if collect_timed_out:
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    session_metadata = dict(result.session_metadata)
    records_collected = session_metadata.pop("records_collected", 0)
    if agent_failure is not None:
        return _aborted_agent_failure(
            input,
            agent_failure,
            records_collected=records_collected,
        )

    if not result.samples:
        if result.empty_reason == "all_truncated":
            logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        else:
            logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    samples = result.samples
    for s in samples:
        s.metadata.update(agent_metadata or {})

    compaction_enabled = session_metadata.get("compaction_schema_version") == 1
    if compaction_enabled:
        for s in samples:
            s.metadata["compaction_trajectory_id"] = tracer.session_id

    # If the agent function reports wall-clock time spent outside policy generation
    # (env/tool steps), surface it on Sample.non_generation_time so throughput
    # accounting subtracts it.
    ngt = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if ngt is not None:
        for s in samples:
            s.non_generation_time = ngt

    if compaction_enabled:
        samples = merge_samples_by_compaction_segment(samples, input.state.tokenizer)
        samples[-1].metadata.update(session_metadata)
        return GenerateFnOutput(samples=samples)

    (sample,) = samples
    sample.metadata.update(session_metadata)
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) "
        "per session. Truncation happens inside the session server during sample assembly; "
        "also forwarded to the Harbor agent server (as max_seq_len) to abort the trial early.",
    )


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
