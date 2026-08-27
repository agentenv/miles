"""Training-sample assembly: session records -> per-turn `Sample`s, truncated at turn boundaries.

Owned by the session package so the assembly runs on the owning instance (records never have to leave the session server). The wire codec for the assembled reply lives in `codec`.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay decoders (accepted utils-level dependency: the decoders have other consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE `merge_samples` — truncation is a turn-level budget decision (which turns survive; the overflowing turn is cut at a turn boundary, later turns are dropped) and the turn structure only exists pre-merge.
"""

from argparse import Namespace
from copy import deepcopy

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.types import SessionRecord
from miles.utils.lifecycle import attach_lifecycle_metadata
from miles.utils.types import Sample


_TERMINAL_ORPHAN_SUMMARY_EXIT_STATUSES = frozenset({"max_seq_len", "max_turns"})


def compute_samples_from_openai_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    accumulated_token_ids_by_context_window: dict[str | int, list[int]] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Each record carries its own ``prompt_token_ids`` and ``output_token_ids``
    (with logprobs).  We want to reuse those per-turn logprobs directly
    instead of re-decoding, but we must first trim "trailing tokens" — stop
    tokens the model emitted that the chat template also renders as the next
    turn's delimiter — to avoid double-counting.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/session/test_samples.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    compaction_records = [record.compaction_schema_version is not None for record in records]
    if any(compaction_records) and not all(compaction_records):
        raise ValueError("session mixes marked and unmarked compaction records")
    compaction_enabled = bool(records and all(compaction_records))
    if compaction_enabled and accumulated_token_ids_by_context_window is None:
        raise ValueError("compaction session is missing per-window accumulated token IDs")

    per_window_ids = {int(key): value for key, value in (accumulated_token_ids_by_context_window or {}).items()}
    current_window: int | None = None
    current_accumulated = accumulated_token_ids

    for i, record in enumerate(records):
        if compaction_enabled:
            if record.compaction_schema_version != 1:
                raise ValueError("unsupported compaction record schema")
            window = record.compaction_context_window
            if window is None or window not in per_window_ids:
                raise ValueError(f"compaction record references missing context window {window!r}")
            if current_window is None or window != current_window:
                if current_window is not None and cursor != len(current_accumulated):
                    raise AssertionError(
                        f"cursor {cursor} != len(accumulated_token_ids) {len(current_accumulated)} "
                        f"after processing compaction context window {current_window}"
                    )
                current_window = window
                current_accumulated = per_window_ids[window]
                cursor = 0
            next_window = records[i + 1].compaction_context_window if i + 1 < len(records) else None
            is_last = next_window != window
        else:
            is_last = i == len(records) - 1
        prompt_ids = record.request["input_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if current_accumulated is not None:
            # Step 1: position cursor right after this turn's prompt
            cursor = len(prompt_ids)

            # Step 2: greedily match output_ids against accumulated[cursor:]
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(current_accumulated) and output_ids[j] == current_accumulated[idx]:
                    matched += 1
                else:
                    break

            # Step 3: unmatched trailing tokens were consumed by the next
            # turn's template rendering (e.g. stop tokens that double as
            # the next message delimiter) — strip them from the sample.
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={current_accumulated[cursor:cursor+3]}"
            )

            # Step 4: advance cursor past matched output to the next turn
            cursor += matched

        sample = _compute_sample_from_openai_record(args, record, tokenizer, trim_count)
        if compaction_enabled:
            sample.metadata.update(
                {
                    "compaction_schema_version": record.compaction_schema_version,
                    "compaction_context_window": record.compaction_context_window,
                    "compaction_segment_index": record.compaction_segment_index,
                    "compaction_segment_type": record.compaction_segment_type,
                    "compaction_context_budget": record.compaction_context_budget,
                }
            )
        attach_lifecycle_metadata(sample, record, records[i - 1] if i else None, turn=i + 1)
        if is_last and args.save_debug_trajectory_data is not None:
            sample.metadata["messages"] = record.request["messages"] + [record.response["choices"][0]["message"]]
        samples.append(sample)

    if current_accumulated is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(current_accumulated), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(current_accumulated)} "
            f"after processing all {len(records)} records"
        )

    return samples


def merge_samples_by_compaction_segment(samples: list[Sample], tokenizer) -> list[Sample]:
    """Merge turns inside each CompactionRL segment, never across a reset."""
    if not samples:
        raise ValueError("cannot merge an empty compaction trajectory")

    groups: list[list[Sample]] = []
    context_budget: int | None = None
    for sample in samples:
        metadata = sample.metadata or {}
        if metadata.get("compaction_schema_version") != 1:
            raise ValueError("compaction sample is missing schema version 1")
        index = metadata.get("compaction_segment_index")
        segment_type = metadata.get("compaction_segment_type")
        sample_context_budget = metadata.get("compaction_context_budget")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("compaction sample has invalid segment index")
        if segment_type not in {"execution", "summary"}:
            raise ValueError("compaction sample has invalid segment type")
        if (
            isinstance(sample_context_budget, bool)
            or not isinstance(sample_context_budget, int)
            or sample_context_budget <= 0
        ):
            raise ValueError("compaction sample has invalid context budget")
        if context_budget is None:
            context_budget = sample_context_budget
        elif sample_context_budget != context_budget:
            raise ValueError("compaction context budget changed within a trajectory")
        if not groups or groups[-1][0].metadata["compaction_segment_index"] != index:
            if index != len(groups):
                raise ValueError("compaction segment indices must be contiguous from zero")
            expected_type = "execution" if index % 2 == 0 else "summary"
            if segment_type != expected_type:
                raise ValueError(f"compaction segment {index} must be {expected_type}, got {segment_type}")
            groups.append([])
        elif groups[-1][0].metadata["compaction_segment_type"] != segment_type:
            raise ValueError("compaction segment type changed within a segment")
        groups[-1].append(sample)

    if any(group[0].metadata["compaction_segment_type"] == "summary" and len(group) != 1 for group in groups):
        raise ValueError("each compaction summary must be exactly one policy response")

    terminal_orphan: Sample | None = None
    if groups[-1][0].metadata["compaction_segment_type"] == "summary":
        # In CompactionRL, a summary is the bridge into a reconstructed execution
        # context.  If a declared resource boundary fires before that context emits
        # any policy tokens, the summary did not affect the terminal outcome.  Do
        # not give that orphan summary non-causal credit, but retain the already
        # completed execution prefix.  Outcome authentication remains downstream;
        # all other terminal-summary shapes remain a hard error here so malformed
        # trajectories cannot be hidden.
        exit_statuses = [(sample.metadata or {}).get("exit_status") for sample in samples]
        exit_status = exit_statuses[0]
        if any(value != exit_status for value in exit_statuses[1:]):
            raise ValueError("terminal compaction summary has inconsistent episode exit statuses")
        terminal_orphan = groups[-1][0]
        if not isinstance(exit_status, str) or exit_status not in _TERMINAL_ORPHAN_SUMMARY_EXIT_STATUSES:
            raise ValueError(
                "compaction trajectory ended in a summary segment without a " "declared max_seq_len/max_turns boundary"
            )
        if terminal_orphan.status not in {
            Sample.Status.COMPLETED,
            Sample.Status.TRUNCATED,
        }:
            raise ValueError("terminal compaction summary has an untrainable status")
        groups.pop()
        if not groups or groups[-1][0].metadata["compaction_segment_type"] != "execution":
            raise ValueError("terminal compaction summary has no execution prefix")

    merged = [merge_samples(group, tokenizer) for group in groups]
    if terminal_orphan is not None:
        # Avoid mutating an input sample when the retained execution group had one
        # turn (merge_samples intentionally returns that object directly).
        retained = deepcopy(merged[-1])
        retained.metadata = dict(retained.metadata or {})
        retained.metadata["compaction_terminal_orphan_summary_dropped"] = True
        retained.metadata["compaction_terminal_orphan_summary_segment_index"] = terminal_orphan.metadata[
            "compaction_segment_index"
        ]
        merged[-1] = retained
    return merged


def _compute_sample_from_openai_record(
    args: Namespace, record: SessionRecord, tokenizer, trim_count: int = 0
) -> Sample:
    choice = record.response["choices"][0]

    prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("input_ids not found in request — the session server should populate it")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = Sample()
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, choice, sample)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, choice, sample)

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    if args.sglang_speculative_algorithm:
        sample.spec_info.add(choice.get("meta_info", {}))
    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        sample.strip_last_output_tokens(overshoot, tokenizer)
        sample.status = Sample.Status.TRUNCATED
        result.append(sample)
        break

    return result
