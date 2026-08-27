"""Fail-closed reward/filter functions for OpenEnv Terminal-Bench runs.

The OpenEnv agent obtains the binary reward through the server's canonical
``evaluate`` action and the Codex harness signs that outcome with a dedicated
Terminal-Bench key. Both the reward function and dynamic filter authenticate
the signed outcome before a sample can enter training. The reward function is
wired through ``--custom-rm-path openenv_generate.reward_func`` and supports
both single-sample and batched call paths.
"""

import math
import numbers

from yeto.rl.tbench_outcome import MAC_KEY, verified_outcome

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

_TERMINAL_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})


def _explicit_reward(sample: Sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    outcome, signed_reward = verified_outcome(metadata)
    reported_reward = metadata.get("reward")
    if isinstance(reported_reward, bool) or not isinstance(reported_reward, numbers.Real):
        raise ValueError("Terminal-Bench sample has no explicit numeric verifier reward")
    reported_reward = float(reported_reward)
    if not math.isfinite(reported_reward) or reported_reward not in (0.0, 1.0) or reported_reward != signed_reward:
        raise ValueError("Terminal-Bench reported reward differs from its signed verifier outcome")
    if metadata.get("exit_status") != outcome["status"]:
        raise ValueError("Terminal-Bench exit status differs from its signed verifier outcome")
    return signed_reward


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [_explicit_reward(sample) for sample in samples]
    return _explicit_reward(samples)


def _flatten_samples(samples):
    for sample in samples:
        if isinstance(sample, list):
            yield from _flatten_samples(sample)
        else:
            yield sample


def check_terminal_bench_episode(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject aborted or unverifiable environment episodes before training."""
    flattened = list(_flatten_samples(samples))
    for sample in flattened:
        if sample.status == Sample.Status.ABORTED:
            return DynamicFilterOutput(keep=False, reason="terminal_bench_aborted")
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if metadata.get("exit_status") not in _TERMINAL_STATUSES:
            return DynamicFilterOutput(keep=False, reason="terminal_bench_no_verdict")
        try:
            _explicit_reward(sample)
        except (RuntimeError, ValueError):
            return DynamicFilterOutput(keep=False, reason="terminal_bench_invalid_reward")

    # Compaction segments from one logical rollout must carry exactly the same
    # host-signed verdict. Segment-specific token data is authenticated later
    # by Yeto's trajectory evidence; only the outcome is shared here.
    compacted: dict[object, set[object]] = {}
    for sample in flattened:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        trajectory_id = metadata.get("compaction_trajectory_id")
        if trajectory_id is None:
            continue
        compacted.setdefault(trajectory_id, set()).add(metadata.get(MAC_KEY))
    if any(len(outcomes) != 1 for outcomes in compacted.values()):
        return DynamicFilterOutput(keep=False, reason="terminal_bench_compaction_verdict_mismatch")
    return DynamicFilterOutput(keep=True)
