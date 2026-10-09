import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0
        # over-sampling tally (yeto rl-algo-supplement 2.4); reported only once a submission is recorded
        self._submit_batch_sizes: list[int] = []
        self._completed_groups = 0
        self._filtered_groups = 0
        self._surplus_groups = 0
        self._kept_groups = 0
        self._inflight_groups_at_end = 0
        self._resumed_groups = 0
        self._failed_groups = 0

    def on_submit(self, num_groups: int) -> None:
        """One call to the data source: ``num_groups`` groups submitted for generation."""
        self._submit_batch_sizes.append(int(num_groups))

    def on_groups_resumed(self, num_groups: int) -> None:
        """Groups carried over from the previous rollout (agentic suspend), not submitted now."""
        self._resumed_groups += int(num_groups)

    def on_group_failed(self) -> None:
        """A generation task raised: the group is neither completed nor in flight."""
        self._failed_groups += 1

    def on_group_completed(self) -> None:
        self._completed_groups += 1

    def on_group_kept(self) -> None:
        self._kept_groups += 1

    def on_group_surplus(self) -> None:
        """A group passed the filter after the batch was already full (not used for training)."""
        self._surplus_groups += 1

    def on_rollout_end(self, inflight_groups: int) -> None:
        """Groups still in flight when the batch is full (aborted, recycled or suspended)."""
        self._inflight_groups_at_end = int(inflight_groups)

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_group_filtered(self) -> None:
        self._filtered_groups += 1

    def over_sampling_stats(self) -> dict[str, int] | None:
        """The over-sampling tally with short keys (None before any submission)."""
        p = "rollout/over_sampling/"
        return {k[len(p):]: v for k, v in self.collect().items() if k.startswith(p)} or None

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        if self._submit_batch_sizes or self._resumed_groups:
            # submitted + resumed == completed + failed + inflight_at_end
            # completed == filtered + kept + surplus
            sizes = self._submit_batch_sizes
            p = "rollout/over_sampling/"
            metrics[p + "submit_calls"] = len(sizes)
            metrics[p + "refill_calls"] = max(0, len(sizes) - 1)
            metrics[p + "submitted_groups"] = sum(sizes)
            metrics[p + "submit_batch_size_first"] = sizes[0] if sizes else 0
            metrics[p + "submit_batch_size_max"] = max(sizes) if sizes else 0
            metrics[p + "resumed_groups"] = self._resumed_groups
            metrics[p + "failed_groups"] = self._failed_groups
            metrics[p + "completed_groups"] = self._completed_groups
            metrics[p + "filtered_groups"] = self._filtered_groups
            metrics[p + "kept_groups"] = self._kept_groups
            metrics[p + "surplus_groups"] = self._surplus_groups
            metrics[p + "inflight_groups_at_end"] = self._inflight_groups_at_end
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
