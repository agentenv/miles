"""Per-rollout tallies a rollout function hands to the hooks of its own process.

The configuration a rollout function sees is immutable, so the tallies it keeps for a
rollout (what the abort discarded, what the agentic suspend kept, the over-sampling
tally) cannot ride on it. ``publish`` records the latest tally of a kind in this process;
``take`` returns it once and clears it, so a reader never sees the previous rollout's
numbers. Both run on the rollout event loop.
"""

from collections.abc import Mapping
from typing import Literal

StatsKind = Literal["abort_discard", "over_sampling", "suspend", "resume"]

_STATS: dict[str, dict[str, int]] = {}


def publish(kind: StatsKind, stats: Mapping[str, int]) -> None:
    _STATS[kind] = dict(stats)


def take(kind: StatsKind) -> dict[str, int] | None:
    return _STATS.pop(kind, None)
