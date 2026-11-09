import pytest

from miles.rollout import rollout_runtime_stats


@pytest.fixture(autouse=True)
def _clear_stats():
    rollout_runtime_stats._STATS.clear()
    yield
    rollout_runtime_stats._STATS.clear()


class TestRolloutRuntimeStats:
    def test_take_returns_the_published_tally_once(self) -> None:
        rollout_runtime_stats.publish("abort_discard", {"groups": 2, "samples": 16})
        assert rollout_runtime_stats.take("abort_discard") == {"groups": 2, "samples": 16}
        assert rollout_runtime_stats.take("abort_discard") is None

    def test_take_without_publish_is_none(self) -> None:
        assert rollout_runtime_stats.take("over_sampling") is None

    def test_publish_copies_and_latest_wins(self) -> None:
        stats = {"groups": 1}
        rollout_runtime_stats.publish("suspend", stats)
        stats["groups"] = 99
        rollout_runtime_stats.publish("resume", {"groups": 3})
        rollout_runtime_stats.publish("resume", {"groups": 4})
        assert rollout_runtime_stats.take("suspend") == {"groups": 1}
        assert rollout_runtime_stats.take("resume") == {"groups": 4}

    def test_kinds_are_independent(self) -> None:
        rollout_runtime_stats.publish("abort_discard", {"a": 1})
        rollout_runtime_stats.publish("over_sampling", {"b": 2})
        assert rollout_runtime_stats.take("over_sampling") == {"b": 2}
        assert rollout_runtime_stats.take("abort_discard") == {"a": 1}
