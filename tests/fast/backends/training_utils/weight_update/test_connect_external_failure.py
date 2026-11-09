"""A rollout engine that dies during a publish fails the sender's connect.

Only the sender talks to the engines; the other trainer ranks must learn about the failure instead of waiting in
the next collective for a rank that gave up, and the trainer must then be ready to connect the old members again.
"""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.protocols.broadcast import RolloutEngineJoinError
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.utils.workers.worker_handle import ExternalFailureError

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_WORLD_SIZE = 4


def _make_updater(connect) -> WeightUpdater:
    protocol = SimpleNamespace(
        required_placement=MagicMock(), supports_lora=False, is_sender=None, connect=connect, rollout_engines=[]
    )
    iterator = MagicMock()
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        return WeightUpdater(
            Namespace(sglang=SimpleNamespace(base_args={"speculative_algorithm": None}), enable_mtp_training=False),
            [MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            iterator_factory=lambda *a, **k: iterator,
            parallel_state=MagicMock(),
            is_lora=False,
        )


def _info() -> UpdatableEngines:
    return UpdatableEngines(
        rollout_engines=[MagicMock(name="engine")],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
        engine_cell_ids=["cell-a"],
        snapshot_cell_id_to_hashes={"cell-a": "h2"},
    )


class _Gloo:
    """Stands in for the gloo group of ``_WORLD_SIZE`` ranks: records what each rank contributed."""

    def __init__(self, contributions: list) -> None:
        self.contributions = contributions

    def all_gather_object(self, out: list, obj, group=None) -> None:
        for i, value in enumerate(self.contributions):
            out[i] = value


def _connect(updater: WeightUpdater, *, gathered: list) -> None:
    gloo = _Gloo(gathered)
    with patch(f"{_UPDATER_MODULE}.dist") as dist_mock, patch(f"{_UPDATER_MODULE}.get_gloo_group"):
        dist_mock.get_world_size.return_value = _WORLD_SIZE
        dist_mock.all_gather_object.side_effect = gloo.all_gather_object
        updater.reconnect(_info())


class TestTheConnectOutcomeIsAgreedByEveryRank:
    def test_the_sender_raises_the_engine_failure_and_marks_the_connection_stale(self) -> None:
        failure = RolloutEngineJoinError("engine 0 failed to join weight update group miles-pp_0")
        updater = _make_updater(connect=MagicMock(side_effect=failure))
        updater.conn_status.mark_reconnected({"cell-a": "h1"})  # the old members were connected before

        with pytest.raises(RolloutEngineJoinError) as info:
            _connect(updater, gathered=[f"RolloutEngineJoinError: {failure}", None, None, None])

        assert info.value is failure
        assert updater.conn_status.needs_reconnect({"cell-a": "h1"}), "the next publication must connect again"

    def test_a_rank_whose_own_connect_succeeded_raises_the_sender_failure(self) -> None:
        connect = MagicMock()  # non-sender ranks do nothing engine-side and succeed locally
        updater = _make_updater(connect=connect)
        updater.protocol.is_sender = False
        updater.conn_status.mark_reconnected({"cell-a": "h1"})

        with pytest.raises(ExternalFailureError, match="rank 0 could not connect the rollout engines: .*engine 0"):
            _connect(updater, gathered=["RolloutEngineJoinError: engine 0 failed to join", None, None, None])

        connect.assert_called_once()
        assert updater.conn_status.needs_reconnect({"cell-a": "h1"})

    def test_when_every_rank_succeeds_nothing_changes(self) -> None:
        def connect(*args, **kwargs):
            protocol.is_sender = True

        updater = _make_updater(connect=connect)
        protocol = updater.protocol
        updater._registered_adapters.add("stale")

        _connect(updater, gathered=[None] * _WORLD_SIZE)

        assert updater._registered_adapters == set()
        assert not updater.conn_status.needs_reconnect({"cell-a": "h2"}), "a successful reconnect is recorded"

    def test_a_trainer_side_failure_is_still_a_trainer_failure(self) -> None:
        """Only engine-side failures are softened; a broken trainer rank keeps failing loudly (and is not agreed)."""
        updater = _make_updater(connect=MagicMock(side_effect=RuntimeError("NCCL init failed")))
        with pytest.raises(RuntimeError, match="NCCL init failed"), patch(f"{_UPDATER_MODULE}.dist") as dist_mock:
            updater.reconnect(_info())
        dist_mock.all_gather_object.assert_not_called()
