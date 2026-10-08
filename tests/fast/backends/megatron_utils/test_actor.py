import importlib
import sys
from argparse import Namespace
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module() -> Iterator[ModuleType]:
    """Import the Megatron actor with its unavailable native memory dependency stubbed."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


class TestCriticValuesValueSpec:
    def test_critic_values_are_shipped_as_a_typed_ragged_field(self, actor_module: ModuleType) -> None:
        """Variable-length critic sequences require the typed ragged object-store codec."""
        assert actor_module.CRITIC_VALUES_VALUE_SPEC["values"].codec == "typed_ragged"


class TestMaterializeCriticValues:
    @pytest.mark.parametrize("tensor_input", [False, True])
    def test_ragged_values_become_owned_float32_tensors(self, actor_module: ModuleType, tensor_input: bool) -> None:
        """Both store codecs yield owned FP32 tensors independent of released storage."""
        sequences = [[1.5, -2.0], [], [3.0]]
        values = [torch.tensor(value, dtype=torch.float32) for value in sequences] if tensor_input else sequences

        result = actor_module._materialize_critic_values(values=values, device=torch.device("cpu"))
        values[0][0] = 99.0

        assert [value.tolist() for value in result] == [[1.5, -2.0], [], [3.0]]
        assert all(value.dtype == torch.float32 for value in result)
        assert all(value.device == torch.device("cpu") for value in result)


class TestSendCheckpoint:
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor_module.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor_module, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted


class TestUpdateWeightsWhenAnEngineFailsToJoin:
    """A27 (GPU 1r5 d2): the engines failed, not the trainer. The actor must raise the engine-side error as is
    (so the cell keeps it alive) and leave its temporary process groups as the normal path would."""

    @staticmethod
    def _make_actor(actor_module: ModuleType, *, offload_train: bool, asleep: bool):
        from miles.backends.training_utils.weight_update.protocols.broadcast import RolloutEngineJoinError

        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = Namespace(
            debug_train_only=False, debug_rollout_only=False, offload_train=offload_train, debug_skip_weight_update=False
        )
        actor._heartbeat = Mock()
        actor._asleep = asleep
        actor.weight_updater = Mock()
        actor.weight_updater.conn_status.needs_reconnect.return_value = True
        actor.weight_updater.connect_rollout_engines.side_effect = RolloutEngineJoinError("engine 0 failed to join")
        return actor

    @staticmethod
    def _info():
        from miles.ray.rollout.inference_controller import UpdatableEngines

        return UpdatableEngines(
            rollout_engines=[Mock()], engine_gpu_counts=[1], engine_gpu_offsets=[0], snapshot_cell_id_to_hashes={"c": "h"}
        )

    def test_the_engine_failure_propagates_and_the_connection_is_not_marked_reconnected(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from miles.utils.workers.worker_handle import ExternalFailureError

        actor = self._make_actor(actor_module, offload_train=False, asleep=False)
        destroy = Mock()
        monkeypatch.setattr(actor_module, "destroy_process_groups", destroy)
        monkeypatch.setattr(actor_module, "reload_process_groups", Mock())
        monkeypatch.setattr(actor_module, "dist", Mock())

        with pytest.raises(ExternalFailureError, match="engine 0 failed to join"):
            actor.update_weights(self._info())

        actor.weight_updater.conn_status.mark_reconnected.assert_not_called()
        actor.weight_updater.update_weights.assert_not_called()
        destroy.assert_not_called()

    def test_temporary_process_groups_are_destroyed_again_on_the_failure_path(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from miles.utils.workers.worker_handle import ExternalFailureError

        actor = self._make_actor(actor_module, offload_train=True, asleep=True)
        monkeypatch.setattr(actor_module, "torch_memory_saver", MagicMock())  # disable() is a context manager
        reload, destroy = Mock(), Mock()
        monkeypatch.setattr(actor_module, "reload_process_groups", reload)
        monkeypatch.setattr(actor_module, "destroy_process_groups", destroy)
        monkeypatch.setattr(actor_module, "dist", Mock())

        with pytest.raises(ExternalFailureError):
            actor.update_weights(self._info())

        reload.assert_called_once()
        destroy.assert_called_once()


class TestDistributedLoraModeGuard:
    """LoRA over distributed (non-colocated) engines: raw is allowed since the broadcast protocol gathers the
    adapter across PP onto one sender; other modes are still refused."""

    def _init(self, actor_module: ModuleType, mode: str, monkeypatch) -> MagicMock:
        updater = MagicMock(name="WeightUpdater")
        monkeypatch.setattr(actor_module, "WeightUpdater", updater)
        monkeypatch.setattr(actor_module, "get_parallel_state", MagicMock())
        monkeypatch.setattr(actor_module, "build_lora_config", MagicMock(return_value={}))
        actor = SimpleNamespace(
            args=Namespace(
                model_name="qwen4_exp", lora_rank=16, colocate=False, megatron_to_hf_mode=mode,
                lora_adapter_targets=["q_proj"],
            ),
            hf_config=SimpleNamespace(),
            model=[MagicMock()],
            _get_actor_weights=MagicMock(),
        )
        actor_module.MegatronTrainRayActor._init_weight_updater_and_publisher(
            actor, update_weights=True, publish_snapshots=False
        )
        return updater

    @pytest.mark.parametrize("mode", ["raw", "bridge"])
    def test_raw_and_bridge_build_the_lora_updater(self, actor_module: ModuleType, mode: str, monkeypatch) -> None:
        updater = self._init(actor_module, mode, monkeypatch)
        assert updater.call_args.kwargs["is_lora"] is True

    def test_other_modes_are_refused(self, actor_module: ModuleType, monkeypatch) -> None:
        with pytest.raises(AssertionError, match="bridge or raw"):
            self._init(actor_module, "hf", monkeypatch)
