import sys
import types
from types import SimpleNamespace

import pytest

constants = types.ModuleType("sglang.srt.constants")
constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
sys.modules.setdefault("sglang", types.ModuleType("sglang"))
sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
sys.modules.setdefault("sglang.srt.constants", constants)


def _module(name, **values):
    module = types.ModuleType(name)
    for key, value in values.items():
        setattr(module, key, value)
    sys.modules[name] = module


_module(
    "miles.ray.placement_group",
    create_placement_groups=None,
    create_rollout_manager=None,
    create_training_models=None,
)
_module("miles.utils.arguments", parse_args=None)
_module("miles.utils.async_utils", eager_create_task=None)
_module(
    "miles.utils.audit_utils.process_identity",
    MainProcessIdentity=object,
)
_module(
    "miles.utils.debug_utils.periodic_py_spy",
    maybe_start_periodic_pyspy_dump=None,
)
_module("miles.utils.ft_utils.control_server.server", start_control_server=None)
_module(
    "miles.utils.ft_utils.mini_ft_controller",
    maybe_start_mini_ft_controller=None,
)
_module("miles.utils.logging_utils", configure_logger=None)
_module("miles.utils.misc", load_function=None, should_run_periodic_action=None)
_module(
    "miles.utils.tracking_utils.tracking",
    finish_tracking=None,
    init_tracking=None,
)

import train as train_module

pytestmark = pytest.mark.asyncio


class _Remote:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


async def test_external_policy_sync_wraps_miles_weight_publication(monkeypatch):
    events = []

    class Actor:
        awake = False

        async def onload(self):
            assert not self.awake
            self.awake = True
            events.append("onload")

        async def offload(self):
            assert self.awake
            self.awake = False
            events.append("offload")

        async def update_weights(self, rollout_id=None):
            assert self.awake is (rollout_id is None)
            events.append(("update", rollout_id))

        async def train(self, rollout_id, rollout_data):
            assert not self.awake
            self.awake = True
            events.append(("train", rollout_id, rollout_data))

        async def clear_memory(self):
            pass

    actor = Actor()
    rollout = SimpleNamespace(
        generate=_Remote(lambda rollout_id: f"rollout-{rollout_id}"),
        dispose=_Remote(lambda: None),
    )

    class Sync:
        async def initialize(self, *, actor_model, rollout_manager):
            assert actor_model is actor and rollout_manager is rollout
            assert actor_model.awake
            events.append("initialize")

        async def after_local_train(self, *, rollout_id, actor_model, rollout_data):
            assert actor_model is actor
            events.append(("sync", rollout_id, rollout_data))

        async def finalize(self):
            events.append("finalize")

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda *_args: (rollout, 1))

    async def create_models(*_args):
        return actor, None

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "load_function", lambda _path: lambda _args: Sync())
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *_args: False)

    args = SimpleNamespace(
        external_policy_sync_path="project.sync.create",
        control_server_port=None,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=1,
        eval_interval=None,
        offload_train=True,
        use_critic=False,
        start_rollout_id=0,
        skip_eval_before_train=False,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
        external_policy_sync_run_until_stop=False,
    )

    await train_module.train(args)

    assert events == [
        "onload",
        "initialize",
        ("update", None),
        "offload",
        ("train", 0, "rollout-0"),
        ("sync", 0, "rollout-0"),
        "offload",
        ("update", 0),
        "finalize",
    ]


async def test_external_policy_sync_stops_only_after_final_weight_publication(
    monkeypatch,
):
    events = []

    class Actor:
        async def update_weights(self, rollout_id=None):
            events.append(("update", rollout_id))

        async def train(self, rollout_id, rollout_data):
            events.append(("train", rollout_id))

        async def clear_memory(self):
            pass

    actor = Actor()
    rollout = SimpleNamespace(
        generate=_Remote(lambda rollout_id: f"rollout-{rollout_id}"),
        dispose=_Remote(lambda: None),
    )

    class Sync:
        async def initialize(self, **_kwargs):
            pass

        async def after_local_train(self, *, rollout_id, **_kwargs):
            events.append(("sync", rollout_id))
            return rollout_id == 1

        async def finalize(self):
            events.append("finalize")

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda *_args: (rollout, 1))

    async def create_models(*_args):
        return actor, None

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "load_function", lambda _path: lambda _args: Sync())
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *_args: False)

    args = SimpleNamespace(
        external_policy_sync_path="project.sync.create",
        external_policy_sync_run_until_stop=True,
        control_server_port=None,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=1,
        eval_interval=None,
        offload_train=False,
        use_critic=False,
        start_rollout_id=0,
        skip_eval_before_train=False,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
    )

    await train_module.train(args)

    assert events == [
        ("update", None),
        ("train", 0),
        ("sync", 0),
        ("update", 0),
        ("train", 1),
        ("sync", 1),
        ("update", 1),
        "finalize",
    ]
