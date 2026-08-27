import json
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


async def test_actor_critic_training_cancels_actor_when_critic_fails(monkeypatch):
    import asyncio

    async def eager_create_task(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(train_module, "eager_create_task", eager_create_task)
    actor_started = asyncio.Event()
    actor_cancelled = asyncio.Event()
    critic_failure = RuntimeError("critic failed")

    class Actor:
        async def train(self, _rollout_id, _rollout_data):
            actor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                actor_cancelled.set()

    class Critic:
        async def train(self, _rollout_id, _rollout_data):
            await actor_started.wait()
            raise critic_failure

    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(
            train_module._train_actor_and_critic(Actor(), Critic(), 7, "batch"),
            timeout=1,
        )

    assert exc_info.value is critic_failure
    assert actor_cancelled.is_set()


async def test_actor_critic_training_cancels_critic_when_actor_fails(monkeypatch):
    import asyncio

    async def eager_create_task(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(train_module, "eager_create_task", eager_create_task)
    critic_started = asyncio.Event()
    critic_cancelled = asyncio.Event()
    actor_failure = RuntimeError("actor failed")

    class Actor:
        async def train(self, _rollout_id, _rollout_data):
            await critic_started.wait()
            raise actor_failure

    class Critic:
        async def train(self, _rollout_id, _rollout_data):
            critic_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                critic_cancelled.set()

    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(
            train_module._train_actor_and_critic(Actor(), Critic(), 7, "batch"),
            timeout=1,
        )

    assert exc_info.value is actor_failure
    assert critic_cancelled.is_set()


async def test_external_policy_sync_requires_explicit_critic_capability(monkeypatch):
    args = SimpleNamespace(external_policy_sync_path="project.sync.create")

    class ActorOnlySync:
        pass

    monkeypatch.setattr(
        train_module,
        "load_function",
        lambda _path: lambda _args: ActorOnlySync(),
    )
    with pytest.raises(ValueError, match="does not declare critic support"):
        train_module._load_external_policy_sync(args, critic_model=object())

    class ActorCriticSync:
        supports_critic = True

    monkeypatch.setattr(
        train_module,
        "load_function",
        lambda _path: lambda _args: ActorCriticSync(),
    )
    synchronizer = train_module._load_external_policy_sync(args, critic_model=object())
    assert isinstance(synchronizer, ActorCriticSync)


async def test_exact_publication_callback_receives_weight_transfer_engine_set():
    observed = []
    actor = object()
    publication_info = object()

    class Sync:
        requires_exact_publication_info = True

        async def after_inference_publication(
            self,
            *,
            rollout_id,
            actor_model,
            publication_info,
        ):
            observed.append((rollout_id, actor_model, publication_info))

    sync = Sync()
    await train_module._notify_external_policy_publication(
        sync,
        rollout_id=3,
        actor_model=actor,
        publication_info=publication_info,
    )
    assert observed == [(3, actor, publication_info)]

    with pytest.raises(RuntimeError, match="exact weight-publication engines"):
        await train_module._notify_external_policy_publication(
            sync,
            rollout_id=4,
            actor_model=actor,
        )


async def test_critic_aware_external_sync_runs_after_actor_and_critic_train(monkeypatch):
    import asyncio

    events = []
    values_ready = asyncio.Event()
    actor_finished = asyncio.Event()
    critic_finished = asyncio.Event()

    class Actor:
        async def update_weights(self, rollout_id=None):
            events.append(("publish", rollout_id))

        async def train(self, rollout_id, rollout_data):
            await values_ready.wait()
            events.append(("actor", rollout_id, rollout_data))
            actor_finished.set()

        async def clear_memory(self):
            events.append("clear")

    class Critic:
        async def train(self, rollout_id, rollout_data):
            events.append(("critic-values", rollout_id, rollout_data))
            values_ready.set()
            await actor_finished.wait()
            events.append(("critic-updates", rollout_id))
            critic_finished.set()

    actor = Actor()
    critic = Critic()
    rollout = SimpleNamespace(
        generate=_Remote(lambda rollout_id: events.append(("rollout", rollout_id)) or "batch"),
        dispose=_Remote(lambda: events.append("dispose")),
    )

    class Sync:
        supports_critic = True

        async def initialize(self, *, actor_model, critic_model, rollout_manager):
            assert actor_model is actor
            assert critic_model is critic
            assert rollout_manager is rollout
            events.append("initialize")

        async def after_local_train(
            self,
            *,
            rollout_id,
            actor_model,
            critic_model,
            rollout_data,
        ):
            assert actor_model is actor
            assert critic_model is critic
            assert rollout_data == "batch"
            assert actor_finished.is_set()
            assert critic_finished.is_set()
            events.append(("sync", rollout_id))

        async def finalize(self):
            events.append("finalize")

    async def eager_create_task(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(
        train_module,
        "create_placement_groups",
        lambda _args: {"rollout": object()},
    )
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda *_args: (rollout, 1),
    )

    async def create_models(*_args):
        return actor, critic

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "eager_create_task", eager_create_task)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "load_function", lambda _path: lambda _args: Sync())
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *_args: False)

    args = SimpleNamespace(
        external_policy_sync_path="project.sync.create",
        external_policy_sync_run_until_stop=False,
        control_server_port=None,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=1,
        eval_interval=None,
        offload_train=False,
        use_critic=True,
        start_rollout_id=0,
        num_critic_only_steps=0,
        skip_eval_before_train=False,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
    )

    await train_module.train(args)

    assert events == [
        "initialize",
        ("publish", None),
        ("rollout", 0),
        ("critic-values", 0, "batch"),
        ("actor", 0, "batch"),
        ("critic-updates", 0),
        ("sync", 0),
        "clear",
        ("publish", 0),
        "finalize",
        "dispose",
    ]


async def test_centralized_actor_critic_publishes_only_after_both_train(monkeypatch):
    import asyncio

    events = []
    values_ready = asyncio.Event()
    actor_finished = asyncio.Event()

    class Actor:
        async def update_weights(self, rollout_id=None):
            events.append(("publish", rollout_id))

        async def train(self, rollout_id, rollout_data):
            await values_ready.wait()
            events.append(("actor", rollout_id, rollout_data))
            actor_finished.set()

        async def clear_memory(self):
            events.append("clear")

    class Critic:
        async def train(self, rollout_id, rollout_data):
            events.append(("critic-values", rollout_id, rollout_data))
            values_ready.set()
            await actor_finished.wait()
            events.append(("critic-updates", rollout_id))

    actor = Actor()
    critic = Critic()
    rollout = SimpleNamespace(
        generate=_Remote(lambda rollout_id: events.append(("rollout", rollout_id)) or "batch"),
        dispose=_Remote(lambda: events.append("dispose")),
    )

    async def eager_create_task(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda *_args: (rollout, 1))

    async def create_models(*_args):
        return actor, critic

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "eager_create_task", eager_create_task)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *_args: False)

    args = SimpleNamespace(
        external_policy_sync_path=None,
        control_server_port=None,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=1,
        eval_interval=None,
        offload_train=False,
        use_critic=True,
        start_rollout_id=0,
        num_critic_only_steps=0,
        skip_eval_before_train=False,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
    )

    await train_module.train(args)

    assert events == [
        ("publish", None),
        ("rollout", 0),
        ("critic-values", 0, "batch"),
        ("actor", 0, "batch"),
        ("critic-updates", 0),
        "clear",
        ("publish", 0),
        "dispose",
    ]


async def test_checkpoint_backed_rollout_only_publishes_then_never_trains(
    monkeypatch,
    tmp_path,
):
    events = []
    checksum_calls = 0
    version_calls = 0

    class Actor:
        async def update_weights(self, rollout_id=None):
            events.append(("publish", rollout_id))

        async def train(self, *_args, **_kwargs):
            raise AssertionError("checkpoint-backed rollout-only mode trained actor")

    actor = Actor()

    def check_weights(**kwargs):
        nonlocal checksum_calls
        action = kwargs["action"]
        events.append(("check_weights", action))
        if action != "checksum":
            return [[{"success": True}]]
        checksum_calls += 1
        value = {
            1: "base",
            2: "trained",
            3: "reset",
            4: "trained",
        }[checksum_calls]
        return [
            [
                {
                    "success": True,
                    "ranks": [
                        {
                            "checksums": {"model.language_model.weight": value},
                            "parallelism_info": [{"role": "target", "rank": 0}],
                        }
                    ],
                }
            ]
        ]

    def get_weight_versions():
        nonlocal version_calls
        version_calls += 1
        value = {1: "default", 2: "1", 3: "2"}[version_calls]
        events.append(("weight_version", value))
        return [value]

    rollout = SimpleNamespace(
        check_weights=_Remote(check_weights),
        get_updatable_weight_versions=_Remote(get_weight_versions),
        generate=_Remote(lambda rollout_id: events.append(("rollout", rollout_id)) or "batch"),
        dispose=_Remote(lambda: events.append("dispose")),
    )

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda *_args: (rollout, 1))

    async def create_models(*_args):
        return actor, None

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)

    checkpoint = tmp_path / "actor-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n")

    args = SimpleNamespace(
        external_policy_sync_path=None,
        control_server_port=None,
        offload_rollout=False,
        check_weight_update_equal=False,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector="target",
        check_weight_update_skip_list=None,
        num_rollout=1,
        eval_interval=None,
        offload_train=False,
        use_critic=False,
        start_rollout_id=0,
        num_critic_only_steps=0,
        skip_eval_before_train=False,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
        rollout_only_from_checkpoint=True,
        rollout_only_publication_evidence=str(tmp_path / "publication.json"),
        load=str(checkpoint),
        ref_load=str(checkpoint),
    )

    await train_module.train(args)

    assert events == [
        ("check_weights", "checksum"),
        ("weight_version", "default"),
        ("publish", None),
        ("check_weights", "checksum"),
        ("weight_version", "1"),
        ("check_weights", "snapshot"),
        ("check_weights", "reset_tensors"),
        ("check_weights", "checksum"),
        ("publish", None),
        ("weight_version", "2"),
        ("check_weights", "compare"),
        ("check_weights", "checksum"),
        ("rollout", 0),
        "dispose",
    ]
    evidence = json.loads((tmp_path / "publication.json").read_text())
    assert evidence["changed_language_tensor_count"] == 1
    assert evidence["reset_changed_language_tensor_count"] == 1
    assert evidence["bootstrap_weight_versions"] == ["default"]
    assert evidence["served_weight_versions"] == ["1"]
    assert evidence["republished_weight_versions"] == ["2"]
    assert evidence["trained_snapshot_reset_republication_equal"] is True


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

        async def prepare_weight_update(self):
            assert self.awake
            events.append("prepare_weight_update")

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

        async def after_inference_publication(self, *, rollout_id, actor_model):
            assert actor_model is actor
            events.append(("published", rollout_id))

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
        ("published", None),
        "offload",
        ("train", 0, "rollout-0"),
        ("sync", 0, "rollout-0"),
        "prepare_weight_update",
        "offload",
        ("update", 0),
        ("published", 0),
        "finalize",
    ]


async def test_external_policy_sync_offloads_trainer_before_initial_rollout_onload(
    monkeypatch,
):
    events = []

    class Actor:
        awake = False

        async def onload(self):
            assert not self.awake
            self.awake = True
            events.append("trainer_onload")

        async def prepare_weight_update(self):
            assert self.awake
            events.append("prepare_weight_update")

        async def offload(self):
            assert self.awake
            self.awake = False
            events.append("trainer_offload")

        async def update_weights(self, rollout_id=None):
            assert not self.awake
            events.append(("update", rollout_id))

    actor = Actor()
    rollout = SimpleNamespace(
        onload_weights=_Remote(lambda: events.append("rollout_onload_weights")),
        onload_kv=_Remote(lambda: events.append("rollout_onload_kv")),
        dispose=_Remote(lambda: events.append("dispose")),
    )

    class Sync:
        async def initialize(self, *, actor_model, rollout_manager):
            assert actor_model is actor and rollout_manager is rollout
            assert actor_model.awake
            events.append("initialize")

        async def finalize(self):
            events.append("finalize")

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(
        train_module,
        "create_placement_groups",
        lambda _args: {"rollout": object()},
    )
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda *_args: (rollout, 1))

    async def create_models(*_args):
        return actor, None

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "load_function", lambda _path: lambda _args: Sync())

    args = SimpleNamespace(
        external_policy_sync_path="project.sync.create",
        external_policy_sync_run_until_stop=False,
        control_server_port=None,
        offload_rollout=True,
        offload_train=True,
        check_weight_update_equal=False,
        num_rollout=0,
        eval_interval=None,
        use_critic=False,
        start_rollout_id=0,
        skip_eval_before_train=False,
    )

    await train_module.train(args)

    assert events == [
        "trainer_onload",
        "initialize",
        "prepare_weight_update",
        "trainer_offload",
        "rollout_onload_weights",
        ("update", None),
        "rollout_onload_kv",
        "finalize",
        "dispose",
    ]


@pytest.mark.parametrize("offload_rollout", (False, True))
async def test_critic_aware_external_sync_has_symmetric_initial_offload(
    monkeypatch,
    offload_rollout,
):
    events = []

    class Actor:
        awake = False

        async def onload(self):
            assert not self.awake
            self.awake = True
            events.append("actor_onload")

        async def prepare_weight_update(self):
            assert self.awake
            events.append("actor_prepare")

        async def offload(self):
            assert self.awake
            self.awake = False
            events.append("actor_offload")

        async def update_weights(self, rollout_id=None):
            assert self.awake is (not offload_rollout)
            events.append(("actor_publish", rollout_id))

    class Critic:
        awake = False

        async def onload(self):
            assert not self.awake
            self.awake = True
            events.append("critic_onload")

        async def offload(self):
            assert self.awake
            self.awake = False
            events.append("critic_offload")

    actor = Actor()
    critic = Critic()

    def rollout_onload_weights():
        assert not actor.awake
        assert not critic.awake
        events.append("rollout_onload_weights")

    rollout = SimpleNamespace(
        onload_weights=_Remote(rollout_onload_weights),
        onload_kv=_Remote(lambda: events.append("rollout_onload_kv")),
        dispose=_Remote(lambda: events.append("dispose")),
    )

    class Sync:
        supports_critic = True

        async def initialize(
            self,
            *,
            actor_model,
            critic_model,
            rollout_manager,
        ):
            assert actor_model is actor
            assert critic_model is critic
            assert rollout_manager is rollout
            assert actor_model.awake
            assert critic_model.awake
            events.append("initialize")

        async def finalize(self):
            assert not actor.awake
            assert not critic.awake
            events.append("finalize")

    monkeypatch.setattr(train_module, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_module, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(
        train_module,
        "create_placement_groups",
        lambda _args: {"rollout": object()},
    )
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda *_args: (rollout, 1),
    )

    async def create_models(*_args):
        return actor, critic

    monkeypatch.setattr(train_module, "create_training_models", create_models)
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "load_function", lambda _path: lambda _args: Sync())

    args = SimpleNamespace(
        external_policy_sync_path="project.sync.create",
        external_policy_sync_run_until_stop=False,
        control_server_port=None,
        offload_rollout=offload_rollout,
        offload_train=True,
        check_weight_update_equal=False,
        num_rollout=0,
        eval_interval=None,
        use_critic=True,
        start_rollout_id=0,
        skip_eval_before_train=False,
    )

    await train_module.train(args)

    expected = [
        "actor_onload",
        "critic_onload",
        "initialize",
    ]
    if offload_rollout:
        expected.extend(
            [
                "actor_prepare",
                "actor_offload",
                "critic_offload",
                "rollout_onload_weights",
                ("actor_publish", None),
                "rollout_onload_kv",
            ]
        )
    else:
        expected.extend(
            [
                ("actor_publish", None),
                "actor_offload",
                "critic_offload",
            ]
        )
    expected.extend(["finalize", "dispose"])
    assert events == expected
    assert not actor.awake
    assert not critic.awake


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
