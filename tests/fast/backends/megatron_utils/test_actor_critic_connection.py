from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import actor as actor_module


@pytest.mark.parametrize(
    ("one_gpu_island", "expected_backend"),
    [(True, "gloo"), (False, "nccl")],
)
def test_actor_critic_process_group_backend_matches_topology(
    monkeypatch,
    one_gpu_island: bool,
    expected_backend: str,
) -> None:
    captured = {}
    process_group = object()

    def fake_init_process_group(**kwargs):
        captured.update(kwargs)
        return process_group

    monkeypatch.setattr(actor_module, "init_process_group", fake_init_process_group)
    actor = SimpleNamespace(
        role="critic",
        args=SimpleNamespace(sao_one_gpu_island=one_gpu_island),
    )

    actor_module.MegatronTrainRayActor.connect_actor_critic.__wrapped__(
        actor,
        master_address="127.0.0.1",
        master_port=12345,
    )

    assert actor._actor_critic_groups is process_group
    assert captured == {
        "backend": expected_backend,
        "init_method": "tcp://127.0.0.1:12345",
        "world_size": 2,
        "rank": 1,
        "group_name": "actor_critic",
    }
