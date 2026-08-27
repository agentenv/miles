from types import SimpleNamespace

import pytest

from miles.ray import placement_group as placement


def _args(**overrides):
    values = {
        "sao_one_gpu_island": True,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": True,
        "use_critic": True,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "critic_num_nodes": 1,
        "critic_num_gpus_per_node": 1,
        "rollout_num_gpus": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_one_gpu_sao_uses_same_bundle_for_all_three_roles(monkeypatch):
    marker = object()
    calls = []

    def fake_create(num_gpus, *, minimum_gpu_memory_bytes=None):
        calls.append((num_gpus, minimum_gpu_memory_bytes))
        return marker, [17], [3]

    monkeypatch.setattr(placement, "_create_placement_group", fake_create)

    groups = placement.create_placement_groups(_args())

    assert calls == [(1, placement._SAO_ONE_GPU_ISLAND_MIN_MEMORY_BYTES)]
    assert groups == {
        "actor": (marker, [17], [3]),
        "critic": (marker, [17], [3]),
        "rollout": (marker, [17], [3]),
    }


def test_one_gpu_sao_memory_gate_accepts_h200_capacity():
    placement._require_minimum_gpu_memory(
        [("node-a", 0.0, 141 * 1024**3)],
        placement._SAO_ONE_GPU_ISLAND_MIN_MEMORY_BYTES,
    )


def test_one_gpu_sao_memory_gate_rejects_smaller_gpu():
    with pytest.raises(RuntimeError, match="requires at least 120 GiB"):
        placement._require_minimum_gpu_memory(
            [("node-a", 0.0, 80 * 1024**3)],
            placement._SAO_ONE_GPU_ISLAND_MIN_MEMORY_BYTES,
        )
