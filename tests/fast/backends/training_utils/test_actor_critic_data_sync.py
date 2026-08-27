from types import SimpleNamespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.data import sync_actor_critic_data
from tests.fast.dist_utils import init_gloo, run_multiprocess


def test_one_gpu_sync_uses_standalone_group_rank(monkeypatch) -> None:
    class Group:
        @staticmethod
        def rank() -> int:
            return 1

    class Handle:
        @staticmethod
        def wait() -> None:
            return None

    sent = []
    monkeypatch.setattr(dist, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(
        dist,
        "get_rank",
        lambda _group: (_ for _ in ()).throw(
            AssertionError("standalone actor/critic rank must come from ProcessGroup.rank()")
        ),
    )
    monkeypatch.setattr(
        dist,
        "broadcast",
        lambda tensor, **_kwargs: sent.append(tensor.clone()) or Handle(),
    )
    values = [torch.tensor([0.25, 0.75], dtype=torch.float32)]
    rollout_data = {"values": values, "log_probs": [], "ref_log_probs": []}
    args = SimpleNamespace(
        kl_coef=0.0,
        use_kl_loss=False,
        use_rollout_logprobs=False,
        sao_one_gpu_island=True,
    )

    sync_actor_critic_data(args, rollout_data, Group())

    assert len(sent) == 1
    torch.testing.assert_close(sent[0], values[0])


def _run_bidirectional_sync(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        values = [torch.tensor([1.5, 2.5], dtype=torch.float32), torch.tensor([-3.0], dtype=torch.float32)]
        log_probs = [torch.tensor([0.1, 0.2], dtype=torch.float32), torch.tensor([0.3], dtype=torch.float32)]
        ref_log_probs = [torch.tensor([-0.1, -0.2], dtype=torch.float32), torch.tensor([-0.3], dtype=torch.float32)]

        rollout_data = {
            "values": [tensor.clone() for tensor in values] if rank == 1 else [],
            "log_probs": [tensor.clone() for tensor in log_probs] if rank == 0 else [],
            "ref_log_probs": [tensor.clone() for tensor in ref_log_probs] if rank == 0 else [],
        }
        args = SimpleNamespace(
            kl_coef=1.0,
            use_kl_loss=False,
            use_rollout_logprobs=False,
            sao_one_gpu_island=True,
        )

        sync_actor_critic_data(args, rollout_data, dist.group.WORLD)

        for actual, expected in zip(rollout_data["values"], values, strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(rollout_data["log_probs"], log_probs, strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(rollout_data["ref_log_probs"], ref_log_probs, strict=True):
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def test_one_gpu_actor_critic_gloo_syncs_values_and_kl_tensors() -> None:
    run_multiprocess(_run_bidirectional_sync)
