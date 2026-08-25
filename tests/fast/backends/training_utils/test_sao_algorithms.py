from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from miles.backends.training_utils.loss_hub.gae_adaptive import (
    _gae_loop,
    compute_adaptive_lambd,
    get_gae_adaptive_advantages,
)
from miles.backends.training_utils.loss_hub.math_utils import compute_sao_dis_policy_loss
from miles.backends.training_utils.sao import (
    apply_sao_online_recipe,
    freeze_attention_parameters,
    validate_sao_train_data,
)


def test_adaptive_lambda_matches_paper_formula() -> None:
    assert compute_adaptive_lambd(6, alpha=1.5, min_length=2) == 1.0 - 1.0 / 9.0
    assert compute_adaptive_lambd(1, alpha=1.5, min_length=1) == 1.0 - 1.0 / 1.5


def test_skip_observation_gae_bridges_between_action_tokens() -> None:
    advantages, returns = _gae_loop(
        token_rewards=torch.tensor([0.0, 0.0, 1.0]),
        token_values=torch.tensor([0.2, 999.0, 0.4]),
        loss_mask=torch.tensor([1, 0, 1]),
        gamma=1.0,
        lambd=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([0.8, 0.0, 0.6]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 0.0, 1.0]))


def test_terminal_reward_lands_on_last_action_not_trailing_observation() -> None:
    advantages, returns = get_gae_adaptive_advantages(
        rewards=[1.0],
        kl=[torch.zeros(4)],
        loss_masks=[torch.tensor([1, 0, 1, 0])],
        values=[torch.zeros(4)],
        response_lengths=[4],
        total_lengths=[6],
        mode="fixed",
        gamma=1.0,
        lambd=0.5,
    )

    torch.testing.assert_close(advantages[0], torch.tensor([0.5, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(returns[0], advantages[0])


def test_sao_dis_uses_direct_ratio_and_rejects_outside_tokens() -> None:
    rollout_log_probs = torch.full((4,), -2.0)
    requested_ratios = torch.tensor([0.69, 0.8, 5.9, 6.1])
    log_probs = (rollout_log_probs + requested_ratios.log()).requires_grad_()
    advantages = torch.tensor([1.0, 2.0, -1.0, 1.0])

    loss, ratio, rejected, below, above = compute_sao_dis_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_low=0.3,
        eps_high=5.0,
    )

    torch.testing.assert_close(ratio, requested_ratios)
    torch.testing.assert_close(rejected, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(below, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(above, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert loss[0] == 0
    assert loss[3] == 0

    loss.sum().backward()
    assert log_probs.grad[0] == 0
    assert log_probs.grad[3] == 0
    assert log_probs.grad[1] != 0
    assert log_probs.grad[2] != 0


def test_sao_dis_keeps_ratio_in_published_objective_gradient() -> None:
    rollout_log_prob = torch.tensor([-2.1])
    log_prob = torch.tensor([-2.0], requires_grad=True)
    advantage = torch.tensor([2.0])

    loss, ratio, *_ = compute_sao_dis_policy_loss(
        log_prob,
        rollout_log_prob,
        advantage,
        eps_low=0.3,
        eps_high=5.0,
    )
    loss.backward()

    expected = -ratio.detach() * advantage * (log_prob.detach() + 1.0)
    torch.testing.assert_close(log_prob.grad, expected)
    assert math.isfinite(log_prob.grad.item())


def test_sao_dis_bounds_are_strict() -> None:
    # With eps_low=0 the lower bound is exactly one; equality is rejected.
    log_prob = torch.tensor([-2.0], requires_grad=True)
    loss, _, rejected, below, _ = compute_sao_dis_policy_loss(
        log_prob,
        log_prob.detach().clone(),
        torch.ones(1),
        eps_low=0.0,
        eps_high=5.0,
    )

    assert rejected.item() == 1.0
    assert below.item() == 1.0
    assert loss.item() == 0.0


def test_frozen_attention_leaves_moe_and_value_head_trainable() -> None:
    class Critic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attention = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Linear(4, 4)
            self.output_layer = torch.nn.Linear(4, 3)

    critic = Critic()
    summary = freeze_attention_parameters(critic)

    assert summary.tensors == 2
    assert summary.elements == 20
    assert not critic.self_attention.weight.requires_grad
    assert not critic.self_attention.bias.requires_grad
    assert critic.mlp.weight.requires_grad
    assert critic.output_layer.weight.requires_grad


def test_coding_online_recipe_applies_complete_sao_structure() -> None:
    args = SimpleNamespace(
        sao_online_recipe="coding",
        value_pretrain_manifest=None,
        n_samples_per_prompt=1,
        advantage_estimator="grpo",
        policy_objective="ppo",
        gae_adaptive_mode="fixed",
        gae_adaptive_alpha=1.0,
        gae_adaptive_min_length=2,
        gamma=0.99,
        critic_lambd=0.9,
        num_critic_epochs=1,
        critic_freeze_attention=False,
        lr=2e-6,
        critic_lr=7e-6,
        critic_lr_warmup_iters=5,
        kl_coef=0.1,
        kl_loss_coef=0.2,
        use_kl_loss=True,
        entropy_coef=0.01,
        sao_dis_eps_low=0.3,
        sao_dis_eps_high=5.0,
    )

    apply_sao_online_recipe(args)

    assert args.advantage_estimator == "gae_adaptive"
    assert args.policy_objective == "sao_dis"
    assert args.gae_adaptive_mode == "adaptive"
    assert args.gae_adaptive_alpha == 1.5
    assert args.gae_adaptive_min_length == 1
    assert args.gamma == 1.0
    assert args.critic_lambd == 1.0
    assert args.num_critic_epochs == 2
    assert args.critic_freeze_attention is True
    assert args.lr == 1e-6
    assert args.critic_lr == 5e-6
    assert args.critic_lr_warmup_iters == 10
    assert args.kl_coef == 0.0
    assert args.kl_loss_coef == 0.0
    assert args.use_kl_loss is False
    assert args.entropy_coef == 0.0
    assert args.sao_dis_eps_low == 0.8
    assert args.sao_dis_eps_high == 3.0


def test_online_recipe_rejects_grouped_rollouts() -> None:
    args = SimpleNamespace(
        sao_online_recipe="coding",
        value_pretrain_manifest=None,
        n_samples_per_prompt=2,
    )
    try:
        apply_sao_online_recipe(args)
    except ValueError as error:
        assert "exactly one rollout" in str(error)
    else:
        raise AssertionError("grouped SAO rollouts must be rejected")


def test_sao_rollout_contract_accepts_multiturn_observation_masks() -> None:
    summary = validate_sao_train_data(
        SimpleNamespace(policy_objective="sao_dis"),
        {
            "tokens": [[10, 11, 12, 13, 14]],
            "response_lengths": [4],
            "rewards": [1.0],
            "loss_masks": [[1, 0, 1, 0]],
            "rollout_log_probs": [[-0.2, 0.0, -0.4, 0.0]],
        },
    )

    assert summary is not None
    assert summary.samples == 1
    assert summary.response_tokens == 4
    assert summary.active_action_tokens == 2


def test_sao_rollout_contract_rejects_missing_behavior_logprobs() -> None:
    try:
        validate_sao_train_data(
            SimpleNamespace(policy_objective="sao_dis"),
            {
                "tokens": [[10, 11]],
                "response_lengths": [1],
                "rewards": [1.0],
                "loss_masks": [[1]],
            },
        )
    except ValueError as error:
        assert "rollout_log_probs" in str(error)
    else:
        raise AssertionError("SAO must reject rollouts without behavior-policy logprobs")
