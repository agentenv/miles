"""Small, backend-independent helpers for the SAO actor-critic recipe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


_ATTENTION_COMPONENTS = {
    "attention",
    "attn",
    "cross_attention",
    "cross_attn",
    "self_attention",
    "self_attn",
}


@dataclass(frozen=True)
class FrozenParameterSummary:
    tensors: int
    elements: int


@dataclass(frozen=True)
class SAORolloutSummary:
    samples: int
    response_tokens: int
    active_action_tokens: int


def validate_sao_train_data(args: Any, data: Mapping[str, Any]) -> SAORolloutSummary | None:
    """Fail before GPU training when a DIS rollout lacks its behavior-policy proof."""
    if getattr(args, "policy_objective", "ppo") != "sao_dis":
        return None

    required = {
        "tokens",
        "response_lengths",
        "rewards",
        "loss_masks",
        "rollout_log_probs",
    }
    compaction = bool(getattr(args, "sao_compaction", False))
    if compaction:
        if not getattr(args, "calculate_per_token_loss", False):
            raise ValueError("SAO compaction requires token-level loss normalization")
        required.update(
            {
                "compaction_trajectory_ids",
                "compaction_segment_indices",
                "compaction_segment_types",
                "compaction_subsequent_active_tokens",
                "compaction_trajectory_active_tokens",
            }
        )
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"SAO rollout data is missing required fields: {', '.join(missing)}")

    sample_count = len(data["response_lengths"])
    if sample_count == 0:
        raise ValueError("SAO rollout data contains no samples")
    for key in required - {"response_lengths"}:
        if len(data[key]) != sample_count:
            raise ValueError(
                f"SAO rollout field {key} has {len(data[key])} rows; expected {sample_count}"
            )

    if compaction:
        for index, (
            trajectory_id,
            segment_index,
            segment_type,
            future_tokens,
            trajectory_tokens,
        ) in enumerate(
            zip(
                data["compaction_trajectory_ids"],
                data["compaction_segment_indices"],
                data["compaction_segment_types"],
                data["compaction_subsequent_active_tokens"],
                data["compaction_trajectory_active_tokens"],
                strict=True,
            )
        ):
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(f"SAO compaction sample {index} has no trajectory ID")
            if isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 0:
                raise ValueError(f"SAO compaction sample {index} has an invalid segment index")
            if segment_type not in {"execution", "summary"}:
                raise ValueError(f"SAO compaction sample {index} has an invalid segment type")
            if isinstance(future_tokens, bool) or not isinstance(future_tokens, int) or future_tokens < 0:
                raise ValueError(f"SAO compaction sample {index} has an invalid future-token count")
            if (
                isinstance(trajectory_tokens, bool)
                or not isinstance(trajectory_tokens, int)
                or trajectory_tokens <= 0
            ):
                raise ValueError(
                    f"SAO compaction sample {index} has an invalid trajectory-token count"
                )

    response_tokens = 0
    active_action_tokens = 0
    for index, (tokens, response_length, reward, loss_mask, rollout_log_probs) in enumerate(
        zip(
            data["tokens"],
            data["response_lengths"],
            data["rewards"],
            data["loss_masks"],
            data["rollout_log_probs"],
            strict=True,
        )
    ):
        if isinstance(response_length, bool) or not isinstance(response_length, int) or response_length <= 0:
            raise ValueError(f"SAO sample {index} has invalid response_length={response_length!r}")
        if len(tokens) < response_length:
            raise ValueError(
                f"SAO sample {index} has {len(tokens)} tokens but response_length={response_length}"
            )
        if len(loss_mask) != response_length:
            raise ValueError(
                f"SAO sample {index} loss-mask length {len(loss_mask)} != {response_length}"
            )
        if rollout_log_probs is None or len(rollout_log_probs) != response_length:
            actual = None if rollout_log_probs is None else len(rollout_log_probs)
            raise ValueError(
                f"SAO sample {index} rollout-logprob length {actual} != {response_length}"
            )
        if any(value not in {0, 1} for value in loss_mask):
            raise ValueError(f"SAO sample {index} loss mask must be binary")
        active = sum(bool(value) for value in loss_mask)
        if active == 0:
            raise ValueError(f"SAO sample {index} contains no trainable action tokens")
        try:
            finite_reward = math.isfinite(float(reward))
            finite_log_probs = all(math.isfinite(float(value)) for value in rollout_log_probs)
        except (TypeError, ValueError) as error:
            raise ValueError(f"SAO sample {index} has non-numeric reward or rollout logprobs") from error
        if not finite_reward:
            raise ValueError(f"SAO sample {index} has a non-finite reward")
        if not finite_log_probs:
            raise ValueError(f"SAO sample {index} has non-finite rollout logprobs")

        response_tokens += response_length
        active_action_tokens += active

    if compaction:
        rows_by_trajectory: dict[str, list[int]] = {}
        for row, trajectory_id in enumerate(data["compaction_trajectory_ids"]):
            rows_by_trajectory.setdefault(trajectory_id, []).append(row)
        for trajectory_id, rows in rows_by_trajectory.items():
            rows.sort(key=lambda row: data["compaction_segment_indices"][row])
            indices = [data["compaction_segment_indices"][row] for row in rows]
            segment_types = [data["compaction_segment_types"][row] for row in rows]
            expected_types = [
                "execution" if index % 2 == 0 else "summary" for index in indices
            ]
            if (
                indices != list(range(len(rows)))
                or segment_types != expected_types
                or segment_types[-1] != "execution"
            ):
                raise ValueError(
                    f"SAO compaction trajectory {trajectory_id!r} has an invalid segment sequence"
                )
            rewards = [float(data["rewards"][row]) for row in rows]
            if any(reward != rewards[0] for reward in rewards[1:]):
                raise ValueError(
                    f"SAO compaction trajectory {trajectory_id!r} does not share one task reward"
                )
            future_tokens = 0
            for row in reversed(rows):
                if data["compaction_subsequent_active_tokens"][row] != future_tokens:
                    raise ValueError(
                        f"SAO compaction trajectory {trajectory_id!r} has an invalid "
                        "future-token count"
                    )
                future_tokens += sum(bool(value) for value in data["loss_masks"][row])
            if any(
                data["compaction_trajectory_active_tokens"][row] != future_tokens
                for row in rows
            ):
                raise ValueError(
                    f"SAO compaction trajectory {trajectory_id!r} has an invalid "
                    "trajectory-token count"
                )

    return SAORolloutSummary(
        samples=sample_count,
        response_tokens=response_tokens,
        active_action_tokens=active_action_tokens,
    )


def apply_sao_online_recipe(args: Any) -> None:
    """Apply the paper-backed structural SAO settings for an online run.

    Batch size, topology, model paths, and SecRLEnv hooks remain deployment
    choices. ``coding`` and ``reasoning`` select the two DIS bands reported in
    the paper.
    """
    domain = getattr(args, "sao_online_recipe", None)
    if domain is None:
        return
    if getattr(args, "value_pretrain_manifest", None) is not None:
        raise ValueError("--sao-online-recipe is for online training, not offline value pretraining")
    if args.n_samples_per_prompt != 1:
        raise ValueError("SAO requires exactly one rollout per prompt")

    args.advantage_estimator = "gae_adaptive"
    args.policy_objective = "sao_dis"
    args.gae_adaptive_mode = "adaptive"
    args.gae_adaptive_alpha = 1.5
    args.gae_adaptive_min_length = 1
    args.gamma = 1.0
    args.critic_lambd = 1.0
    args.num_critic_epochs = 2
    args.critic_freeze_attention = True
    args.lr = 1e-6
    args.kl_coef = 0.0
    args.kl_loss_coef = 0.0
    args.use_kl_loss = False
    args.entropy_coef = 0.0
    args.critic_lr = 5e-6
    args.critic_lr_warmup_iters = 10
    if getattr(args, "sao_compaction", False):
        if getattr(args, "group_rm", False):
            raise ValueError("--sao-compaction requires one terminal reward per trajectory")
        if getattr(args, "generate_multi_samples", False):
            raise ValueError("--sao-compaction owns segment assembly; remove --generate-multi-samples")
        # Equation 12 in CompactionRL normalizes once across generated tokens,
        # rather than giving each variable-length segment equal weight.
        args.calculate_per_token_loss = True
        # Segment count varies by trajectory.  Train the complete collected
        # batch in one step so fixed sample-count trimming cannot cut a
        # trajectory at a compaction boundary.
        args.use_dynamic_global_batch_size = True

    if domain == "coding":
        args.sao_dis_eps_low = 0.8
        args.sao_dis_eps_high = 3.0
    elif domain == "reasoning":
        args.sao_dis_eps_low = 0.3
        args.sao_dis_eps_high = 5.0
    else:
        raise ValueError(f"unknown SAO online recipe: {domain}")


def freeze_attention_parameters(model: torch.nn.Module) -> FrozenParameterSummary:
    """Freeze parameters owned by attention modules, once per shared tensor."""
    parameters: dict[int, torch.nn.Parameter] = {}
    for module_name, module in model.named_modules():
        components = set(module_name.lower().split("."))
        if not components.intersection(_ATTENTION_COMPONENTS):
            continue
        for parameter in module.parameters(recurse=True):
            parameters[id(parameter)] = parameter

    if not parameters:
        raise RuntimeError("critic attention freeze matched no parameters")
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    return FrozenParameterSummary(
        tensors=len(parameters),
        elements=sum(parameter.numel() for parameter in parameters.values()),
    )
