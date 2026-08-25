"""SAO's length-adaptive, observation-skipping GAE estimator.

Only tokens selected by ``loss_mask`` participate in the backwards recursion.
Environment observations are therefore transparent: the action before an
observation bootstraps directly from the value of the next generated action.
"""

from __future__ import annotations

import torch

from miles.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.math_utils import chunked_gae
from miles.backends.training_utils.parallel import get_parallel_state


def compute_adaptive_lambd(effective_length: int, alpha: float, min_length: int = 1) -> float:
    """Return ``1 - 1 / (alpha * length)`` with safe short-sequence limits."""
    if effective_length < 0:
        raise ValueError("effective_length must be non-negative")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_length < 0:
        raise ValueError("min_length must be non-negative")
    if effective_length < min_length or effective_length == 0:
        return 1.0
    return max(1.0 - 1.0 / (alpha * effective_length), 0.0)


def _gae_loop(
    token_rewards: torch.Tensor,
    token_values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GAE across active action tokens, skipping all inactive positions."""
    if token_rewards.ndim != 1 or token_values.ndim != 1 or loss_mask.ndim != 1:
        raise ValueError("rewards, values, and loss_mask must be one-dimensional")
    if not (token_rewards.shape == token_values.shape == loss_mask.shape):
        raise ValueError("rewards, values, and loss_mask must have identical shapes")

    advantages = torch.zeros_like(token_rewards, dtype=torch.float64)
    returns = torch.zeros_like(token_rewards, dtype=torch.float64)
    last_gae = 0.0
    next_value = 0.0

    # Float64 accumulation matters for long agent trajectories. Outputs retain
    # the caller's dtype so the rest of Miles' data path remains unchanged.
    rewards64 = token_rewards.to(torch.float64)
    values64 = token_values.to(torch.float64)
    for index in range(token_rewards.numel() - 1, -1, -1):
        if bool(loss_mask[index]):
            delta = rewards64[index] + gamma * next_value - values64[index]
            last_gae = delta + gamma * lambd * last_gae
            advantages[index] = last_gae
            returns[index] = last_gae + values64[index]
            next_value = values64[index]

    return advantages.to(token_rewards.dtype), returns.to(token_rewards.dtype)


def _gae_chunked(
    token_rewards: torch.Tensor,
    token_values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress inactive tokens, run the parallel scan, and scatter back."""
    active = loss_mask.bool()
    active_count = int(active.sum().item())
    if active_count == 0:
        return torch.zeros_like(token_rewards), torch.zeros_like(token_rewards)

    advantages, returns = chunked_gae(
        token_rewards[active].unsqueeze(0).to(torch.float64),
        token_values[active].unsqueeze(0).to(torch.float64),
        gamma=gamma,
        lambd=lambd,
        chunk_size=min(chunk_size, active_count),
    )
    full_advantages = torch.zeros_like(token_rewards)
    full_returns = torch.zeros_like(token_rewards)
    full_advantages[active] = advantages.squeeze(0).to(token_rewards.dtype)
    full_returns[active] = returns.squeeze(0).to(token_rewards.dtype)
    return full_advantages, full_returns


def get_gae_adaptive_advantages(
    *,
    rewards: list[float],
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    values: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float = 0.0,
    gamma: float = 1.0,
    lambd: float = 1.0,
    mode: str = "adaptive",
    alpha: float = 1.5,
    min_length: int = 1,
    chunked_threshold: int = 8192,
    chunk_size: int = 128,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-token advantages and returns for single-rollout SAO."""
    if mode not in {"fixed", "adaptive"}:
        raise ValueError(f"unknown GAE mode: {mode}")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not 0 <= lambd <= 1:
        raise ValueError("lambd must be in [0, 1]")
    if chunked_threshold < 1 or chunk_size < 1:
        raise ValueError("chunked_threshold and chunk_size must be positive")

    batch_size = len(rewards)
    fields = (kl, loss_masks, values, response_lengths, total_lengths)
    if any(len(field) != batch_size for field in fields):
        raise ValueError("all adaptive GAE inputs must have the same batch size")

    try:
        cp_size = get_parallel_state().cp.size
    except (AssertionError, AttributeError):
        cp_size = 1
    if cp_size > 1 and qkv_format == "bshd" and max_seq_lens is None:
        raise ValueError("max_seq_lens is required for BSHD context parallelism")
    padded_lengths = max_seq_lens if max_seq_lens is not None else [None] * batch_size

    advantages_list: list[torch.Tensor] = []
    returns_list: list[torch.Tensor] = []
    for index in range(batch_size):
        max_seq_len = padded_lengths[index]
        if cp_size > 1:
            full_kl = all_gather_with_cp(
                kl[index],
                total_lengths[index],
                response_lengths[index],
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
            )
            full_values = all_gather_with_cp(
                values[index],
                total_lengths[index],
                response_lengths[index],
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
            )
        else:
            full_kl = kl[index]
            full_values = values[index]

        full_mask = loss_masks[index].to(device=full_values.device)
        if full_mask.numel() != response_lengths[index]:
            raise ValueError(
                f"loss mask length mismatch for sample {index}: "
                f"{full_mask.numel()} != {response_lengths[index]}"
            )
        if full_kl.shape != full_values.shape or full_values.numel() != response_lengths[index]:
            raise ValueError(f"KL/value length mismatch for sample {index}")

        effective_length = int(full_mask.bool().sum().item())
        sample_lambd = (
            compute_adaptive_lambd(effective_length, alpha, min_length)
            if mode == "adaptive"
            else lambd
        )
        token_rewards = -kl_coef * full_kl.float()
        active_indices = full_mask.nonzero(as_tuple=True)[0]
        if active_indices.numel() > 0:
            token_rewards[active_indices[-1]] += float(rewards[index])

        if effective_length >= chunked_threshold:
            advantages, returns = _gae_chunked(
                token_rewards,
                full_values,
                full_mask,
                gamma,
                sample_lambd,
                chunk_size,
            )
        else:
            advantages, returns = _gae_loop(
                token_rewards,
                full_values,
                full_mask,
                gamma,
                sample_lambd,
            )

        if cp_size > 1:
            advantages = slice_log_prob_with_cp(
                advantages,
                total_lengths[index],
                response_lengths[index],
                qkv_format=qkv_format,
                max_token_len=max_seq_len,
            )
            returns = slice_log_prob_with_cp(
                returns,
                total_lengths[index],
                response_lengths[index],
                qkv_format=qkv_format,
                max_token_len=max_seq_len,
            )
        advantages_list.append(advantages.detach())
        returns_list.append(returns.detach())

    return advantages_list, returns_list
