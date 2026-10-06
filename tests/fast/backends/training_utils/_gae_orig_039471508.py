"""Frozen pre-change GAE (yeto/ports 039471508, math_utils.py) for the default-unchanged test."""
import torch
import torch.nn.functional as F
from miles.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.parallel import get_parallel_state


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    terminal_rewards,
    qkv_format,
    max_seq_lens,
    loss_masks,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support, computed over trainable tokens only.

    Semantics:
      - Masked tokens (`loss_mask == 0`, e.g. tool/env observations in
        multi-turn rollouts) are not MDP transitions. GAE runs on the
        subsequence of trainable tokens, so masked tokens carry no reward
        (including KL shaping), contribute no value delta, and the GAE carry
        crosses them without extra `gamma * lambd` decay.
      - The terminal reward is added at the last trainable token, not the last
        response token.
      - Fully masked samples get zero advantages and returns; their terminal
        reward is dropped.
      - Truncated sequences use the same zero bootstrap as terminated ones:
        the value after the last trainable token is taken as 0 and the
        observed terminal reward is still applied.
      - This function outputs zero advantages and returns at masked positions.
        Downstream transforms may still shift these entries to nonzero values
        (advantage whitening applies its affine transform to every position,
        and the on-policy distillation KL penalty is added per token), but the
        whitening statistics themselves are mask-weighted, so the injected
        zeros do not bias them. Correctness relies on the policy and value
        losses masking these positions out (the policy loss re-zeros
        advantages at inactive tokens and all loss reducers weight by
        `loss_mask`), so masked positions never receive gradient.

    C_i is the length of values_list[i] and rewards_list[i] on the current CP rank.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each current-CP-rank tensor has shape [C_i]
        rewards_list:      list[Tensor], same shape as values_list
        terminal_rewards:  list[float], one scalar sequence reward per sample
        qkv_format:        str, sequence layout used to split tensors across CP ranks
        max_seq_lens:      list[int] of BSHD padded lengths, or None for THD
        loss_masks:        list[Tensor], full-response masks, each has shape [R_i]
    Output:
        advantages_list:   list[Tensor], each current-CP-rank tensor has shape [C_i]
        returns_list:      list[Tensor], same shape
    """

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)
        assert B == len(terminal_rewards)
        assert B == len(loss_masks)

        cp_size = get_parallel_state().cp.size
        if cp_size > 1 and qkv_format == "bshd":
            assert max_seq_lens is not None, "max_seq_lens is required for BSHD with CP"
            assert B == len(max_seq_lens)
            max_seq_lens_per_sample = max_seq_lens
        else:  # qkv_format == "thd" or cp_size == 1
            max_seq_lens_per_sample = [None] * B

        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r, max_seq_len in zip(
                total_lengths,
                response_lengths,
                values_list,
                rewards_list,
                max_seq_lens_per_sample,
                strict=False,
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len, qkv_format=qkv_format, max_seq_len=max_seq_len)
                full_r = all_gather_with_cp(r, total_len, resp_len, qkv_format=qkv_format, max_seq_len=max_seq_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [resp_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # Compress each sample to its trainable positions so that masked
        # tokens do not act as MDP transitions in the GAE recursion.
        trainable_indices = [
            loss_masks[i][: response_lengths[i]].to(device).nonzero(as_tuple=True)[0] for i in range(B)
        ]
        trainable_lengths = [idx.numel() for idx in trainable_indices]

        # pad to max_len for batched GAE
        max_len = max(trainable_lengths)

        packed_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        packed_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            K = trainable_lengths[i]
            if K > 0:
                idx = trainable_indices[i]
                packed_values[i, :K] = full_values_list[i][idx]
                packed_rewards[i, :K] = full_rewards_list[i][idx]
                packed_rewards[i, K - 1] += terminal_rewards[i]

        if max_len == 0:
            packed_advantages = torch.zeros(B, 0, device=device, dtype=dtype)
            packed_returns = torch.zeros(B, 0, device=device, dtype=dtype)
        elif not chunked:
            packed_advantages, packed_returns = vanilla_gae(
                rewards=packed_rewards,
                values=packed_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            packed_advantages, packed_returns = chunked_gae(
                rewards=packed_rewards,
                values=packed_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        for i in range(B):
            resp_len = response_lengths[i]
            K = trainable_lengths[i]

            adv_full = torch.zeros(resp_len, device=device, dtype=dtype)
            ret_full = torch.zeros(resp_len, device=device, dtype=dtype)
            if K > 0:
                idx = trainable_indices[i]
                adv_full[idx] = packed_advantages[i, :K]
                ret_full[idx] = packed_returns[i, :K]

            if cp_size > 1:
                max_seq_len = max_seq_lens_per_sample[i]
                adv_full = slice_log_prob_with_cp(
                    adv_full,
                    total_lengths[i],
                    resp_len,
                    qkv_format=qkv_format,
                    max_token_len=max_seq_len,
                )
                ret_full = slice_log_prob_with_cp(
                    ret_full,
                    total_lengths[i],
                    resp_len,
                    qkv_format=qkv_format,
                    max_token_len=max_seq_len,
                )

            advantages_list.append(adv_full)
            returns_list.append(ret_full)

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns

