from argparse import Namespace
from collections.abc import Callable
from typing import Protocol

import torch

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    get_local_response_loss_masks,
    get_sum_of_sample_mean,
)
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss_hub.logit_processors import (
    _value_support,
    get_log_probs_and_entropy,
    get_responses,
    get_values,
)
from miles.backends.training_utils.loss_hub.math_utils import (
    compute_approx_kl,
    compute_ess_ratio_contribution,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    compute_sao_dis_policy_loss,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.misc import load_function
from miles.utils.types import RolloutBatch


class LossFunction(Protocol):
    """Common signature of the per-loss-type functions dispatched by `get_loss_function`."""

    def __call__(
        self,
        args: Namespace,
        batch: RolloutBatch,
        logits: torch.Tensor,
        sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute a scalar loss with gradient + a dict of detached scalar metrics.

        Args:
            args: Configuration. Each loss type reads its own subset of flags
                (PPO eps, KL coefficients, value clip, TIS settings, OPSM, etc.).
            batch: Mini-batch (`miles.utils.types.RolloutBatch`). Loss-type-
                dependent keys; common ones are `unconcat_tokens`, `response_lengths`,
                `total_lengths`, `loss_masks`, optional `max_seq_lens` (required for
                qkv_format="bshd"). Per-implementation docstrings list extras.
            logits: Float32. Last dim is vocab_size (policy/sft) or 1 (value).
                Outer shape is `[1, T, ...]` for qkv_format="thd" (T = sum of
                total_lengths) or `[B, max_seq_len, ...]` for "bshd".
            sum_of_sample_mean: CP-aware reducer; takes a flat per-token tensor
                and returns a scalar, sample-mean-weighted by `loss_masks`.

        Returns:
            `(loss, metrics)`:
              * `loss`: scalar tensor with grad, un-rescaled (the dispatcher
                applies Megatron scaling on top).
              * `metrics`: dict of detached 0-d scalars; surfaced under `train/`
                in the training log / wandb. Keys per loss type are documented
                on each implementation.
        """
        ...


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    policy_objective = getattr(args, "policy_objective", "ppo")
    if policy_objective == "sao_dis":
        if not batch.get("rollout_log_probs"):
            raise ValueError("SAO DIS requires per-token rollout_log_probs")
        old_log_probs = batch["rollout_log_probs"]
    else:
        old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=calculate_entropy,
        entropy_requires_grad=args.entropy_coef != 0,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    train_log_probs_list = log_probs
    old_log_probs_list = old_log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    local_loss_mask_list = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    local_loss_masks = torch.cat(local_loss_mask_list, dim=0).to(device=ppo_kl.device)
    active_tokens = local_loss_masks.bool()
    ppo_kl = torch.where(
        active_tokens,
        torch.nan_to_num(ppo_kl, nan=0.0, posinf=0.0, neginf=0.0),
        ppo_kl.new_zeros(()),
    )
    advantages = torch.where(
        active_tokens,
        torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
        advantages.new_zeros(()),
    )

    dis_ratio = dis_below = dis_above = None
    if policy_objective == "sao_dis":
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        pg_loss, dis_ratio, pg_clipfrac, dis_below, dis_above = compute_sao_dis_policy_loss(
            log_probs,
            rollout_log_probs,
            advantages,
            args.sao_dis_eps_low,
            args.sao_dis_eps_high,
        )
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(
            ppo_kl, advantages, args.eps_clip, args.eps_clip_high, getattr(args, "eps_clip_c", None)
        )

    if getattr(args, "dump_details", None) is not None:
        from miles.backends.training_utils.debug_dump import maybe_dump_policy_loss_debug

        maybe_dump_policy_loss_debug(
            args=args,
            batch=batch,
            train_log_probs=train_log_probs_list,
            old_log_probs=old_log_probs_list,
            rollout_log_probs=batch.get("rollout_log_probs"),
            advantages=batch["advantages"],
            local_loss_masks=local_loss_mask_list,
            ppo_kl=ppo_kl,
            pg_loss=pg_loss,
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if args.custom_pg_loss_reducer_function_path is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    # ESS (Effective Sample Size) ratio from per-token IS weights
    # w = π_new/π_old = exp(-ppo_kl).  A value of 1.0 is on-policy; near 0
    # means the per-token weights are highly concentrated.
    ess_ratio_sum = compute_ess_ratio_contribution(
        ppo_kl=ppo_kl,
        loss_masks=batch["loss_masks"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy_loss = pg_loss.new_zeros(())
    loss = pg_loss
    if calculate_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
        if args.entropy_coef != 0:
            loss = pg_loss - args.entropy_coef * entropy_loss
        else:
            entropy_loss = entropy_loss.detach()

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl = torch.where(
            active_tokens,
            torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
            kl.new_zeros(()),
        )
        kl_loss = sum_of_sample_mean(kl)

        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_scored_log_probs = old_log_probs
    train_rollout_logprob_abs_diff = None
    train_rollout_kl = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
        abs_diff = torch.where(
            active_tokens,
            torch.nan_to_num(abs_diff, nan=0.0, posinf=0.0, neginf=0.0),
            abs_diff.new_zeros(()),
        )
        train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

        # KL(rollout || train) at sampled tokens via Schulman k3 with per-token clamp [-10, 10]
        rollout_train_kl = compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
        rollout_train_kl = torch.where(
            active_tokens,
            torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
            rollout_train_kl.new_zeros(()),
        )
        train_rollout_kl = sum_of_sample_mean(rollout_train_kl)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ess_ratio": ess_ratio_sum.squeeze(),
    }
    if policy_objective == "sao_dis":
        reported_loss.update(
            {
                "sao_dis_ratio": sum_of_sample_mean(dis_ratio).clone().detach(),
                "sao_dis_rejectfrac": pg_clipfrac.clone().detach(),
                "sao_dis_low_rejectfrac": sum_of_sample_mean(dis_below).clone().detach(),
                "sao_dis_high_rejectfrac": sum_of_sample_mean(dis_above).clone().detach(),
            }
        )

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if train_rollout_kl is not None:
        reported_loss["train_rollout_kl"] = train_rollout_kl.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if batch.get("opd_reverse_kl") is not None:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def _two_hot_target_distribution(targets: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Linearly project scalar targets onto their two adjacent support bins."""
    num_bins = support.shape[0]
    low, high = support[0].item(), support[-1].item()
    positions = (targets.float().clamp(low, high) - low) / (high - low) * (num_bins - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    upper_weight = positions - lower.float()
    distribution = torch.zeros((*targets.shape, num_bins), dtype=torch.float32, device=targets.device)
    distribution.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def _hl_gauss_target_distribution(
    targets: torch.Tensor,
    support: torch.Tensor,
    sigma_ratio: float = 0.75,
) -> torch.Tensor:
    """Project scalar targets using Gaussian CDF mass over each support bin."""
    bin_width = support[1] - support[0]
    sigma = sigma_ratio * bin_width
    bin_lo = support - bin_width / 2
    bin_hi = support + bin_width / 2
    mu = targets.float().clamp(support[0], support[-1]).unsqueeze(-1)
    normalizer = sigma * 1.4142135623730951
    probs = 0.5 * (torch.erf((bin_hi - mu) / normalizer) - torch.erf((bin_lo - mu) / normalizer))
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _get_value_logits_list(logits: torch.Tensor, args: Namespace, batch: RolloutBatch) -> list[torch.Tensor]:
    return [
        logits_chunk
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
            apply_temperature=False,
        )
    ]


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute online PPO value loss or direct offline value-pretraining loss."""
    returns = torch.cat(batch["returns"], dim=0)
    use_classification = getattr(args, "value_loss_type", "mse") == "classification"
    offline_pretraining = bool(getattr(args, "value_pretrain_manifest", None))

    if use_classification:
        num_bins = args.value_num_bins
        raw_logits = torch.cat(
            [chunk.reshape(-1, num_bins) for chunk in _get_value_logits_list(logits, args, batch)],
            dim=0,
        )
        support = _value_support(args, raw_logits.device)
        values = raw_logits.float().softmax(dim=-1) @ support
        if getattr(args, "value_target_type", "hl_gauss") == "hl_gauss":
            target_dist = _hl_gauss_target_distribution(
                returns,
                support,
                getattr(args, "hl_gauss_sigma_ratio", 0.75),
            )
        else:
            target_dist = _two_hot_target_distribution(returns, support)
        log_probs = torch.nn.functional.log_softmax(raw_logits.float(), dim=-1)
        loss = sum_of_sample_mean(-(target_dist * log_probs).sum(dim=-1))
        if values.numel() == 0:
            loss += 0 * raw_logits.sum()
        probs = log_probs.exp()
        reported_loss = {
            "value_loss": loss.clone().detach(),
            "value_accuracy": sum_of_sample_mean(
                (raw_logits.argmax(dim=-1) == target_dist.argmax(dim=-1)).float()
            ).detach(),
            "value_entropy": sum_of_sample_mean(-(probs * log_probs).sum(dim=-1)).detach(),
            "value_confidence": sum_of_sample_mean(probs.max(dim=-1).values).detach(),
        }
    else:
        values_by_sample = get_values(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        values = torch.cat([value.flatten() for value in values_by_sample["values"]], dim=0)
        if offline_pretraining:
            loss = sum_of_sample_mean((values - returns) ** 2)
            reported_loss = {"value_loss": loss.clone().detach()}
        else:
            old_values = torch.cat(batch["values"], dim=0)
            values_clipfrac = torch.abs(values - old_values) > args.value_clip
            values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
            loss = sum_of_sample_mean(torch.max((values_clipped - returns) ** 2, (values - returns) ** 2))
            reported_loss = {
                "value_loss": loss.clone().detach(),
                "value_clipfrac": sum_of_sample_mean(values_clipfrac.float()).detach(),
            }
        if values.numel() == 0:
            loss += 0 * values.sum()

    # Carry sufficient statistics through the existing distributed metric
    # reduction so explained variance is globally correct and mask-aware.
    with torch.no_grad():
        if values.numel() > 0:
            local_loss_masks = get_local_response_loss_masks(
                batch["total_lengths"],
                batch["response_lengths"],
                batch["loss_masks"],
                args.qkv_format,
                batch.get("max_seq_lens", None),
            )
            active = torch.cat(local_loss_masks, dim=0).to(device=values.device).bool()
            ev_returns = returns[active]
            ev_residual = ev_returns - values[active]
            reported_loss["_ev_n"] = torch.tensor(float(ev_returns.numel()), device=values.device)
            reported_loss["_ev_returns_sum"] = ev_returns.sum()
            reported_loss["_ev_returns_sq_sum"] = ev_returns.square().sum()
            reported_loss["_ev_residual_sum"] = ev_residual.sum()
            reported_loss["_ev_residual_sq_sum"] = ev_residual.square().sum()

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def get_loss_function(args: Namespace) -> LossFunction:
    match args.loss_type:
        case "policy_loss":
            return policy_loss_function
        case "value_loss":
            return value_loss_function
        case "sft_loss":
            return sft_loss_function
        case "custom_loss":
            return load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
