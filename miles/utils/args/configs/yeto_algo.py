"""Fork-only algorithm options (michaellchung/miles yeto line).

Kept in their own trait, mixed into every leaf config in ``runtime.py`` next to ``AlgoConfig``
(like ``PartialExpertsConfig``), so a rebase onto upstream touches no upstream config file;
every field defaults to the upstream behaviour.
"""

from miles.utils.args.schema import A, Arg, BaseConfig


class YetoAlgoConfig(BaseConfig):
    policy_loss_variant: A[
        str,
        Arg(
            choices=["policy_loss", "cispo", "sapo", "gmpo"],
            help=(
                "Token-level policy objective used by --loss-type policy_loss. policy_loss = PPO clip (default). "
                "cispo reuses --eps-clip/--eps-clip-high as the IS-weight clip range [1-eps_clip, 1+eps_clip_high]; "
                "its lower bound therefore follows --eps-clip (default 0.2), whereas the CISPO paper essentially does "
                "not clip from below (pass a large --eps-clip, e.g. 1.0, to approximate that). The papers' token-level "
                "normalization needs --calculate-per-token-loss (default is per-sample mean). "
                "gmpo cannot be combined with --calculate-per-token-loss. gmpo reports gmpo_clip_num/gmpo_clip_den "
                "(global clipped fraction = their ratio)."
            ),
        ),
    ] = "policy_loss"
    sapo_tau_pos: A[float, Arg(help="SAPO gate temperature for A > 0")] = 1.0
    sapo_tau_neg: A[float, Arg(help="SAPO gate temperature for A <= 0")] = 1.05
    gmpo_log_clip_low: A[float, Arg(help="GMPO log-space clip: lower bound is -value")] = 0.4
    gmpo_log_clip_high: A[float, Arg(help="GMPO log-space clip: upper bound is +value")] = 0.4

    gae_variant: A[
        str,
        Arg(
            choices=["vanilla", "length_adaptive", "decoupled", "cross_segment_whole_rollout", "cross_segment_per_sample"],
            help=(
                "PPO GAE variant. cross_segment_per_sample (CompactionRL, one sample per compaction segment) runs "
                "local GAE per sample and multiplies the advantage by (gamma*lambda)^tokens_after from "
                "sample.metadata['tokens_after'] (optional metadata 'gae_length' sets l for length-adaptive lambda). "
                "cross_segment_whole_rollout is an explicit CONTROL mode (one sample per rollout with per-token "
                "sample.metadata['segment_ids']; earlier segments get no terminal reward, which differs from "
                "CompactionRL eq. 15). The ambiguous value cross_segment is rejected."
            ),
        ),
    ] = "vanilla"
    gae_lambd_mode: A[
        str,
        Arg(
            choices=["fixed", "length_adaptive"],
            help="Policy lambda: fixed (--lambd) or length_adaptive (1 - 1/(alpha * response_len)).",
        ),
    ] = "fixed"
    gae_length_alpha: A[float, Arg(help="alpha for length-adaptive lambda")] = 1.5
    gae_critic_lambd: A[float, Arg(help="critic-target lambda for --gae-variant decoupled")] = 1.0
    positive_example_lm_loss_coef: A[
        float,
        Arg(
            help=(
                "VAPO positive-example LM loss weight mu (paper eq. 9-10): adds mu * NLL over the response tokens "
                "of positive samples, normalized by the global (all micro-batches and DP ranks of the optimizer "
                "step) count of positive-sample tokens (0 disables)."
            )
        ),
    ] = 0.0
    positive_example_source: A[
        str,
        Arg(
            choices=["success", "reward"],
            help=(
                "How a sample is judged positive. success (default): the explicit boolean "
                "sample.metadata['success'] (or 'is_correct') set by the reward function; a missing field is an "
                "error. reward: reward > --positive-example-reward-threshold; only valid when the reward is a "
                "binary success signal (a positive reward always means complete success)."
            ),
        ),
    ] = "success"
    positive_example_reward_threshold: A[
        float | None,
        Arg(
            help=(
                "With --positive-example-source reward: a sample is positive when reward > this (default 0.0). "
                "Rejected with --positive-example-source success."
            )
        ),
    ] = None

    policy_objective: A[
        str,
        Arg(
            choices=["ppo", "sao_dis"],
            help="Policy-gradient objective. sao_dis uses current/rollout ratios and strict rejection.",
        ),
    ] = "ppo"
    sao_dis_eps_low: A[float, Arg(help="SAO DIS lower deviation; the strict lower ratio bound is 1-eps-low.")] = 0.3
    sao_dis_eps_high: A[float, Arg(help="SAO DIS upper deviation; the strict upper ratio bound is 1+eps-high.")] = 5.0
    value_loss_type: A[
        str,
        Arg(choices=["mse", "classification"], help="Scalar MSE or categorical value prediction."),
    ] = "mse"
    value_num_bins: A[int, Arg(help="Number of support bins for classification value prediction.")] = 51
    value_reward_range: A[
        tuple[float, float],
        Arg(type_parser=float, nargs=2, metavar=("LOW", "HIGH"), help="Categorical value support range."),
    ] = (0.0, 1.0)
    value_target_type: A[
        str,
        Arg(choices=["two_hot", "hl_gauss"], help="Projection from scalar returns to categorical targets."),
    ] = "hl_gauss"
    hl_gauss_sigma_ratio: A[float, Arg(help="HL-Gauss sigma divided by categorical bin width.")] = 0.75
    num_critic_epochs: A[
        int,
        Arg(
            aliases=("--critic-updates-per-step",),
            help=(
                "Critic optimizer updates per rollout batch (one actor step), reusing the fixed value targets. "
                "--critic-updates-per-step is an alias (CompactionRL: 2; SAO: --num-critic-epochs 2)."
            ),
        ),
    ] = 1
    critic_freeze_attention: A[
        bool,
        Arg(help="Freeze critic attention modules while keeping its value head and non-attention blocks trainable."),
    ] = False


_DEFAULTS = {name: field.default for name, field in YetoAlgoConfig.model_fields.items()}


def yeto_algo_option(args, name: str):
    """A YetoAlgoConfig field, or its upstream-preserving default on a hand-built namespace without it."""
    return getattr(args, name, _DEFAULTS[name])  # config-access-exempt: fork-only fields default when a hand-built namespace (upstream tests) omits them
