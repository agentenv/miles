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


_DEFAULTS = {name: field.default for name, field in YetoAlgoConfig.model_fields.items()}


def yeto_algo_option(args, name: str):
    """A YetoAlgoConfig field, or its upstream-preserving default on a hand-built namespace without it."""
    return getattr(args, name, _DEFAULTS[name])  # config-access-exempt: fork-only fields default when a hand-built namespace (upstream tests) omits them
