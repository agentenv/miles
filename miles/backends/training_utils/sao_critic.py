"""SAO faster value update (ported from agentenv/miles SAO 16a9bea4); megatron-free for CPU tests."""

from miles.utils.args.configs.yeto_algo import yeto_algo_option


def refresh_values_after_critic_update(args, rollout_id: int) -> bool:
    """True when the critic must re-predict values after its update(s) for the actor.

    Upstream SAO (``train_critic``, policy_objective == "sao_dis"): the actor consumes
    values from the just-updated critic, not the stale predictions used to build this
    rollout's fixed value targets. Critic-only warmup steps train no actor, so no refresh.
    """
    if yeto_algo_option(args, "policy_objective") != "sao_dis":
        return False
    return rollout_id >= args.num_critic_only_steps
