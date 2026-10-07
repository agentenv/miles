"""SAO faster value update (ported from agentenv/miles SAO 16a9bea4); megatron-free for CPU tests."""


def refresh_values_after_critic_update(args, rollout_id: int) -> bool:
    """True when the critic must re-predict values after its update(s) for the actor.

    Upstream SAO (``train_critic``, policy_objective == "sao_dis"): the actor consumes
    values from the just-updated critic, not the stale predictions used to build this
    rollout's fixed value targets. Critic-only warmup steps train no actor, so no refresh.
    """
    if getattr(args, "policy_objective", "ppo") != "sao_dis":
        return False
    return rollout_id >= getattr(args, "num_critic_only_steps", 0)
