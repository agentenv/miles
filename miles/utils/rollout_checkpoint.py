"""Which HF checkpoint the rollout engines start from (no heavy imports)."""


def rollout_hf_checkpoint(args) -> str:
    """``--rollout-hf-checkpoint`` when set (e.g. an FP8 copy of the BF16 trainer checkpoint), else ``--hf-checkpoint``."""
    return getattr(args, "rollout_hf_checkpoint", None) or args.hf_checkpoint
