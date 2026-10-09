"""CPU tests for the dual_clipfrac observation metric (yeto rl-algo-supplement 2.3).

dual_clipfrac = share of loss-masked tokens with A < 0 where the dual-clip lower bound -c*A
is the active term of the PPO loss.  It is observation only: the loss must stay bitwise unchanged.
"""

import torch

from miles.backends.training_utils.loss_hub.math_utils import compute_dual_clipfrac, compute_policy_loss

from .test_policy_loss_variants import _inputs, _run, make_args


def test_dual_clipfrac_hand_values():
    # ratio r = exp(-ppo_kl); c = 3, eps = 0.2
    r = torch.tensor([4.0, 2.0, 4.0, 0.5, 5.0])
    adv = torch.tensor([-1.0, -1.0, 1.0, -1.0, 0.0])
    ppo_kl = -torch.log(r)
    got = compute_dual_clipfrac(ppo_kl, adv, 0.2, 0.2, 3.0)
    # token 0: A<0, max(r*1, clip(r)*1)=4 > c=3 -> dual active; token 1: 2 < 3 -> not;
    # token 2: A>0 -> not; token 3: 0.8 < 3 -> not; token 4: A=0 -> not
    assert got.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_dual_clipfrac_matches_where_loss_takes_dual_branch():
    g = torch.Generator().manual_seed(3)
    ppo_kl = torch.randn(2000, generator=g) * 1.5
    adv = torch.randn(2000, generator=g)
    c = 1.5
    eager = compute_policy_loss.__wrapped__
    loss_dual, _ = eager(ppo_kl, adv, 0.2, 0.28, c)
    loss_plain, _ = eager(ppo_kl, adv, 0.2, 0.28, None)
    frac = compute_dual_clipfrac(ppo_kl, adv, 0.2, 0.28, c)
    changed = (loss_dual != loss_plain).float()
    assert torch.equal(frac, changed)
    assert 0 < frac.mean().item() < 1


def _dual_inputs(args):
    inputs = _inputs(args)
    g = torch.Generator().manual_seed(21)
    # wide ratio spread so some A < 0 tokens exceed c
    inputs["log_probs"] = [x - 1.5 * torch.rand(x.shape, generator=g) for x in inputs["log_probs"]]
    return inputs


def test_policy_loss_function_reports_dual_clipfrac_without_changing_loss():
    args = make_args(entropy_coef=0.0, eps_clip=0.2, eps_clip_high=0.28, eps_clip_c=1.5)
    inputs = _dual_inputs(args)
    loss_a, metrics_a, grad_a = _run(args, inputs)
    assert "dual_clipfrac" in metrics_a
    v = metrics_a["dual_clipfrac"]
    assert torch.isfinite(v) and v.item() > 0
    # the metric is detached and computing it does not touch the loss or gradient
    from miles.backends.training_utils.loss_hub import losses as losses_mod

    saved = losses_mod.compute_dual_clipfrac
    try:
        losses_mod.compute_dual_clipfrac = lambda *a, **k: torch.zeros_like(a[0])
        loss_b, metrics_b, grad_b = _run(args, inputs)
    finally:
        losses_mod.compute_dual_clipfrac = saved
    assert torch.equal(loss_a, loss_b) and torch.equal(grad_a, grad_b)
    for k in metrics_a:
        if k != "dual_clipfrac":
            assert torch.equal(metrics_a[k], metrics_b[k]), k


def test_dual_clipfrac_reduction_matches_hand():
    """Reported value = sum over samples of the masked per-sample mean (same reducer as pg_clipfrac)."""
    args = make_args(entropy_coef=0.0, eps_clip=0.2, eps_clip_high=0.28, eps_clip_c=1.5)
    inputs = _dual_inputs(args)
    _, metrics, _ = _run(args, inputs)
    from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

    from .loss_test_utils import make_batch, make_parallel_state

    make_parallel_state()
    batch = make_batch(inputs, "policy_loss")
    lps = get_log_probs_and_entropy(
        inputs["policy_logits"],
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )["log_probs"]
    want = 0.0
    for lp, old, adv, m in zip(lps, batch["log_probs"], batch["advantages"], batch["loss_masks"], strict=True):
        r = torch.exp(lp.detach() - old)
        l12 = torch.maximum(-r * adv, -r.clamp(0.8, 1.28) * adv)
        tok = ((adv < 0) & (-1.5 * adv < l12)).float()
        want += ((tok * m).sum() / torch.clamp_min(m.sum(), 1)).item()
    assert abs(metrics["dual_clipfrac"].item() - want) < 1e-6


def test_default_path_has_no_dual_clipfrac_key():
    args = make_args(entropy_coef=0.0, eps_clip=0.2, eps_clip_high=0.28)
    _, metrics, _ = _run(args, _inputs(args))
    assert "dual_clipfrac" not in metrics
