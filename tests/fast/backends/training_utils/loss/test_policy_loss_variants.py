"""CPU tests for --policy-loss-variant (cispo / sapo / gmpo).

Reference implementations below are written independently from the formulas
(they do not import the loss_hub functions under test):
  - CISPO (MiniMax-M1, arXiv:2506.13585, eq. 4): -sg(clip(r, 1-eps_l, 1+eps_h)) * A * log pi
  - SAPO (Qwen, arXiv:2511.20347): -sigmoid(tau * (r - 1)) * 4 / tau * A, tau by sign(A)
  - GMPO (arXiv:2507.20673v3 eq. 4, Algorithm 1): seq ratio = exp(mean_t l_t), where
    l_t = min(log r_t, d_h) for A > 0 and max(log r_t, -d_l) for A < 0 (PPO-style one-sided clip)
"""

import argparse
import math

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import (
    compute_cispo_loss,
    compute_gmpo_loss,
    compute_sapo_loss,
)
from miles.utils.arguments import get_miles_extra_args_provider, validate_policy_loss_variant_args

from .loss_test_utils import deep_clone, make_batch, make_inputs, make_parallel_state
from .loss_test_utils import make_args as _make_args

# policy_loss_function reads a few flags that the shared snapshot defaults do not set.
_EXTRA = dict(observe_training_entropy=False)


def make_args(**kw):
    return _make_args(**{**_EXTRA, **kw})


cispo_eager = compute_cispo_loss.__wrapped__
sapo_eager = compute_sapo_loss.__wrapped__


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------


def ref_cispo(log_probs, old_log_probs, adv, eps_l, eps_h):
    r = torch.exp(log_probs.detach() - old_log_probs)
    w = torch.clamp(r, 1 - eps_l, 1 + eps_h)
    return -w * adv * log_probs


def ref_sapo(log_probs, old_log_probs, adv, tau_pos, tau_neg):
    r = torch.exp(log_probs - old_log_probs)
    out = []
    for ri, ai in zip(r, adv, strict=True):
        tau = tau_pos if ai > 0 else tau_neg
        out.append(-torch.sigmoid(tau * (ri - 1)) * 4 / tau * ai)
    return torch.stack(out)


def ref_gmpo_seq(log_probs, old_log_probs, adv, mask, d_l, d_h):
    terms = []
    for lp, olp, a, m in zip(log_probs, old_log_probs, adv, mask, strict=True):
        if m == 0:
            continue
        lr = lp - olp
        if a > 0:  # one-sided: only the upper bound applies
            terms.append(torch.clamp(lr, max=d_h))
        elif a < 0:  # only the lower bound applies
            terms.append(torch.clamp(lr, min=-d_l))
        else:
            terms.append(lr * 0)
    if not terms:
        return torch.zeros(()) * log_probs.sum()
    return torch.exp(torch.stack(terms).mean())


# ---------------------------------------------------------------------------
# Token-level functions vs references
# ---------------------------------------------------------------------------


def test_cispo_hand_values_and_gradient():
    old = torch.zeros(4)
    lp = torch.tensor([math.log(1.5), math.log(0.5), math.log(1.1), math.log(0.9)], requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.5, 0.0])
    loss, clipfrac = cispo_eager(old - lp, lp, adv, 0.2, 0.28)
    w = torch.tensor([1.28, 0.8, 1.1, 0.9])
    torch.testing.assert_close(loss, -w * adv * lp.detach())
    torch.testing.assert_close(clipfrac, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    loss.sum().backward()
    # every token keeps a gradient of -w * A (stop-gradient on the clipped weight)
    torch.testing.assert_close(lp.grad, -w * adv)


def test_cispo_matches_reference_random():
    g = torch.Generator().manual_seed(0)
    old = torch.randn(64, generator=g) - 2
    lp = (old + 0.5 * torch.randn(64, generator=g)).requires_grad_(True)
    adv = torch.randn(64, generator=g)
    loss, _ = cispo_eager(old - lp, lp, adv, 0.2, 0.28)
    lp2 = lp.detach().clone().requires_grad_(True)
    ref = ref_cispo(lp2, old, adv, 0.2, 0.28)
    torch.testing.assert_close(loss, ref)
    loss.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(lp.grad, lp2.grad)


def test_sapo_hand_values_and_gradient():
    # at r == 1 the gate is 2/tau and d gate / dr = 1 (same local gradient as PPO)
    lp = torch.zeros(3, requires_grad=True)
    adv = torch.tensor([1.0, -1.0, 0.0])
    loss, clipfrac = sapo_eager(-lp, adv, 1.0, 1.05)
    torch.testing.assert_close(loss, torch.tensor([-2.0, 2 / 1.05, 0.0]))
    torch.testing.assert_close(clipfrac, torch.zeros(3))
    loss.sum().backward()
    torch.testing.assert_close(lp.grad, -adv)


def test_sapo_matches_reference_random():
    g = torch.Generator().manual_seed(1)
    old = torch.randn(64, generator=g) - 2
    lp = (old + 0.5 * torch.randn(64, generator=g)).requires_grad_(True)
    adv = torch.randn(64, generator=g)
    loss, _ = sapo_eager(old - lp, adv, 0.7, 1.3)
    lp2 = lp.detach().clone().requires_grad_(True)
    ref = ref_sapo(lp2, old, adv, 0.7, 1.3)
    torch.testing.assert_close(loss, ref)
    loss.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(lp.grad, lp2.grad)


def test_gmpo_matches_reference_with_mask_and_clip():
    g = torch.Generator().manual_seed(2)
    lens = [5, 7, 3]
    olds = [torch.randn(n, generator=g) - 2 for n in lens]
    lps = [(o + torch.randn(o.shape, generator=g)).requires_grad_(True) for o in olds]
    advs = [torch.full((5,), 1.5), torch.full((7,), -0.7), torch.zeros(3)]
    masks = [torch.tensor([1.0, 1, 0, 1, 1]), torch.ones(7), torch.ones(3)]
    loss, clipfrac = compute_gmpo_loss(lps, olds, advs, masks, advs, 0.4, 0.3)

    lps2 = [lp.detach().clone().requires_grad_(True) for lp in lps]
    ref = torch.cat(
        [-ref_gmpo_seq(lp, o, a, m, 0.4, 0.3) * a for lp, o, a, m in zip(lps2, olds, advs, masks, strict=True)]
    )
    torch.testing.assert_close(loss, ref)
    loss.sum().backward()
    ref.sum().backward()
    for a, b in zip(lps, lps2, strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    # masked token gets no gradient; zero-advantage sequence contributes nothing
    assert lps[0].grad[2].item() == 0.0
    assert (clipfrac >= 0).all() and (clipfrac <= 1).all()


def test_gmpo_all_clipped_sequence_has_no_gradient_and_clipfrac_one():
    old = [torch.zeros(4)]
    lp = [torch.full((4,), 2.0, requires_grad=True)]  # log r = 2 > 0.4 with A > 0
    adv = [torch.ones(4)]
    loss, clipfrac = compute_gmpo_loss(lp, old, adv, [torch.ones(4)], adv, 0.4, 0.4)
    torch.testing.assert_close(loss, torch.full((4,), -math.exp(0.4)))
    torch.testing.assert_close(clipfrac, torch.ones(4))
    loss.sum().backward()
    torch.testing.assert_close(lp[0].grad, torch.zeros(4))


def _gmpo(lr, adv, d_l, d_h):
    lp = [lr.clone().requires_grad_(True)]
    a = [torch.full(lr.shape, adv)]
    loss, clipfrac = compute_gmpo_loss(lp, [torch.zeros(lr.shape)], a, [torch.ones(lr.shape)], a, d_l, d_h)
    loss.sum().backward()
    return loss, clipfrac, lp[0].grad


def test_gmpo_positive_advantage_far_below_lower_bound_keeps_gradient():
    lr = torch.tensor([-3.0, -3.0])
    loss, clipfrac, grad = _gmpo(lr, 1.0, 0.4, 0.4)
    torch.testing.assert_close(loss, torch.full((2,), -math.exp(-3.0)))
    torch.testing.assert_close(clipfrac, torch.zeros(2))
    assert (grad != 0).all()


def test_gmpo_asymmetric_bounds():
    # A > 0: capped at +d_h only; A < 0: floored at -d_l only
    lr = torch.tensor([0.5, -0.5, 0.1, 0.25])
    loss, clipfrac, grad = _gmpo(lr, 2.0, 0.1, 0.2)
    expected = (torch.tensor([0.2, -0.5, 0.1, 0.2])).mean().exp()
    torch.testing.assert_close(loss, torch.full((4,), -2.0 * expected.item()))
    torch.testing.assert_close(clipfrac, torch.full((4,), 0.5))
    assert grad[0] == 0 and grad[3] == 0 and grad[1] != 0 and grad[2] != 0

    loss, clipfrac, grad = _gmpo(lr, -2.0, 0.1, 0.2)
    expected = (torch.tensor([0.5, -0.1, 0.1, 0.25])).mean().exp()
    torch.testing.assert_close(loss, torch.full((4,), 2.0 * expected.item()))
    torch.testing.assert_close(clipfrac, torch.full((4,), 0.25))
    assert grad[1] == 0 and (grad[[0, 2, 3]] != 0).all()


def test_gmpo_negative_advantage_symmetric_matches_algorithm1():
    lr = torch.tensor([0.7, -0.7, 0.3, -0.3])
    loss, clipfrac, _ = _gmpo(lr, -1.0, 0.4, 0.4)
    # Algorithm 1 verbatim: sgn_A * min(sgn_A * d, clamp(sgn_A * d, -eps, eps))
    s = -1.0
    ell = s * torch.minimum(s * lr, torch.clamp(s * lr, -0.4, 0.4))
    torch.testing.assert_close(loss, torch.full((4,), ell.mean().exp().item()))
    torch.testing.assert_close(clipfrac, torch.full((4,), 0.25))


def test_gmpo_clipfrac_ignores_zero_advantage_tokens():
    lp = [torch.tensor([2.0, 2.0, 0.0, 0.0])]
    adv = [torch.tensor([1.0, 1.0, 0.0, 0.0])]
    _, clipfrac = compute_gmpo_loss(lp, [torch.zeros(4)], adv, [torch.ones(4)], adv, 0.4, 0.4)
    torch.testing.assert_close(clipfrac, torch.ones(4))
    zero = [torch.zeros(3)]
    _, clipfrac = compute_gmpo_loss([torch.ones(3)], [torch.zeros(3)], zero, [torch.ones(3)], zero, 0.4, 0.4)
    torch.testing.assert_close(clipfrac, torch.zeros(3))


def test_gmpo_simulated_cp_split_matches_full():
    """CP>1: sequence statistics come from the gathered full sequence; each rank
    only holds a slice of the local advantages. Concatenating the rank outputs
    must equal the CP=1 result."""
    g = torch.Generator().manual_seed(3)
    old = [torch.randn(8, generator=g)]
    lp = [old[0] + 0.3 * torch.randn(8, generator=g)]
    adv = [torch.full((8,), -1.2)]
    mask = [torch.ones(8)]
    full, _ = compute_gmpo_loss(lp, old, adv, mask, adv, 0.4, 0.4)
    parts = [compute_gmpo_loss(lp, old, adv, mask, [adv[0][idx]], 0.4, 0.4)[0] for idx in (slice(0, 3), slice(3, 8))]
    torch.testing.assert_close(torch.cat(parts), full)


# ---------------------------------------------------------------------------
# End-to-end through policy_loss_function
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("loss-variants") / "pg"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


PROMPT_LENS = [6, 9, 4]
RESPONSE_LENS = [5, 8, 6]


def _inputs(args):
    make_parallel_state()
    inputs = make_inputs(7, 3, PROMPT_LENS, RESPONSE_LENS, 32, args)
    # old log-probs close to the current policy so some ratios are in range
    lp = get_log_probs_and_entropy(
        inputs["policy_logits"],
        args=args,
        unconcat_tokens=inputs["unconcat_tokens"],
        total_lengths=inputs["total_lens"],
        response_lengths=inputs["response_lens"],
        with_entropy=False,
    )["log_probs"]
    g = torch.Generator().manual_seed(11)
    inputs["log_probs"] = [x.detach() + 0.3 * torch.randn(x.shape, generator=g) for x in lp]
    inputs["rollout_log_probs"] = [x + 0.2 * torch.randn(x.shape, generator=g) for x in inputs["log_probs"]]
    inputs["advantages"] = [torch.full((n,), v) for n, v in zip(RESPONSE_LENS, [1.3, -0.8, 0.6], strict=True)]
    inputs["loss_masks"][1][2] = 0.0
    return inputs


def _som(batch):
    def f(x):
        out, off = x.new_zeros(()), 0
        for m in batch["loss_masks"]:
            n = m.numel()
            out = out + (x[off : off + n] * m).sum() / torch.clamp_min(m.sum(), 1)
            off += n
        return out

    return f


def _run(args, inputs):
    make_parallel_state()
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    loss, metrics = policy_loss_function(args, batch, logits, _som(batch))
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _ref(args, inputs, variant, use_tis=False):
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    make_parallel_state()
    lps = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )["log_probs"]
    total = lps[0].new_zeros(())
    for lp, old, roll, adv, m in zip(
        lps, batch["log_probs"], batch["rollout_log_probs"], batch["advantages"], batch["loss_masks"], strict=True
    ):
        if variant == "cispo":
            tok = ref_cispo(lp, old, adv, args.eps_clip, args.eps_clip_high)
        elif variant == "sapo":
            tok = ref_sapo(lp, old, adv, args.sapo_tau_pos, args.sapo_tau_neg)
        else:
            tok = -ref_gmpo_seq(lp, old, adv, m, args.gmpo_log_clip_low, args.gmpo_log_clip_high) * adv
        if use_tis:
            tok = tok * torch.clamp(torch.exp(old - roll), args.tis_clip_low, args.tis_clip)
        total = total + (tok * m).sum() / torch.clamp_min(m.sum(), 1)
    total.backward()
    return total.detach(), logits.grad.clone()


VARIANT_ARGS = dict(eps_clip=0.2, eps_clip_high=0.28, sapo_tau_pos=1.0, sapo_tau_neg=1.05)
VARIANT_ARGS.update(gmpo_log_clip_low=0.4, gmpo_log_clip_high=0.3)


@pytest.mark.parametrize("use_tis", [False, True])
@pytest.mark.parametrize("variant", ["cispo", "sapo", "gmpo"])
def test_policy_loss_function_variant_matches_reference(variant, use_tis):
    args = make_args(policy_loss_variant=variant, entropy_coef=0.0, use_tis=use_tis, **VARIANT_ARGS)
    inputs = _inputs(args)
    loss, metrics, grad = _run(args, inputs)
    ref_loss, ref_grad = _ref(args, inputs, variant, use_tis=use_tis)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(metrics["pg_clipfrac"])


def test_default_path_bitwise_unchanged():
    """No policy_loss_variant attribute (old Namespace) and explicit policy_loss give identical results,
    and the variant functions are never called."""
    base = make_args(entropy_coef=0.0, **{k: v for k, v in VARIANT_ARGS.items() if k.startswith("eps")})
    inputs = _inputs(base)
    loss_a, metrics_a, grad_a = _run(base, inputs)
    explicit = make_args(policy_loss_variant="policy_loss", entropy_coef=0.0, eps_clip=0.2, eps_clip_high=0.28)
    loss_b, metrics_b, grad_b = _run(explicit, inputs)
    assert torch.equal(loss_a, loss_b) and torch.equal(grad_a, grad_b)
    for k in metrics_a:
        assert torch.equal(metrics_a[k], metrics_b[k]), k

    # the variant functions are never reached on the default path (the numerics of that path are pinned
    # against stored snapshots by test_loss_snapshot.py)
    from miles.backends.training_utils.loss_hub import losses as losses_mod

    def _boom(*a, **k):
        raise AssertionError("variant function called on the default path")

    saved = {n: getattr(losses_mod, n) for n in ("compute_cispo_loss", "compute_sapo_loss", "compute_gmpo_loss")}
    try:
        for n in saved:
            setattr(losses_mod, n, _boom)
        loss_c, _, grad_c = _run(base, inputs)
    finally:
        for n, f in saved.items():
            setattr(losses_mod, n, f)
    assert torch.equal(loss_c, loss_a) and torch.equal(grad_c, grad_a)


# ---------------------------------------------------------------------------
# CLI parsing and validation
# ---------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser


def test_cli_defaults_and_flags():
    ns, _ = _parser().parse_known_args([])
    assert ns.policy_loss_variant == "policy_loss"
    assert (ns.sapo_tau_pos, ns.sapo_tau_neg) == (1.0, 1.05)
    assert (ns.gmpo_log_clip_low, ns.gmpo_log_clip_high) == (0.4, 0.4)
    ns, _ = _parser().parse_known_args(
        ["--policy-loss-variant", "sapo", "--sapo-tau-pos", "0.5", "--sapo-tau-neg", "2", "--gmpo-log-clip-low", "0.1"]
    )
    assert (ns.policy_loss_variant, ns.sapo_tau_pos, ns.sapo_tau_neg, ns.gmpo_log_clip_low) == ("sapo", 0.5, 2.0, 0.1)
    with pytest.raises(SystemExit):
        _parser().parse_known_args(["--policy-loss-variant", "grpo"])


def _ns(**kw):
    d = dict(
        policy_loss_variant="policy_loss",
        loss_type="policy_loss",
        advantage_estimator="grpo",
        eps_clip_c=None,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
        gmpo_log_clip_low=0.4,
        gmpo_log_clip_high=0.4,
    )
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.mark.parametrize("variant", ["policy_loss", "cispo", "sapo", "gmpo"])
def test_validation_accepts_defaults(variant):
    validate_policy_loss_variant_args(_ns(policy_loss_variant=variant))


def test_validation_default_variant_ignores_other_settings():
    validate_policy_loss_variant_args(_ns(advantage_estimator="gspo", eps_clip_c=3.0, sapo_tau_pos=-1.0))


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(policy_loss_variant="cispo", advantage_estimator="gspo"), "gspo"),
        (dict(policy_loss_variant="gmpo", eps_clip_c=3.0), "eps-clip-c"),
        (dict(policy_loss_variant="sapo", loss_type="custom_loss"), "loss-type"),
        (dict(policy_loss_variant="sapo", sapo_tau_pos=0.0), "sapo-tau-pos"),
        (dict(policy_loss_variant="sapo", sapo_tau_neg=float("inf")), "sapo-tau-neg"),
        (dict(policy_loss_variant="gmpo", gmpo_log_clip_low=-0.1), "gmpo-log-clip-low"),
        (dict(policy_loss_variant="gmpo", gmpo_log_clip_high=float("nan")), "gmpo-log-clip-high"),
    ],
)
def test_validation_rejects(kw, msg):
    with pytest.raises(AssertionError, match=msg):
        validate_policy_loss_variant_args(_ns(**kw))
