"""SAO port (sao_dis + HL-Gauss categorical value loss) CPU parity tests.

Reference math is written independently with python ``math`` and does not
import the code under test. Source of the ported implementation:
agentenv/miles feat/sao-tbench21-e2e-validation @ 16a9bea4 (introduced in
e25048ed "feat(rl): add SAO value and online training").
"""

import argparse
import math

import pytest
import torch

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    g = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=g, intra_dp_cp=g, cp=g, tp=g, pp=g, ep=g, etp=g, indep_dp=g))


# ---------------- independent references ----------------
def ref_dis(logp, rollout_logp, adv, eps_low, eps_high):
    out = []
    for lp, rlp, a in zip(logp, rollout_logp, adv, strict=True):
        r = math.exp(max(min(lp - rlp, 20.0), -20.0))
        rejected = r <= 1 - eps_low or r >= 1 + eps_high
        out.append((0.0 if rejected else -r * a * lp, r, float(rejected)))
    return out


def ref_hl_gauss(target, low, high, bins, sigma_ratio):
    width = (high - low) / (bins - 1)
    sigma = sigma_ratio * width
    mu = min(max(target, low), high)
    mass = []
    for k in range(bins):
        c = low + k * width
        mass.append(0.5 * (math.erf((c + width / 2 - mu) / (sigma * math.sqrt(2))) - math.erf((c - width / 2 - mu) / (sigma * math.sqrt(2)))))
    z = sum(mass)
    return [m / z for m in mass]


def ref_ce(logits_row, target_dist):
    m = max(logits_row)
    lse = m + math.log(sum(math.exp(x - m) for x in logits_row))
    return -sum(t * (x - lse) for t, x in zip(target_dist, logits_row, strict=True))


# ---------------- tests ----------------
def test_sao_dis_matches_reference_and_rejects_strictly():
    from miles.backends.training_utils.loss_hub.math_utils import compute_sao_dis_policy_loss

    gen = torch.Generator().manual_seed(0)
    rollout = -torch.rand(64, generator=gen, dtype=torch.float64) * 3
    logp = rollout + torch.randn(64, generator=gen, dtype=torch.float64) * 0.8
    adv = torch.randn(64, generator=gen, dtype=torch.float64)
    for eps_low, eps_high in [(0.8, 3.0), (0.3, 5.0)]:
        loss, ratio, rejected, below, above = compute_sao_dis_policy_loss(logp, rollout, adv, eps_low, eps_high)
        ref = ref_dis(logp.tolist(), rollout.tolist(), adv.tolist(), eps_low, eps_high)
        torch.testing.assert_close(loss, torch.tensor([r[0] for r in ref], dtype=torch.float64))
        torch.testing.assert_close(ratio, torch.tensor([r[1] for r in ref], dtype=torch.float32))  # ratio is fp32 by design (_safe_clamp_log_ratio)
        torch.testing.assert_close(rejected, torch.tensor([r[2] for r in ref], dtype=torch.float32))
        torch.testing.assert_close(rejected, below + above)
        assert 0 < rejected.sum() < 64


def test_sao_dis_gradient_keeps_ratio_in_graph():
    from miles.backends.training_utils.loss_hub.math_utils import compute_sao_dis_policy_loss

    lp = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    loss, *_ = compute_sao_dis_policy_loss(lp, torch.tensor([-1.2], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64), 0.3, 5.0)
    loss.sum().backward()
    r = math.exp(0.2)
    # d/dlp [-r(lp) * A * lp] = -A * r * (lp + 1)
    assert lp.grad.item() == pytest.approx(-2.0 * r * (-1.0 + 1.0), abs=1e-6)
    lp2 = torch.tensor([-0.5], dtype=torch.float64, requires_grad=True)
    loss, *_ = compute_sao_dis_policy_loss(lp2, torch.tensor([-1.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64), 0.3, 5.0)
    loss.sum().backward()
    r = math.exp(0.5)
    assert lp2.grad.item() == pytest.approx(-2.0 * r * (-0.5 + 1.0), rel=1e-6)


@pytest.mark.parametrize("target", [0.0, 0.013, 0.5, 0.731, 1.0, 1.7, -0.4])
def test_hl_gauss_projection_matches_reference(target):
    from miles.backends.training_utils.loss_hub.losses import _hl_gauss_target_distribution

    support = torch.linspace(0.0, 1.0, 51, dtype=torch.float32)
    got = _hl_gauss_target_distribution(torch.tensor([target]), support, 0.75)[0]
    torch.testing.assert_close(got, torch.tensor(ref_hl_gauss(target, 0.0, 1.0, 51, 0.75), dtype=torch.float32), atol=1e-6, rtol=1e-5)


def _args(**kw):
    base = dict(
        qkv_format="thd", true_on_policy_mode=False, rollout_temperature=0.7, context_parallel_size=1,
        allgather_cp=False, value_clip=0.2, value_loss_type="mse",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _batch(lengths, out_dim, seed=1):
    gen = torch.Generator().manual_seed(seed)
    prompt = 3
    total = [prompt + n for n in lengths]
    logits = torch.randn(1, sum(total), out_dim, generator=gen)
    returns = [torch.rand(n, generator=gen) for n in lengths]
    batch = {
        "unconcat_tokens": [torch.zeros(t, dtype=torch.long) for t in total],
        "total_lengths": total,
        "response_lengths": lengths,
        "returns": returns,
        "values": [torch.rand(n, generator=gen) for n in lengths],
        "loss_masks": [torch.ones(n) for n in lengths],
    }
    return logits, batch


def _response_rows(logits, lengths, prompt=3):
    rows, off = [], 0
    flat = logits[0]
    for n in lengths:
        t = prompt + n
        rows.append(flat[off + t - n - 1 : off + t - 1])
        off += t
    return rows


def test_classification_value_loss_matches_reference():
    from miles.backends.training_utils.loss_hub.losses import value_loss_function

    lengths = [4, 2, 5]
    logits, batch = _batch(lengths, 51)
    args = _args(value_loss_type="classification", value_num_bins=51, value_reward_range=[0.0, 1.0],
                 value_target_type="hl_gauss", hl_gauss_sigma_ratio=0.75)
    mean = lambda x: x.sum() / x.numel()  # noqa: E731
    loss, metrics = value_loss_function(args, batch, logits, mean)
    rows = torch.cat(_response_rows(logits, lengths)).tolist()
    rets = torch.cat(batch["returns"]).tolist()
    ref = sum(ref_ce(r, ref_hl_gauss(t, 0.0, 1.0, 51, 0.75)) for r, t in zip(rows, rets, strict=True)) / len(rets)
    assert loss.item() == pytest.approx(ref, rel=1e-5)
    assert set(metrics) == {"value_loss", "value_accuracy", "value_entropy", "value_confidence"}


def test_get_values_classification_is_expectation_and_skips_temperature():
    from miles.backends.training_utils.loss_hub.logit_processors import get_values

    lengths = [3, 2]
    logits, batch = _batch(lengths, 51)
    args = _args(value_loss_type="classification", value_num_bins=51, value_reward_range=[-1.0, 1.0], true_on_policy_mode=True)
    vals = torch.cat(get_values(logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
                                total_lengths=batch["total_lengths"], response_lengths=lengths)["values"])
    ref = []
    for row in torch.cat(_response_rows(logits, lengths)).tolist():
        m = max(row); e = [math.exp(x - m) for x in row]; z = sum(e)
        ref.append(sum(ei / z * (-1.0 + 2.0 * k / 50) for k, ei in enumerate(e)))
    torch.testing.assert_close(vals, torch.tensor(ref), atol=1e-5, rtol=1e-5)


def test_default_mse_value_path_unchanged():
    from miles.backends.training_utils.loss_hub.losses import value_loss_function

    lengths = [4, 2]
    logits, batch = _batch(lengths, 1)
    mean = lambda x: x.sum() / x.numel()  # noqa: E731
    loss, metrics = value_loss_function(_args(), batch, logits, mean)
    v = torch.cat(_response_rows(logits, lengths))[:, 0]
    old = torch.cat(batch["values"]); ret = torch.cat(batch["returns"])
    clipped = old + (v - old).clamp(-0.2, 0.2)
    ref = torch.max((clipped - ret) ** 2, (v - ret) ** 2).mean()
    torch.testing.assert_close(loss, ref)
    assert set(metrics) == {"value_loss", "value_clipfrac"}


def test_cli_defaults_unchanged():
    from miles.utils.arguments import get_miles_extra_args_provider

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    ns, _ = parser.parse_known_args([])
    assert ns.policy_objective == "ppo"
    assert ns.value_loss_type == "mse"
    assert ns.value_num_bins == 51 and ns.value_target_type == "hl_gauss"
