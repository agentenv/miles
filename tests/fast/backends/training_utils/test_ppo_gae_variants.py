"""--gae-variant dispatch: cross-check against the independent yeto reference
(rl_gae_reference.py, copied verbatim) and default-unchanged guarantees."""

import argparse

import pytest
import torch

from miles.backends.training_utils.loss_hub import advantages as adv_mod
from miles.backends.training_utils.loss_hub.math_utils import (
    get_advantages_and_returns_batch,
    length_adaptive_lambda,
    vanilla_gae,
)
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from tests.fast.backends.training_utils import rl_gae_reference as ref
from tests.fast.backends.training_utils import _gae_orig_039471508 as _orig

D = torch.float64


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    g = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(intra_dp=g, intra_dp_cp=g, cp=g, tp=g, pp=g, ep=g, etp=g, indep_dp=g)
    )


def _batch(seed=0, lengths=(5, 9, 1, 7), dtype=D, masked=False):
    gen = torch.Generator().manual_seed(seed)
    values = [torch.randn(n, generator=gen, dtype=dtype) for n in lengths]
    rewards = [0.1 * torch.randn(n, generator=gen, dtype=dtype) for n in lengths]
    term = [float(x) for x in torch.randn(len(lengths), generator=gen)]
    if masked:
        masks = [(torch.rand(n, generator=gen) > 0.3).int() for n in lengths]
    else:
        masks = [torch.ones(n, dtype=torch.int) for n in lengths]
    return values, rewards, term, masks


def _run(values, rewards, term, masks, gamma, lambd, **kw):
    lens = [v.numel() for v in values]
    return get_advantages_and_returns_batch(
        total_lengths=lens, response_lengths=lens, values_list=values, rewards_list=rewards,
        terminal_rewards=term, qkv_format="thd", max_seq_lens=None, loss_masks=masks,
        gamma=gamma, lambd=lambd, **kw,
    )


def _ref_inputs(v, r, t, m):
    idx = m.nonzero(as_tuple=True)[0]
    rr = r[idx].clone()
    if idx.numel():
        rr[-1] += t
    return idx, rr, v[idx]


# ---------------- default unchanged (element-wise) ----------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("chunked", [True, False])
def test_default_and_explicit_vanilla_are_elementwise_original(dtype, masked, chunked):
    values, rewards, term, masks = _batch(1, dtype=dtype, masked=masked)
    a0, r0 = _run(values, rewards, term, masks, 0.97, 0.9, chunked=chunked)
    a1, r1 = _run(values, rewards, term, masks, 0.97, 0.9, chunked=chunked, gae_variant="vanilla",
                  gae_lambd_mode="fixed", segment_ids_list=[[0] * v.numel() for v in values])
    for i in range(len(values)):
        assert torch.equal(a0[i], a1[i]) and torch.equal(r0[i], r1[i])
    # and it is bit-identical to the pre-change implementation (frozen copy of
    # get_advantages_and_returns_batch + kernels from yeto/ports 039471508)
    lens = [v.numel() for v in values]
    oa, orr = _orig.get_advantages_and_returns_batch(
        total_lengths=lens, response_lengths=lens, values_list=values, rewards_list=rewards,
        terminal_rewards=term, qkv_format="thd", max_seq_lens=None, loss_masks=masks,
        gamma=0.97, lambd=0.9, chunked=chunked,
    )
    for i in range(len(values)):
        assert torch.equal(a0[i], oa[i]) and torch.equal(r0[i], orr[i])


def test_compute_advantages_default_args_pass_no_variant_kwargs(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return [torch.zeros(1)], [torch.zeros(1)]

    monkeypatch.setattr(adv_mod, "get_advantages_and_returns_batch", fake)
    args = argparse.Namespace(advantage_estimator="ppo", kl_coef=0.0, qkv_format="thd", gamma=1.0, lambd=1.0)
    adv_mod.compute_advantages(args, kl=[torch.zeros(1)], rewards=[1.0], log_probs=None,
                               loss_masks=[torch.ones(1)], total_lengths=[1], response_lengths=[1],
                               values=[torch.zeros(1)])
    assert not any(k.startswith("gae_") or k == "segment_ids_list" for k in seen)
    args.gae_variant, args.gae_lambd_mode = "vanilla", "fixed"
    seen.clear()
    adv_mod.compute_advantages(args, kl=[torch.zeros(1)], rewards=[1.0], log_probs=None,
                               loss_masks=[torch.ones(1)], total_lengths=[1], response_lengths=[1],
                               values=[torch.zeros(1)])
    assert not any(k.startswith("gae_") for k in seen)
    args.gae_variant = "cross_segment"
    seen.clear()
    adv_mod.compute_advantages(args, kl=[torch.zeros(1)], rewards=[1.0], log_probs=None,
                               loss_masks=[torch.ones(1)], total_lengths=[1], response_lengths=[1],
                               values=[torch.zeros(1)], segment_ids=[[0]])
    assert seen["gae_variant"] == "cross_segment" and seen["segment_ids_list"] == [[0]]


def test_arguments_defaults():
    from miles.utils import arguments
    src = open(arguments.__file__).read()
    for flag in ("--gae-variant", "--gae-lambd-mode", "--gae-length-alpha", "--gae-critic-lambd"):
        assert flag in src


# ---------------- cross-check vs reference ----------------

@pytest.mark.parametrize("masked", [False, True])
def test_length_adaptive_matches_reference(masked):
    values, rewards, term, masks = _batch(2, masked=masked)
    for kw in ({"gae_variant": "length_adaptive"}, {"gae_lambd_mode": "length_adaptive"}):
        a, r = _run(values, rewards, term, masks, 0.99, 0.123, gae_length_alpha=1.5, **kw)
        for i, (v, rw, t, m) in enumerate(zip(values, rewards, term, masks)):
            idx, rr, vv = _ref_inputs(v, rw, t, m)
            if not idx.numel():
                continue
            ea, er = ref.gae_length_adaptive(rr, vv, 0.99, v.numel(), alpha=1.5)
            assert torch.allclose(a[i][idx], ea, atol=1e-12) and torch.allclose(r[i][idx], er, atol=1e-12)


def test_length_adaptive_lambda_value():
    assert length_adaptive_lambda(100, 1.5) == pytest.approx(1 - 1 / 150)
    assert length_adaptive_lambda(100, 1.5) == pytest.approx(ref.length_adaptive_lambda(100, 1.5))


def test_decoupled_matches_reference():
    values, rewards, term, masks = _batch(3, masked=True)
    a, r = _run(values, rewards, term, masks, 1.0, 0.95, gae_variant="decoupled", gae_critic_lambd=1.0)
    for i, (v, rw, t, m) in enumerate(zip(values, rewards, term, masks)):
        idx, rr, vv = _ref_inputs(v, rw, t, m)
        if not idx.numel():
            continue
        ea, er = ref.gae_decoupled(rr, vv, 1.0, 0.95, 1.0)
        assert torch.allclose(a[i][idx], ea, atol=1e-12) and torch.allclose(r[i][idx], er, atol=1e-12)


def test_decoupled_with_length_adaptive_policy_lambda():
    values, rewards, term, masks = _batch(4)
    a, r = _run(values, rewards, term, masks, 1.0, 0.5, gae_variant="decoupled",
                gae_lambd_mode="length_adaptive", gae_critic_lambd=1.0)
    for i, (v, rw, t, m) in enumerate(zip(values, rewards, term, masks)):
        idx, rr, vv = _ref_inputs(v, rw, t, m)
        ea, er = ref.gae_decoupled(rr, vv, 1.0, ref.length_adaptive_lambda(v.numel()), 1.0)
        assert torch.allclose(a[i][idx], ea, atol=1e-12) and torch.allclose(r[i][idx], er, atol=1e-12)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("lambd_mode", ["fixed", "length_adaptive"])
def test_cross_segment_matches_reference(masked, lambd_mode):
    values, rewards, term, masks = _batch(5, lengths=(6, 10, 3, 8), masked=masked)
    segs = [[0, 0, 1, 1, 1, 2], [0] * 3 + [1] * 4 + [2] * 3, [0, 0, 0], [0, 1, 1, 1, 2, 2, 3, 3]]
    a, r = _run(values, rewards, term, masks, 0.95, 0.9, gae_variant="cross_segment",
                gae_lambd_mode=lambd_mode, segment_ids_list=segs)
    for i, (v, rw, t, m) in enumerate(zip(values, rewards, term, masks)):
        idx, rr, vv = _ref_inputs(v, rw, t, m)
        if not idx.numel():
            continue
        lam = ref.length_adaptive_lambda(v.numel()) if lambd_mode == "length_adaptive" else 0.9
        seg = torch.tensor(segs[i])[idx].tolist()
        ea, er = ref.gae_cross_segment(rr, vv, seg, 0.95, lam)
        assert torch.allclose(a[i][idx], ea, atol=1e-12) and torch.allclose(r[i][idx], er, atol=1e-12)
        assert torch.all(a[i][m == 0] == 0)


def test_cross_segment_two_segments_factor_n2():
    # spec scenario: front segment advantage = local GAE * (gamma*lambda)^{n_2}
    values, rewards, term, masks = _batch(6, lengths=(7,))
    seg = [[0, 0, 0, 1, 1, 1, 1]]
    g, l = 0.9, 0.8
    a, _ = _run(values, rewards, term, masks, g, l, gae_variant="cross_segment", segment_ids_list=seg)
    local, _ = vanilla_gae(rewards[0][None, :3], values[0][None, :3], g, l)
    assert torch.allclose(a[0][:3], local[0] * (g * l) ** 4, atol=1e-12)


def test_cross_segment_degenerates_to_vanilla():
    values, rewards, term, masks = _batch(7, masked=True)
    base_a, base_r = _run(values, rewards, term, masks, 0.97, 0.9, chunked=False)
    for segs in (None, [[3] * v.numel() for v in values]):
        a, r = _run(values, rewards, term, masks, 0.97, 0.9, gae_variant="cross_segment", segment_ids_list=segs)
        for i in range(len(values)):
            assert torch.allclose(a[i], base_a[i], atol=1e-12) and torch.allclose(r[i], base_r[i], atol=1e-12)


def test_cross_segment_rejects_decreasing_ids_and_unknown_variant():
    values, rewards, term, masks = _batch(8, lengths=(4,))
    with pytest.raises(ValueError):
        _run(values, rewards, term, masks, 1.0, 1.0, gae_variant="cross_segment", segment_ids_list=[[1, 1, 0, 0]])
    with pytest.raises(ValueError):
        _run(values, rewards, term, masks, 1.0, 1.0, gae_variant="bogus")
