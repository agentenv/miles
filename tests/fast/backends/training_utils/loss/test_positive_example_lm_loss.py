"""VAPO positive-example LM loss (--positive-example-lm-loss-coef).

The reference below is independent of the code under test: it recomputes
log pi(a_t|s_t) from the logits with plain torch.log_softmax and applies the
VAPO paper's eq. (9)-(10) NLL term: the sum over positive-sample tokens divided by
the global positive-token count P of the optimizer step. Miles divides the summed
loss by the step's global normalizer D afterwards, so the term returned by the
policy loss must equal D * eq9.
"""

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import (
    compute_advantages_and_returns,
    positive_example_flags,
    positive_example_flags_from_rollout,
    positive_example_lm_weights,
)
from miles.ray.rollout.train_data_conversion import sample_success_flag
from miles.utils.arguments import validate_positive_example_args
from miles.backends.training_utils.loss_hub.losses import policy_loss_function

from .loss_test_utils import deep_clone, make_batch, make_inputs, make_parallel_state, make_rollout_data
from .loss_test_utils import make_args as _make_args

PROMPTS = [3, 2, 4, 2]
RESPONSES = [5, 3, 6, 4]
VOCAB = 11
# policy_loss_function attributes the shared loss-test defaults predate
_EXTRA = dict(observe_training_entropy=False)


def make_args(**overrides):
    return _make_args(**{**_EXTRA, **overrides})


def _run(args, inputs, weights):
    make_parallel_state()
    batch = make_batch(inputs, "policy_loss")
    if weights is not None:
        batch["positive_example_lm_weights"] = weights
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    som = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], args.calculate_per_token_loss
    )
    loss, metrics = policy_loss_function(args, batch, logits, som)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _ref_positive_logp(inputs, prompts, responses):
    """Per sample: (sum of log pi over masked response tokens, mask token count), via plain log_softmax."""
    logits = inputs["policy_logits"][0]
    out, offset = [], 0
    for p, r, tokens, mask in zip(prompts, responses, inputs["unconcat_tokens"], inputs["loss_masks"], strict=True):
        # token at position i is predicted by the logits at position i-1
        lp = torch.log_softmax(logits[offset + p - 1 : offset + p + r - 1].double(), dim=-1)
        token_lp = lp.gather(-1, tokens[p : p + r].unsqueeze(-1)).squeeze(-1)
        out.append(((token_lp * mask.double()).sum(), float(mask.sum())))
        offset += p + r
    return out


def _eq9(per_sample, flags):
    num = -sum(lp for (lp, _), f in zip(per_sample, flags, strict=True) if f)
    den = sum(n for (_, n), f in zip(per_sample, flags, strict=True) if f)
    return num / den


def _weights(flags, inputs, per_token, groups=None, num_rollouts=None, reduce=None):
    mask_sums = [float(m.sum()) for m in inputs["loss_masks"]]
    groups = groups or [list(range(len(flags)))]
    return positive_example_lm_weights(flags, mask_sums, step_groups=groups, per_token=per_token,
                                       num_rollouts=num_rollouts, all_reduce_sum=reduce)


@pytest.mark.parametrize("per_token", [False, True])
def test_matches_eq9_reference(per_token):
    args = make_args(entropy_coef=0.0, calculate_per_token_loss=per_token)
    inputs = make_inputs(0, 4, PROMPTS, RESPONSES, VOCAB, args)
    flags = [1.0, 0.0, 1.0, 0.0]
    base, _, base_grad = _run(args, inputs, None)
    mu = 0.1
    args_mu = make_args(entropy_coef=0.0, calculate_per_token_loss=per_token, positive_example_lm_loss_coef=mu)
    weights = _weights(flags, inputs, per_token, num_rollouts=[4])
    loss, metrics, grad = _run(args_mu, inputs, weights)
    per_sample = _ref_positive_logp(inputs, PROMPTS, RESPONSES)
    D = sum(max(n, 1.0) for _, n in per_sample) if per_token else 4.0
    ref = _eq9(per_sample, flags) * D
    torch.testing.assert_close(metrics["positive_lm_loss"].double(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss.double(), base.double() + mu * ref, rtol=1e-5, atol=1e-5)
    assert not torch.equal(grad, base_grad)


@pytest.mark.parametrize("per_token", [False, True])
def test_global_count_across_microbatches_and_ranks(per_token):
    """2 DP ranks x 2 micro-batches x 2 samples, one optimizer step: summing every micro-batch's term and
    dividing by the step's global normalizer D gives eq. 9 over all 8 samples (P counted globally)."""
    args = make_args(entropy_coef=0.0, calculate_per_token_loss=per_token, positive_example_lm_loss_coef=0.1)
    shapes = [([3, 2], [5, 3]), ([4, 2], [6, 4]), ([2, 3], [4, 7]), ([3, 3], [2, 5])]
    mbs = [make_inputs(10 + k, 2, p, r, VOCAB, args) for k, (p, r) in enumerate(shapes)]
    flags = [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]]
    ranks = [[0, 1], [2, 3]]  # micro-batch ids per rank

    def rank_inputs(rank):
        return {"loss_masks": [m for k in ranks[rank] for m in mbs[k]["loss_masks"]]}

    def rank_flags(rank):
        return [f for k in ranks[rank] for f in flags[k]]

    local_stats = []
    for rank in range(2):  # pass 1: record each rank's local stats (what all_reduce would sum)
        _weights(rank_flags(rank), rank_inputs(rank), per_token, groups=[[0, 1, 2, 3]], num_rollouts=[8],
                 reduce=lambda t: local_stats.append(t.clone()))
    total = local_stats[0] + local_stats[1]
    term = 0.0
    for rank in range(2):
        w = _weights(rank_flags(rank), rank_inputs(rank), per_token, groups=[[0, 1, 2, 3]], num_rollouts=[8],
                     reduce=lambda t: t.copy_(total))
        for j, k in enumerate(ranks[rank]):
            _, metrics, _ = _run(args, mbs[k], w[2 * j : 2 * j + 2])
            term += metrics["positive_lm_loss"].double()
    per_sample = [x for k, (p, r) in enumerate(shapes) for x in _ref_positive_logp(mbs[k], p, r)]
    all_flags = [f for fl in flags for f in fl]
    D = sum(max(n, 1.0) for _, n in per_sample) if per_token else 8.0
    torch.testing.assert_close(term / D, _eq9(per_sample, all_flags), rtol=1e-5, atol=1e-6)


def test_weights_per_step_groups_independent():
    w = positive_example_lm_weights([1.0, 0.0, 1.0, 1.0], [2.0, 3.0, 4.0, 0.0], step_groups=[[0, 1], [2, 3]],
                                    per_token=True, num_rollouts=None)
    assert w == pytest.approx([5.0 / 2.0, 0.0, 5.0 / 4.0, 5.0 / 4.0])  # step 2: D = 4 + max(0, 1) = 5, P = 4
    w = positive_example_lm_weights([1.0, 0.0], [2.0, 3.0], step_groups=[[0, 1]], per_token=False, num_rollouts=[6])
    assert w == pytest.approx([3.0, 0.0])
    assert positive_example_lm_weights([0.0], [2.0], step_groups=[[0]], per_token=True, num_rollouts=None) == [0.0]


def test_no_positive_samples_adds_nothing():
    args = make_args(entropy_coef=0.0)
    inputs = make_inputs(1, 4, PROMPTS, RESPONSES, VOCAB, args)
    base, _, base_grad = _run(args, inputs, None)
    weights = _weights([0.0] * 4, inputs, False, num_rollouts=[4])
    loss, metrics, grad = _run(make_args(entropy_coef=0.0, positive_example_lm_loss_coef=0.1), inputs, weights)
    assert metrics["positive_lm_loss"].item() == 0.0
    torch.testing.assert_close(loss, base)
    torch.testing.assert_close(grad, base_grad)


def test_default_unchanged():
    """coef absent / 0: identical loss, metrics keys and gradients; weights ignored."""
    args = make_args()
    inputs = make_inputs(2, 4, PROMPTS, RESPONSES, VOCAB, args)
    base, base_metrics, base_grad = _run(args, inputs, None)
    loss, metrics, grad = _run(make_args(positive_example_lm_loss_coef=0.0), inputs, [1.0] * 4)
    assert torch.equal(loss, base) and torch.equal(grad, base_grad)
    assert set(metrics) == set(base_metrics) and "positive_lm_loss" not in metrics


def test_missing_weights_raises():
    args = make_args(positive_example_lm_loss_coef=0.1)
    inputs = make_inputs(3, 4, PROMPTS, RESPONSES, VOCAB, args)
    with pytest.raises(ValueError, match="positive_example_lm_weights"):
        _run(args, inputs, None)


def test_flags_from_rewards():
    assert positive_example_flags([1.0, 0.0, -1.0, 0.5], 0.0) == [1.0, 0.0, 0.0, 1.0]
    assert positive_example_flags([1.0, 0.5], 0.5) == [1.0, 0.0]


def test_success_source_is_default_and_strict():
    args = make_args(positive_example_lm_loss_coef=0.1)
    assert getattr(args, "positive_example_source", "success") == "success"
    rewards = [1.0, 1.0, 0.0]
    assert positive_example_flags_from_rollout(args, {"positive_example_success": [1, 0, 0]}, rewards) == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="success"):  # no field at all: no silent reward fallback
        positive_example_flags_from_rollout(args, {}, rewards)
    with pytest.raises(ValueError, match="lack"):
        positive_example_flags_from_rollout(args, {"positive_example_success": [1, -1, 0]}, rewards)
    reward_args = make_args(positive_example_source="reward", positive_example_reward_threshold=0.5)
    assert positive_example_flags_from_rollout(reward_args, {}, [1.0, 0.5, 0.6]) == [1.0, 0.0, 1.0]
    reward_default = make_args(positive_example_source="reward", positive_example_reward_threshold=None)
    assert positive_example_flags_from_rollout(reward_default, {}, [1.0, 0.0]) == [1.0, 0.0]


def test_validate_args():
    validate_positive_example_args(make_args(positive_example_source="success", positive_example_reward_threshold=None))
    validate_positive_example_args(make_args(positive_example_source="reward", positive_example_reward_threshold=0.0))
    with pytest.raises(ValueError, match="only valid"):
        validate_positive_example_args(make_args(positive_example_source="success",
                                                 positive_example_reward_threshold=0.0))
    with pytest.raises(ValueError):
        validate_positive_example_args(make_args(positive_example_source="bogus",
                                                 positive_example_reward_threshold=None))


def test_sample_success_flag():
    assert sample_success_flag({"success": True}) == 1
    assert sample_success_flag({"is_correct": False}) == 0
    assert sample_success_flag({"success": 1, "is_correct": True}) == 1
    assert sample_success_flag({}) == -1 and sample_success_flag(None) == -1
    with pytest.raises(ValueError):
        sample_success_flag({"success": 0.7})
    with pytest.raises(ValueError):
        sample_success_flag({"success": True, "is_correct": False})


@pytest.mark.parametrize("coef", [0.0, 0.1])
def test_rollout_data_weights_only_when_enabled(coef):
    make_parallel_state()
    args = make_args(positive_example_lm_loss_coef=coef, positive_example_source="reward",
                     positive_example_reward_threshold=0.0, global_batch_size=4)
    inputs = make_inputs(4, 4, PROMPTS, RESPONSES, VOCAB, args)
    rollout_data = make_rollout_data(inputs)
    compute_advantages_and_returns(args, rollout_data)
    if coef:
        flags = [1.0 if r > 0 else 0.0 for r in inputs["rewards"]]
        expected = _weights(flags, inputs, args.calculate_per_token_loss, num_rollouts=[4])
        assert rollout_data["positive_example_lm_weights"] == pytest.approx(expected)
    else:
        assert "positive_example_lm_weights" not in rollout_data
