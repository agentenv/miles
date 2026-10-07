"""VAPO positive-example LM loss (--positive-example-lm-loss-coef).

The reference below is independent of the code under test: it recomputes
log pi(a_t|s_t) from the logits with plain torch.log_softmax and applies the
VAPO paper's eq. (9)-(10) NLL term with the policy loss's reduction.
"""

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns, positive_example_flags
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


def _run(args, inputs, flags):
    make_parallel_state()
    batch = make_batch(inputs, "policy_loss")
    if flags is not None:
        batch["positive_example_flags"] = flags
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    som = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], args.calculate_per_token_loss
    )
    loss, metrics = policy_loss_function(args, batch, logits, som)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _reference_nll(inputs, flags, per_token: bool) -> torch.Tensor:
    logits = inputs["policy_logits"][0]
    out, offset = [], 0
    for p, r, tokens, mask, flag in zip(
        PROMPTS, RESPONSES, inputs["unconcat_tokens"], inputs["loss_masks"], flags, strict=True
    ):
        # token at position i is predicted by the logits at position i-1
        lp = torch.log_softmax(logits[offset + p - 1 : offset + p + r - 1].double(), dim=-1)
        token_lp = lp.gather(-1, tokens[p : p + r].unsqueeze(-1)).squeeze(-1)
        m = mask.double() * flag
        s = (token_lp * m).sum()
        out.append(s if per_token else s / torch.clamp_min(mask.double().sum() * flag, 1))
        offset += p + r
    return -sum(out)


@pytest.mark.parametrize("per_token", [False, True])
def test_matches_reference(per_token):
    args = make_args(entropy_coef=0.0, calculate_per_token_loss=per_token)
    inputs = make_inputs(0, 4, PROMPTS, RESPONSES, VOCAB, args)
    flags = [1.0, 0.0, 1.0, 0.0]
    base, _, base_grad = _run(args, inputs, None)
    mu = 0.1
    args_mu = make_args(entropy_coef=0.0, calculate_per_token_loss=per_token, positive_example_lm_loss_coef=mu)
    loss, metrics, grad = _run(args_mu, inputs, flags)
    ref = _reference_nll(inputs, flags, per_token)
    torch.testing.assert_close(metrics["positive_lm_loss"].double(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss.double(), base.double() + mu * ref, rtol=1e-5, atol=1e-5)
    assert not torch.equal(grad, base_grad)


def test_no_positive_samples_adds_nothing():
    args = make_args(entropy_coef=0.0)
    inputs = make_inputs(1, 4, PROMPTS, RESPONSES, VOCAB, args)
    base, _, base_grad = _run(args, inputs, None)
    loss, metrics, grad = _run(make_args(entropy_coef=0.0, positive_example_lm_loss_coef=0.1), inputs, [0.0] * 4)
    assert metrics["positive_lm_loss"].item() == 0.0
    torch.testing.assert_close(loss, base)
    torch.testing.assert_close(grad, base_grad)


def test_default_unchanged():
    """coef absent / 0: identical loss, metrics keys and gradients; flags ignored."""
    args = make_args()
    inputs = make_inputs(2, 4, PROMPTS, RESPONSES, VOCAB, args)
    base, base_metrics, base_grad = _run(args, inputs, None)
    loss, metrics, grad = _run(make_args(positive_example_lm_loss_coef=0.0), inputs, [1.0] * 4)
    assert torch.equal(loss, base) and torch.equal(grad, base_grad)
    assert set(metrics) == set(base_metrics) and "positive_lm_loss" not in metrics


def test_missing_flags_raises():
    args = make_args(positive_example_lm_loss_coef=0.1)
    inputs = make_inputs(3, 4, PROMPTS, RESPONSES, VOCAB, args)
    with pytest.raises(ValueError, match="positive_example_flags"):
        _run(args, inputs, None)


def test_flags_from_rewards():
    assert positive_example_flags([1.0, 0.0, -1.0, 0.5], 0.0) == [1.0, 0.0, 0.0, 1.0]
    assert positive_example_flags([1.0, 0.5], 0.5) == [1.0, 0.0]


@pytest.mark.parametrize("coef", [0.0, 0.1])
def test_rollout_data_flags_only_when_enabled(coef):
    make_parallel_state()
    args = make_args(positive_example_lm_loss_coef=coef, positive_example_reward_threshold=0.0)
    inputs = make_inputs(4, 4, PROMPTS, RESPONSES, VOCAB, args)
    rollout_data = make_rollout_data(inputs)
    compute_advantages_and_returns(args, rollout_data)
    if coef:
        assert rollout_data["positive_example_flags"] == [1.0 if r > 0 else 0.0 for r in inputs["rewards"]]
    else:
        assert "positive_example_flags" not in rollout_data
