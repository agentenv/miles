"""The LoRA bridge path must hand --calculate-per-token-loss to the Megatron model config, like the non-LoRA
bridge path (model_provider._apply_bridge_runtime_config) does."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.bridge import _setup_lora_model_via_bridge


class TestLoraBridgeCarriesPerTokenLoss:
    @pytest.mark.parametrize("value", [True, False])
    def test_the_provider_config_matches_the_argument_before_finalize(
        self, bridge_args: Namespace, bridge_provider: object, value: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Megatron's schedule and DDP read the flag from the model config that finalize() freezes."""
        seen: list[bool] = []
        finalize = bridge_provider.finalize
        monkeypatch.setattr(
            bridge_provider, "finalize", lambda: (seen.append(bridge_provider.calculate_per_token_loss), finalize())
        )
        bridge_args.backend.calculate_per_token_loss = value

        _setup_lora_model_via_bridge(bridge_args)

        assert seen == [value]


def _megatron_scaled_grad(per_token_args: bool, config_per_token: bool, lengths, adv, n_mb):
    """Gradient of one DP rank's microbatches under Miles loss.py scaling + Megatron forward_step scaling.

    Miles returns ``(loss, num_tokens)``;
    Megatron's schedule divides by num_tokens and num_microbatches only when config.calculate_per_token_loss is
    False, and with it True finalize_model_grads divides by the total token count instead.
    """
    from miles.backends.training_utils.data import context_parallel as cp_utils

    torch.manual_seed(0)
    theta = torch.randn(sum(lengths), dtype=torch.float64, requires_grad=True)
    logp = torch.nn.functional.logsigmoid(theta)
    x = -torch.exp(logp - logp.detach()) * torch.cat([a.repeat(n) for a, n in zip(adv, lengths)])
    per_mb = len(lengths) // n_mb
    total, offset, total_tokens = 0, 0, sum(lengths)
    for mb in range(n_mb):
        ls = lengths[mb * per_mb : (mb + 1) * per_mb]
        n = sum(ls)
        reducer = cp_utils.get_sum_of_sample_mean([0] * len(ls), ls, [torch.ones(k) for k in ls], per_token_args)
        loss = reducer(x[offset : offset + n])
        offset += n
        num_tokens = n if per_token_args else 1
        if not per_token_args:
            loss = loss * n_mb / len(lengths)
        if not config_per_token:
            loss = loss / num_tokens / n_mb
        total = total + loss
    total.backward()
    grad = theta.grad
    return grad / total_tokens if config_per_token else grad


class TestPerTokenLossNumerics:
    @pytest.fixture(autouse=True)
    def _cp1(self, monkeypatch):
        from miles.backends.training_utils.data import context_parallel as cp_utils

        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))

    def test_with_the_config_set_per_token_differs_from_sample_mean_for_unequal_lengths(self):
        lengths, adv = [3, 7, 5, 9], torch.tensor([1.0, -1.0, 0.5, -0.5])
        sample = _megatron_scaled_grad(False, False, lengths, adv, n_mb=2)
        token = _megatron_scaled_grad(True, True, lengths, adv, n_mb=2)
        assert not torch.allclose(sample, token)

    def test_the_missing_config_with_one_sample_per_microbatch_collapses_to_sample_mean(self):
        """The pre-fix LoRA path: per-microbatch token mean / n_mb == sample mean when each microbatch is one
        sample, so a per-token run silently trained on the sample mean."""
        lengths, adv = [3, 7, 5, 9], torch.tensor([1.0, -1.0, 0.5, -0.5])
        sample = _megatron_scaled_grad(False, False, lengths, adv, n_mb=4)
        buggy_token = _megatron_scaled_grad(True, False, lengths, adv, n_mb=4)
        torch.testing.assert_close(buggy_token, sample)
        fixed_token = _megatron_scaled_grad(True, True, lengths, adv, n_mb=4)
        assert not torch.allclose(fixed_token, sample)
