"""The LoRA bridge path must hand --calculate-per-token-loss to the Megatron model config, like the non-LoRA
bridge path (model_provider._apply_bridge_runtime_config) does."""

import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.lora import bridge as lora_bridge


class _Provider:
    """Records the attributes the builder sets, and what they were when finalize() ran."""

    def __init__(self):
        self.finalized_with = None
        self.hooks = []

    def finalize(self):
        self.finalized_with = dict(vars(self))

    def register_pre_wrap_hook(self, hook):
        self.hooks.append(hook)

    def provide_distributed_model(self, *, wrap_with_ddp, ddp_config):
        return ["model-chunk"]


@pytest.fixture
def fake_bridge(monkeypatch):
    provider = _Provider()
    auto_bridge = MagicMock()
    auto_bridge.from_hf_pretrained.return_value = MagicMock(to_megatron_provider=MagicMock(return_value=provider))
    modules = {
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.conversion": types.ModuleType("megatron.bridge.models.conversion"),
        "megatron.bridge.models.conversion.model_bridge": types.ModuleType("mb"),
        "megatron.bridge.training": types.ModuleType("megatron.bridge.training"),
        "megatron.bridge.training.config": types.ModuleType("cfg"),
        "megatron.bridge.utils": types.ModuleType("megatron.bridge.utils"),
        "megatron.bridge.utils.fusions": types.ModuleType("fusions"),
    }
    modules["megatron.bridge"].AutoBridge = auto_bridge
    modules["megatron.bridge.models.conversion.model_bridge"]._megatron_local_name_to_global = MagicMock()
    modules["megatron.bridge.training.config"].DistributedDataParallelConfig = MagicMock()
    modules["megatron.bridge.utils.fusions"].validate_rope_fusion_compatibility = lambda p: True
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        lora_bridge, "load_hf_config", lambda path: SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    )
    monkeypatch.setattr(lora_bridge, "HfWeightMapping", MagicMock())
    monkeypatch.setattr(lora_bridge, "apply_dsa_backend_args", lambda provider, args: None)
    return provider


def _args(**overrides):
    base = dict(
        hf_checkpoint="/ckpt",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        gradient_accumulation_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=None,
        distribute_saved_activations=False,
        attention_backend="auto",
        apply_rope_fusion=False,
        bias_swiglu_fusion=False,
        moe_router_dtype=None,
        moe_router_use_torch_mm=True,
        multi_lora_n_adapters=0,
        lora_type="lora",
        optimizer="adam",
        accumulate_allreduce_grads_in_fp32=False,
        offload_train=False,
        calculate_per_token_loss=False,
    )
    return Namespace(**{**base, **overrides})


class TestLoraBridgeCarriesPerTokenLoss:
    @pytest.mark.parametrize("value", [True, False])
    def test_the_provider_config_matches_the_argument_before_finalize(self, fake_bridge, value):
        assert lora_bridge._setup_lora_model_via_bridge(_args(calculate_per_token_loss=value)) == ["model-chunk"]
        assert fake_bridge.finalized_with["calculate_per_token_loss"] is value

    def test_the_default_stays_false(self, fake_bridge):
        lora_bridge._setup_lora_model_via_bridge(_args())
        assert fake_bridge.calculate_per_token_loss is False


def _megatron_scaled_grad(per_token_args: bool, config_per_token: bool, lengths, adv, n_mb):
    """Gradient of one DP rank's microbatches under Miles loss.py scaling + Megatron forward_step scaling.

    Transcribed from the 1b offline report (repro_token_grad.py): Miles returns ``(loss, num_tokens)``;
    Megatron's schedule divides by num_tokens and num_microbatches only when config.calculate_per_token_loss is
    False, and with it True finalize_model_grads divides by the total token count instead.
    """
    from miles.backends.training_utils import cp_utils

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
        from miles.backends.training_utils import cp_utils

        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))

    def test_with_the_config_set_per_token_differs_from_sample_mean_for_unequal_lengths(self):
        lengths, adv = [3, 7, 5, 9], torch.tensor([1.0, -1.0, 0.5, -0.5])
        sample = _megatron_scaled_grad(False, False, lengths, adv, n_mb=2)
        token = _megatron_scaled_grad(True, True, lengths, adv, n_mb=2)
        assert not torch.allclose(sample, token)

    def test_the_missing_config_with_one_sample_per_microbatch_collapses_to_sample_mean(self):
        """The pre-fix LoRA path: per-microbatch token mean / n_mb == sample mean when each microbatch is one
        sample, i.e. a candidate explanation for the bit-identical GPU gradients in 1b (to confirm on GPU)."""
        lengths, adv = [3, 7, 5, 9], torch.tensor([1.0, -1.0, 0.5, -0.5])
        sample = _megatron_scaled_grad(False, False, lengths, adv, n_mb=4)
        buggy_token = _megatron_scaled_grad(True, False, lengths, adv, n_mb=4)
        torch.testing.assert_close(buggy_token, sample)
        fixed_token = _megatron_scaled_grad(True, True, lengths, adv, n_mb=4)
        assert not torch.allclose(fixed_token, sample)
