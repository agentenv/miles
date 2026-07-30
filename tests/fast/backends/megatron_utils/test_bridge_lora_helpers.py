from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_bridge_lora_provider_uses_requested_attention_backend():
    from miles.backends.megatron_utils.bridge_lora_helpers import (
        _setup_lora_model_via_bridge,
    )

    provider = MagicMock()
    provider.attention_backend = "auto"
    provider.num_layers_in_first_pipeline_stage = None
    provider.num_layers_in_last_pipeline_stage = None
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider
    lora = MagicMock(side_effect=lambda model: model)
    args = Namespace(
        hf_checkpoint="model",
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
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        dsa_attention_backend="megatron",
        attention_backend="unfused",
        external_policy_sync_path="policy.sync",
        use_distributed_optimizer=False,
        accumulate_allreduce_grads_in_fp32=True,
        optimizer="adam",
        offload_train=False,
    )

    with (
        patch(
            "megatron.bridge.AutoBridge.from_hf_pretrained",
            return_value=bridge,
        ),
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers.load_hf_config",
            return_value=SimpleNamespace(architectures=["ForCausalLM"]),
        ),
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers.is_multi_lora_enabled",
            return_value=False,
        ),
        patch(
            "miles.backends.megatron_utils.lora_utils.create_lora_instance",
            return_value=lora,
        ),
        patch(
            "megatron.bridge.training.config.DistributedDataParallelConfig"
        ) as ddp_config,
    ):
        provider.provide_distributed_model.return_value = [object()]
        _setup_lora_model_via_bridge(args)

    assert provider.attention_backend == "unfused"
    ddp_config.return_value.finalize.assert_called_once_with()
