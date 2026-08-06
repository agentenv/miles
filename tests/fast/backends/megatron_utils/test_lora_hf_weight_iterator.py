"""Unit tests for HfWeightIteratorBase factory routing.

Validates that the right iterator subclass is selected based on megatron_to_hf_mode.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase

_BASE_MODULE = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_base"


class TestHfWeightIteratorFactory:
    def _make_args(self, mode="bridge"):
        return Namespace(
            megatron_to_hf_mode=mode,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_bridge_mode_creates_bridge_iterator(self, mock_init):
        """Factory should select HfWeightIteratorBridge for 'bridge' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

        with patch.object(HfWeightIteratorBridge, "__init__", return_value=None):
            args = self._make_args("bridge")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=True, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorBridge)

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_raw_mode_creates_direct_iterator(self, mock_init):
        """Factory should select HfWeightIteratorDirect for 'raw' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        with patch.object(HfWeightIteratorDirect, "__init__", return_value=None):
            args = self._make_args("raw")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorDirect)

    def test_invalid_mode_raises(self):
        args = self._make_args("invalid_mode")
        with pytest.raises(KeyError):
            HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )


def test_bridge_lora_materializes_cpu_weights_before_ipc(monkeypatch):
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as bridge_module

    calls = []

    class Bridge:
        def export_adapter_weights(self, model, *, cpu, show_progress):
            calls.append((model, cpu, show_progress))
            return [("layer.lora_A.weight", torch.ones(1), "adapter.weight")]

    iterator = object.__new__(bridge_module.HfWeightIteratorBridge)
    iterator._bridge = Bridge()
    iterator.model = [object()]
    iterator.model_name = "qwen"
    iterator.args = Namespace(update_weight_buffer_size=1024)
    iterator.quantization_config = None
    iterator._postprocess_and_quantize = lambda weights, _weight_type: weights
    monkeypatch.setattr(bridge_module, "get_atomic_update_groups", lambda *_args: [])
    monkeypatch.setattr(
        bridge_module.megatron_bridge_utils,
        "patch_megatron_model",
        lambda _model: nullcontext(),
    )

    chunks = list(iterator.get_hf_weight_chunks({}, weight_type="lora"))

    assert calls == [(iterator.model, True, False)]
    assert chunks[0][0][0] == "layer.lora_A.weight"
