"""Unit tests for LoRA-related helpers in miles.backends.megatron_utils.checkpoint.

Covers pure path-detection functions and the LoRA branch routing in
save_checkpoint_with_lora / load_checkpoint — the latter using mocks to avoid
GPU / distributed requirements.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.checkpoint import (
    _is_megatron_checkpoint,
    _omit_fresh_critic_head_from_sharded_state,
    load_checkpoint,
    save_checkpoint_with_lora,
)

# ---------------------------------------------------------------------------
# _is_megatron_checkpoint
# ---------------------------------------------------------------------------


class TestIsMegatronCheckpoint:
    def test_has_latest_file(self, tmp_path):
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
        assert _is_megatron_checkpoint(tmp_path) is True

    def test_iter_dir_name(self, tmp_path):
        iter_dir = tmp_path / "iter_0000100"
        iter_dir.mkdir()
        assert _is_megatron_checkpoint(iter_dir) is True

    def test_regular_dir(self, tmp_path):
        assert _is_megatron_checkpoint(tmp_path) is False

    def test_hf_checkpoint_dir(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")
        assert _is_megatron_checkpoint(tmp_path) is False

    @pytest.mark.parametrize(
        "name",
        [
            "iter_0000001",
            "iter_0000000",
            "iter_9999999",
        ],
    )
    def test_valid_iter_patterns(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is True

    @pytest.mark.parametrize(
        "name",
        [
            "iter_123",  # too short
            "iter_00000001",  # too long
            "iteration_0000001",
            "checkpoint",
        ],
    )
    def test_invalid_iter_patterns(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is False


# ---------------------------------------------------------------------------
# save_checkpoint_with_lora — branch routing
# ---------------------------------------------------------------------------


class TestSaveCheckpointWithLoRA:
    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_lora_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=True)
    def test_lora_model_saves_adapter(self, mock_is_lora, mock_save_lora, mock_get_args, tmp_path):
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock())

        mock_save_lora.assert_called_once()
        call_args = mock_save_lora.call_args
        assert "adapter" in call_args[1].get("save_dir", call_args[0][2] if len(call_args[0]) > 2 else "")

    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=False)
    def test_non_lora_model_saves_regular(self, mock_is_lora, mock_save_ckpt, mock_get_args, tmp_path):
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()


class _FakeShard:
    def __init__(self, key):
        self.key = key


class _FakeCriticModule:
    output_layer = object()

    def __init__(self):
        self.calls = 0

    def sharded_state_dict(self):
        self.calls += 1
        return {
            "decoder.layers.0.weight": _FakeShard("decoder.layers.0.weight"),
            "output_layer.weight": _FakeShard("output_layer.weight"),
            "output_layer.bias": _FakeShard("output_layer.bias"),
        }


class _FakeWrapper:
    role = "critic"

    def __init__(self, module):
        self.module = module


def test_fresh_critic_bootstrap_filters_only_value_head_and_restores_method():
    module = _FakeCriticModule()
    original_method = module.sharded_state_dict

    with _omit_fresh_critic_head_from_sharded_state([_FakeWrapper(module)]) as omitted:
        assert set(module.sharded_state_dict()) == {"decoder.layers.0.weight"}

    assert omitted == {"output_layer.weight", "output_layer.bias"}
    assert set(module.sharded_state_dict()) == {
        "decoder.layers.0.weight",
        "output_layer.weight",
        "output_layer.bias",
    }
    assert module.sharded_state_dict.__func__ is original_method.__func__


@patch("miles.backends.megatron_utils.checkpoint.is_lora_enabled", return_value=False)
@patch("miles.backends.megatron_utils.checkpoint.get_args")
@patch("miles.backends.megatron_utils.checkpoint._load_checkpoint_megatron")
def test_checkpoint_load_bootstraps_backbone_but_not_fresh_critic_head(
    mock_megatron_load,
    mock_get_args,
    _mock_lora_enabled,
    tmp_path,
):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    module = _FakeCriticModule()
    wrapped = _FakeWrapper(module)
    mock_get_args.return_value = Namespace(
        load=str(tmp_path),
        _critic_bootstrap_from_actor_checkpoint=True,
    )

    def observe_requested_state(**kwargs):
        assert kwargs["ddp_model"] == [wrapped]
        assert set(module.sharded_state_dict()) == {"decoder.layers.0.weight"}
        return 0, 0

    mock_megatron_load.side_effect = observe_requested_state

    assert load_checkpoint([wrapped], None, None, None, False) == (0, 0)
    assert set(module.sharded_state_dict()) == {
        "decoder.layers.0.weight",
        "output_layer.weight",
        "output_layer.bias",
    }


@patch("miles.backends.megatron_utils.checkpoint.is_lora_enabled", return_value=False)
@patch("miles.backends.megatron_utils.checkpoint.get_args")
@patch("miles.backends.megatron_utils.checkpoint._load_checkpoint_megatron")
def test_contracted_critic_resume_requests_trained_value_head(
    mock_megatron_load,
    mock_get_args,
    _mock_lora_enabled,
    tmp_path,
):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("8")
    module = _FakeCriticModule()
    wrapped = _FakeWrapper(module)
    mock_get_args.return_value = Namespace(
        load=str(tmp_path),
        _critic_bootstrap_from_actor_checkpoint=False,
    )

    def observe_requested_state(**_kwargs):
        assert set(module.sharded_state_dict()) == {
            "decoder.layers.0.weight",
            "output_layer.weight",
            "output_layer.bias",
        }
        return 8, 0

    mock_megatron_load.side_effect = observe_requested_state

    assert load_checkpoint([wrapped], None, None, None, False) == (8, 0)
