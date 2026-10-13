"""A full-parameter LoRA hybrid trains base parameters, so every publish carries base weights."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


def _lora_updater(*, hybrid: bool) -> tuple[WeightUpdater, MagicMock]:
    protocol = SimpleNamespace(
        use_weight_update_session=False,
        needs_base_resync_for_lora=False,
        is_sender=True,
        group_name="test",
        rollout_engines=[],
        required_placement=MagicMock(),
        supports_lora=True,
        begin_sync=lambda weight_version, iter_buckets: True,
        send_bucket=MagicMock(),
        after_base_weights=MagicMock(),
        finalize=MagicMock(),
        after_engines_resumed=MagicMock(),
    )
    iterator = MagicMock()
    iterator.iter_hf_weights.return_value = iter([])
    args = Namespace(check_lora_weight_equal=False, full_parameter_lora_hybrid=hybrid)
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        updater = WeightUpdater(
            args,
            [MagicMock()],
            weights_getter=lambda: {},
            model_name="deepseekv4",
            quantization_config=None,
            iterator_factory=lambda *a, **k: iterator,
            parallel_state=MagicMock(),
            is_lora=True,
            lora_sync_config={"r": 8},
        )
    return updater, iterator


@pytest.mark.parametrize("hybrid", [False, True])
def test_lora_publish_carries_base_weights_only_for_the_hybrid(hybrid):
    updater, iterator = _lora_updater(hybrid=hybrid)
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert iterator.iter_hf_weights.call_args.kwargs["include_base"] is hybrid
    assert iterator.iter_hf_weights.call_args.kwargs["adapters"]
