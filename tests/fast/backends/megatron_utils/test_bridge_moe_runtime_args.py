"""CPU tests: MoE runtime flags reach a bridge-built provider, and the unsafe ones are rejected.

Bridge mode builds the model config from the HF checkpoint, so a command-line flag reaches the
model only through ``model_provider._apply_bridge_runtime_config``. These tests run that function
on a stand-in provider with the Nemotron-H bridge defaults; no Megatron model, GPU or Ray.
"""

import argparse
import dataclasses
import json
from collections.abc import Callable

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils import megatron_bridge_utils
from miles.utils.megatron_bridge_utils import (
    BRIDGE_MOE_RUNTIME_ARGS,
    apply_bridge_moe_runtime_args,
    bridge_moe_runtime_overrides,
    describe_bridge_moe_runtime_config,
)
from miles.utils.replay_base import reject_router_fusion_with_routing_replay

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# The CLI defaults of radixark/Megatron-LM a84b105, the version the RL island runs used. Production reads
# them from the installed parser (megatron_cli_defaults); these tests pin them so they run without Megatron.
_CLI_DEFAULTS = {
    "moe_flex_dispatcher_backend": "deepep",
    "moe_flex_dispatcher_num_sms": None,
    "moe_deepep_num_sms": None,
    "moe_hybridep_num_sms": None,
    "moe_enable_deepep": False,
    "moe_permute_fusion": False,
    "moe_router_fusion": False,
    "moe_shared_expert_overlap": False,
    "overlap_moe_expert_parallel_comm": False,
    "delay_wgrad_compute": False,
    "cuda_graph_impl": "none",
}
assert set(BRIDGE_MOE_RUNTIME_ARGS) < set(_CLI_DEFAULTS)
_PARSER_CLI_DEFAULTS = megatron_bridge_utils.megatron_cli_defaults  # the real reader, before the fixture pins it


@pytest.fixture(autouse=True)
def _pinned_cli_defaults(monkeypatch):
    monkeypatch.setattr(megatron_bridge_utils, "megatron_cli_defaults", lambda: _CLI_DEFAULTS)


@dataclasses.dataclass
class _NemotronHProvider:
    """The MoE runtime fields of a Nemotron-H bridge provider before Miles copies args onto it.

    ``NemotronHBridge.provider_bridge`` sets the dispatcher to alltoall and turns permute fusion and
    shared-expert overlap on; the other fields keep the TransformerConfig defaults.
    """

    moe_token_dispatcher_type: str = "alltoall"
    moe_flex_dispatcher_backend: str = "deepep"
    moe_flex_dispatcher_num_sms: int | None = None
    moe_deepep_num_sms: int | None = None
    moe_hybridep_num_sms: int | None = None
    moe_enable_deepep: bool = False
    moe_permute_fusion: bool = True
    moe_router_fusion: bool = False
    moe_shared_expert_overlap: bool = True
    overlap_moe_expert_parallel_comm: bool = False
    delay_wgrad_compute: bool = False
    cuda_graph_impl: str = "none"
    mtp_num_layers: int | None = None
    mtp_detach_heads: bool = False


_TOP_LEVEL = ("use_routing_replay", "use_rollout_routing_replay")


def _megatron_args(**overrides) -> argparse.Namespace:
    """The args of an RL island run (G-AN Xdeepep / s27-ba-a): flex + DeepEP, permute fusion, R3.

    Megatron flags live on ``args.backend``; routing replay is a Miles (AlgoConfig) field on ``args``.
    """
    backend = argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        gradient_accumulation_fusion=True,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity="selective",
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=["moe", "layernorm"],
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe="delayed",
        attention_backend="auto",
        mtp_num_layers=None,
        mtp_detach_heads=False,
        moe_token_dispatcher_type="flex",
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        moe_router_bias_update_rate=None,
        moe_aux_loss_coeff=None,
        mamba_training_ssm_states_dtype=None,
        **_CLI_DEFAULTS,
    )
    backend.moe_permute_fusion = True
    args = argparse.Namespace(
        backend=backend,
        dsa_attention_backend=None,
        enable_mtp_training=False,
        use_routing_replay=True,
        use_rollout_routing_replay=True,
    )
    for name, value in overrides.items():
        setattr(args if name in _TOP_LEVEL else backend, name, value)
    return args


def test_an_rl_island_run_keeps_the_provider_moe_settings(apply_bridge_runtime_config: Callable) -> None:
    """The args of the existing runs must build the model they built before this copy existed."""
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, _megatron_args())

    assert provider.moe_token_dispatcher_type == "flex"
    assert provider.moe_flex_dispatcher_backend == "deepep"
    assert provider.moe_permute_fusion is True
    # args hold the CLI default False (the flag is absent); the provider's True stays
    assert provider.moe_shared_expert_overlap is True
    assert provider.moe_router_fusion is False
    assert provider.moe_flex_dispatcher_num_sms is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("moe_flex_dispatcher_backend", "deepepv2"),
        ("moe_flex_dispatcher_backend", "ncclep"),
        ("moe_flex_dispatcher_num_sms", 32),
        ("moe_deepep_num_sms", 24),
        ("moe_hybridep_num_sms", 16),
        ("moe_enable_deepep", True),
        ("overlap_moe_expert_parallel_comm", True),
        ("delay_wgrad_compute", True),
    ],
)
def test_a_set_flag_reaches_the_provider(apply_bridge_runtime_config: Callable, name: str, value: object) -> None:
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, _megatron_args(**{name: value}))

    assert getattr(provider, name) == value


@pytest.mark.parametrize("name", ["moe_permute_fusion", "moe_shared_expert_overlap"])
def test_a_store_true_flag_turns_a_provider_setting_on(apply_bridge_runtime_config: Callable, name: str) -> None:
    provider = _NemotronHProvider(**{name: False})

    apply_bridge_runtime_config(provider, _megatron_args(**{name: True}))

    assert getattr(provider, name) is True


def test_router_fusion_reaches_the_provider_without_routing_replay(apply_bridge_runtime_config: Callable) -> None:
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, _megatron_args(moe_router_fusion=True, use_routing_replay=False, use_rollout_routing_replay=False))

    assert provider.moe_router_fusion is True


def test_args_without_the_moe_flags_leave_the_provider_unchanged(apply_bridge_runtime_config: Callable) -> None:
    """Callers that register only part of the arguments (the other bridge tests) must keep working."""
    args = _megatron_args(use_routing_replay=False, use_rollout_routing_replay=False)
    for name in _CLI_DEFAULTS:
        delattr(args.backend, name)
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, args)

    assert dataclasses.replace(provider, moe_token_dispatcher_type="alltoall") == _NemotronHProvider()


def test_a_set_flag_without_a_provider_field_raises(apply_bridge_runtime_config: Callable) -> None:
    @dataclasses.dataclass
    class _OlderProvider:
        moe_token_dispatcher_type: str = "alltoall"
        moe_router_fusion: bool = False

    with pytest.raises(AttributeError, match="--moe-flex-dispatcher-num-sms is set"):
        apply_bridge_runtime_config(_OlderProvider(), _megatron_args(moe_flex_dispatcher_num_sms=32))


@pytest.mark.parametrize(
    ("use_routing_replay", "use_rollout_routing_replay"),
    [(True, True), (True, False), (False, True)],
    ids=["r3", "r2", "r3-before-validation"],
)
def test_router_fusion_with_routing_replay_is_rejected(apply_bridge_runtime_config: Callable, use_routing_replay: bool, use_rollout_routing_replay: bool) -> None:
    """The fused top-k returns before the replay hook, so replay would be skipped without an error."""
    args = _megatron_args(
        moe_router_fusion=True,
        use_routing_replay=use_routing_replay,
        use_rollout_routing_replay=use_rollout_routing_replay,
    )

    with pytest.raises(ValueError, match="cannot be used with routing replay"):
        apply_bridge_runtime_config(_NemotronHProvider(), args)


def test_a_provider_that_fuses_the_router_is_rejected_under_replay(apply_bridge_runtime_config: Callable) -> None:
    """The check reads the provider, so a bridge that turns router fusion on by itself is caught too."""
    with pytest.raises(ValueError, match="cannot be used with routing replay"):
        apply_bridge_runtime_config(_NemotronHProvider(moe_router_fusion=True), _megatron_args())


@pytest.mark.parametrize("impl", ["local", "transformer_engine", "full_iteration"])
def test_cuda_graphs_are_rejected_in_bridge_mode(apply_bridge_runtime_config: Callable, impl: str) -> None:
    with pytest.raises(ValueError, match=f"--cuda-graph-impl {impl} is not supported"):
        apply_bridge_runtime_config(_NemotronHProvider(), _megatron_args(cuda_graph_impl=impl))


def test_the_log_line_names_the_effective_values_and_the_copied_flags() -> None:
    provider = _NemotronHProvider(moe_token_dispatcher_type="flex")
    args = _megatron_args(moe_flex_dispatcher_backend="deepepv2", moe_flex_dispatcher_num_sms=32)
    apply_bridge_moe_runtime_args(provider, args)

    line = json.loads(describe_bridge_moe_runtime_config(provider, args))

    assert line["from_args"] == ["moe_flex_dispatcher_backend", "moe_flex_dispatcher_num_sms", "moe_permute_fusion"]
    assert line["effective"]["moe_token_dispatcher_type"] == "flex"
    assert line["effective"]["moe_flex_dispatcher_backend"] == "deepepv2"
    assert line["effective"]["moe_flex_dispatcher_num_sms"] == 32
    assert line["effective"]["moe_shared_expert_overlap"] is True
    assert line["effective"]["cuda_graph_impl"] == "none"


def test_a_flag_counts_as_set_against_the_installed_parsers_default() -> None:
    """--moe-deepep-num-sms defaults to None in one Megatron and 20 in another; the parser decides, not a table."""
    args = _megatron_args(moe_deepep_num_sms=20, moe_permute_fusion=False)

    assert bridge_moe_runtime_overrides(args, cli_defaults=_CLI_DEFAULTS) == {"moe_deepep_num_sms": 20}
    assert bridge_moe_runtime_overrides(args, cli_defaults={**_CLI_DEFAULTS, "moe_deepep_num_sms": 20}) == {}
    assert bridge_moe_runtime_overrides(
        _megatron_args(moe_deepep_num_sms=24, moe_permute_fusion=False),
        cli_defaults={**_CLI_DEFAULTS, "moe_deepep_num_sms": 20},
    ) == {"moe_deepep_num_sms": 24}


def test_a_flag_the_installed_megatron_lacks_is_never_copied() -> None:
    """An older parser declares no default for the flag, so a stray value in args does not reach the provider."""
    defaults = {name: value for name, value in _CLI_DEFAULTS.items() if name != "moe_hybridep_num_sms"}

    args = _megatron_args(moe_hybridep_num_sms=16, moe_permute_fusion=False)

    assert bridge_moe_runtime_overrides(args, cli_defaults=defaults) == {}


def test_the_defaults_come_from_megatrons_parser() -> None:
    """Where Megatron is importable: the parser's own defaults are read, and a backend at those defaults sets nothing."""
    pytest.importorskip("megatron.training.arguments")
    try:
        defaults = _PARSER_CLI_DEFAULTS()
    except (AttributeError, TypeError) as exc:  # a stubbed megatron package in CPU-only test environments
        pytest.skip(f"megatron.training.arguments is not a real parser here: {exc}")

    assert defaults["num_experts"] is None and "moe_token_dispatcher_type" in defaults
    declared = [name for name in BRIDGE_MOE_RUNTIME_ARGS if name in defaults]
    assert declared, "the installed Megatron declares none of the MoE runtime flags"
    backend = argparse.Namespace(**{name: defaults[name] for name in declared})
    args = argparse.Namespace(backend=backend)
    assert bridge_moe_runtime_overrides(args, cli_defaults=defaults) == {}


@pytest.mark.parametrize(
    ("router_fusion", "routing_replay", "rejected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_the_validation_check_rejects_only_fusion_with_replay(router_fusion: bool, routing_replay: bool, rejected: bool) -> None:
    """miles_validate_args calls this before any worker starts; the bridge path calls it on the provider."""
    if rejected:
        with pytest.raises(ValueError, match="cannot be used with routing replay"):
            reject_router_fusion_with_routing_replay(router_fusion=router_fusion, routing_replay=routing_replay)
    else:
        reject_router_fusion_with_routing_replay(router_fusion=router_fusion, routing_replay=routing_replay)
