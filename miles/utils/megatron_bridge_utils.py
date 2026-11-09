import argparse
import functools
import json
from collections.abc import Mapping
from contextlib import contextmanager
from types import MappingProxyType

from miles.backends.training_utils.model_companion import ModelCompanionInstallationUtils
from miles.utils.replay_base import reject_router_fusion_with_routing_replay

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def load_auto_bridge(hf_checkpoint: str):
    from megatron.bridge import AutoBridge

    import miles_plugins.megatron_bridge  # noqa: F401

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):  # config-access-exempt: Bridge module capabilities vary by architecture
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):  # config-access-exempt: Bridge module capabilities vary by architecture
                module._maintain_float32_expert_bias()

    try:
        with ModelCompanionInstallationUtils.hide(model):
            yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_dsa_backend_args(provider, args) -> None:
    """Resolve DSV4 kernel selection separately from the GLM DSA backend option."""
    explicit = getattr(args.backend, "dsa_kernel_backend", None)  # config-access-exempt: older Megatron parsers omit the DSA kernel backend switch
    if getattr(provider, "experimental_attention_variant", None) == "dsv4_hybrid":  # config-access-exempt: Megatron providers without experimental attention omit this field
        provider.dsa_kernel_backend = explicit or "cudnn"
        return

    backend = args.dsa_attention_backend
    if hasattr(provider, "dsa_attention_backend"):  # config-access-exempt: third-party providers differ in dsa_attention_backend support
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):  # config-access-exempt: third-party providers differ in dsa_kernel_backend support
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]


# MoE runtime flags that the bridge path copies onto the provider only when they are set
# (model_provider._apply_bridge_runtime_config). Megatron's parser builds each flag from the
# TransformerConfig field of the same name, with the field default as the CLI default. A bridge provider
# may hold its own value on purpose (Nemotron-H sets moe_permute_fusion and moe_shared_expert_overlap to
# True), and args hold the CLI default when a flag is absent, so a field is copied only when args differ
# from the CLI default, which is read from the installed Megatron's parser at runtime (the defaults move
# between Megatron versions: --moe-deepep-num-sms is None in one and 20 in another). A store-true flag can
# therefore turn a provider setting on, never off.
BRIDGE_MOE_RUNTIME_ARGS = (
    "moe_flex_dispatcher_backend",
    "moe_flex_dispatcher_num_sms",
    "moe_deepep_num_sms",  # deprecated alias; TransformerConfig.__post_init__ routes it
    "moe_hybridep_num_sms",  # deprecated alias; same
    "moe_enable_deepep",  # deprecated; TransformerConfig.__post_init__ maps it to the deepep backend
    "moe_permute_fusion",
    "moe_router_fusion",
    "moe_shared_expert_overlap",
    "overlap_moe_expert_parallel_comm",
    "delay_wgrad_compute",
)


@functools.cache
def megatron_cli_defaults() -> Mapping[str, object]:
    """``dest -> default`` of every argument the installed Megatron's ``add_megatron_arguments`` declares.

    Read once per process from the parser itself, so the "is this flag set" decision follows the Megatron
    version that is actually installed instead of a table written for one version.
    """
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    return MappingProxyType({action.dest: action.default for action in parser._actions if action.dest != argparse.SUPPRESS})


def bridge_moe_runtime_overrides(args, *, cli_defaults: Mapping[str, object] | None = None) -> dict:
    """The BRIDGE_MOE_RUNTIME_ARGS fields that ``args.backend`` sets to a value other than the CLI default.

    ``cli_defaults`` defaults to :func:`megatron_cli_defaults`. A Megatron version without one of the flags
    (the deprecated aliases come and go) declares no default and leaves it out of the backend namespace; it
    then counts as unset. A backend namespace that carries none of the flags (callers that register only
    part of the Megatron arguments) sets nothing and does not read the parser.
    """
    backend = vars(args.backend)
    present = [name for name in BRIDGE_MOE_RUNTIME_ARGS if name in backend]
    if not present:
        return {}
    if cli_defaults is None:
        cli_defaults = megatron_cli_defaults()
    return {name: backend[name] for name in present if name in cli_defaults and backend[name] != cli_defaults[name]}


def apply_bridge_moe_runtime_args(provider, args, *, cli_defaults: Mapping[str, object] | None = None) -> None:
    """Copy the MoE runtime flags set in args onto a bridge-built provider, before ``finalize()``.

    Raises instead of dropping a setting: for a set flag that the provider has no field for, for
    ``--cuda-graph-impl``, and for router fusion together with routing replay. ``cli_defaults`` is for
    tests; production reads the installed Megatron's parser.
    """
    _reject_bridge_cuda_graphs(cuda_graph_impl=vars(args.backend).get("cuda_graph_impl", "none"))
    for name, value in bridge_moe_runtime_overrides(args, cli_defaults=cli_defaults).items():
        if not hasattr(provider, name):  # config-access-exempt: bridge providers differ in the MoE fields they carry
            raise AttributeError(f"--{name.replace('_', '-')} is set, but the bridge provider {type(provider).__name__} has no field {name!r}")
        setattr(provider, name, value)
    router_fusion = bool(
        getattr(provider, "moe_router_fusion", False)  # config-access-exempt: older providers lack router fusion
    )
    reject_router_fusion_with_routing_replay(router_fusion=router_fusion, routing_replay=args.use_routing_replay or args.use_rollout_routing_replay)


def describe_bridge_moe_runtime_config(provider, args, *, cli_defaults: Mapping[str, object] | None = None) -> str:
    """A JSON line: the MoE runtime fields of a bridge provider and the ones that args set."""
    names = ("moe_token_dispatcher_type", *BRIDGE_MOE_RUNTIME_ARGS, "cuda_graph_impl")
    return json.dumps(
        {
            "effective": {
                name: getattr(provider, name, "<absent>")  # config-access-exempt: logs whatever the provider carries
                for name in names
            },
            "from_args": sorted(bridge_moe_runtime_overrides(args, cli_defaults=cli_defaults)),
        },
        default=str,
        sort_keys=True,
    )


def _reject_bridge_cuda_graphs(*, cuda_graph_impl) -> None:
    # The bridge copy does not carry the CUDA graph settings (modules, warmup, RNG tracker, ...), and
    # copying the impl alone would capture graphs with the provider's defaults for the rest. Fail
    # instead of training without graphs while args claim otherwise.
    if cuda_graph_impl not in (None, "none"):
        raise ValueError(f"--cuda-graph-impl {cuda_graph_impl} is not supported with --megatron-to-hf-mode bridge: the bridge provider does not receive the CUDA graph settings, so the flag would have no effect. Remove it, or build the model with --custom-model-provider-path.")
