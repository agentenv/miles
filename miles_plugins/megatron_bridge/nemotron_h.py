"""Miles integration for ``megatron.bridge`` Nemotron-H (Mamba+Attention hybrid).

This plugin is a non-invasive drop-in that:

1. Registers :class:`MilesNemotronHBridge` for ``NemotronHForCausalLM`` (overriding
   the upstream :class:`NemotronHBridge`) so MoE variants (Nano-30B-A3B,
   Super-120B-A12B) round-trip correctly through the HF↔Megatron
   ``mapping_registry`` and get the right ``moe_router_*`` fields on the
   provider.
2. For Nemotron-H models, patches
   :class:`~megatron.core.transformer.transformer_layer.TransformerLayer` so
   hybrid layers whose ``self_attention`` / ``mlp`` is an ``IdentityOp`` don't
   blow up on the 2-tuple unpack in ``_forward_attention`` / ``_forward_mlp``.
3. For Nemotron-H models, patches
   :class:`~megatron.core.models.mamba.MambaModel` ``forward`` to
   adapt Miles' ``mtp_kwargs`` to the model's forward arguments while
   preserving ``loss_mask``.
4. Registers :class:`MilesNemotronHLanguageOnlyBridge` for the Nemotron-H VL
   checkpoint (``NemotronH_Omni_Reasoning_V3``, Nemotron-3.5 Super VL): Megatron
   trains its language model only, the vision part stays frozen (see the class).

Importing this module is idempotent — safe to import multiple times.
"""

from __future__ import annotations

import copy
import logging

import torch

logger = logging.getLogger(__name__)


_NEMOTRONH_MOE_MAPPINGS: dict[str, str] = {
    "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
    "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
    # Routed experts: up-only FFN (nemotron_h uses squared_relu, no gate).
    "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
    "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "backbone.layers.*.mixer.experts.*.up_proj.weight",
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "backbone.layers.*.mixer.experts.*.down_proj.weight",
    "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
    # MoE latent projections (Super-120B-A12B uses moe_latent_size=1024 to bottleneck
    # expert input/output; Nano variants leave moe_latent_size=None and these tensors
    # do not exist, so the mappings stay unreferenced).
    "decoder.layers.*.mlp.fc1_latent_proj.weight": "backbone.layers.*.mixer.fc1_latent_proj.weight",
    "decoder.layers.*.mlp.fc2_latent_proj.weight": "backbone.layers.*.mixer.fc2_latent_proj.weight",
}

# Nemotron-H VL checkpoints (Nemotron-3.5 Super VL 120B-A12B): a C-RADIO vision
# encoder (``vision_model.*``), a projector (``mlp1.*``, ``vision_projector.*``)
# and a NemotronHForCausalLM stored under ``language_model.`` (``backbone.*``,
# ``lm_head.*`` and the MTP block ``mtp.*``).
NEMOTRON_H_OMNI_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
LANGUAGE_MODEL_PREFIX = "language_model."


def language_model_config(hf_config):
    """The NemotronH config of a VL checkpoint config (``llm_config``)."""
    llm_config = getattr(hf_config, "llm_config", None)
    if llm_config is None:
        raise ValueError(f"{NEMOTRON_H_OMNI_ARCHITECTURE} config has no llm_config")
    return llm_config


class LanguageModelView:
    """An HF pretrained wrapper seen through its language model.

    ``config`` is the wrapper config's ``llm_config``; every other attribute
    (weights state, path, tokenizer) is the wrapper's own, so weights are read
    under their checkpoint names.
    """

    def __init__(self, hf_pretrained):
        self._hf_pretrained = hf_pretrained
        self.config = language_model_config(hf_pretrained.config)

    def __getattr__(self, name):
        return getattr(self._hf_pretrained, name)


def prefixed_mapping(mapping, prefix: str):
    """A copy of a bridge parameter mapping whose HF name(s) carry ``prefix``.

    Copies keep the mapping class, so the conversion (QKV split, Mamba conv1d
    chunks, MTP index flattening) is unchanged; only the HF side moves.
    """
    copied = copy.copy(mapping)
    if isinstance(mapping.hf_param, str):
        copied.hf_param = prefix + mapping.hf_param
    else:
        copied.hf_param = {key: prefix + value for key, value in mapping.hf_param.items()}
    return copied


# HF name → provider name → cast. Miles' generic ``model_provider.py`` does not
# forward ``moe_router_topk_scaling_factor`` by default, which silently runs
# Megatron with scaling=1.0 instead of the HF config's (e.g.) 2.5 — producing
# ~0.28 train-vs-rollout logprob drift for Nano-30B-A3B.
_NEMOTRONH_MOE_ROUTING_FIELDS: tuple[tuple[str, str, type], ...] = (
    ("routed_scaling_factor", "moe_router_topk_scaling_factor", float),
    ("n_group", "moe_router_num_groups", int),
    ("topk_group", "moe_router_group_topk", int),
)


def _build_bridge_subclass():
    """Build the ``MilesNemotronHBridge`` class lazily so importing this module
    does not force megatron.bridge.models.nemotronh to load until first use.
    """
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import AutoMapping, MambaConv1dMapping
    from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
    from megatron.core.models.mamba import MambaModel

    @MegatronModelBridge.register_bridge(source="NemotronHForCausalLM", target=MambaModel)
    class MilesNemotronHBridge(NemotronHBridge):
        """Nemotron-H bridge with MoE support on top of upstream dense bridge.

        Upstream :class:`NemotronHBridge` only handles the dense Mamba+Attention
        variants. The Nano-30B-A3B / Super-120B-A12B MoE variants add router,
        routed-expert, and shared-expert tensors under ``backbone.layers.*.mixer.*``
        and need ``num_moe_experts`` / ``moe_router_*`` on the provider.
        """

        def provider_bridge(self, hf_pretrained):
            _install_nemotronh_hybrid_layer_shims()
            _install_mamba_model_loss_mask_shim()
            provider = super().provider_bridge(hf_pretrained)
            hf = hf_pretrained.config

            n_exp = int(getattr(hf, "num_experts", None) or getattr(hf, "n_routed_experts", None) or 0)
            if n_exp == 0:
                return provider

            provider.num_moe_experts = n_exp
            provider.moe_router_topk = int(getattr(hf, "num_experts_per_tok", 1))
            provider.moe_router_score_function = "sigmoid"
            provider.moe_router_enable_expert_bias = True
            provider.moe_router_dtype = "fp32"
            provider.moe_grouped_gemm = True
            provider.moe_ffn_hidden_size = int(getattr(hf, "moe_intermediate_size", None) or provider.ffn_hidden_size)

            # hybrid_override_pattern ('MEMEM*...') marks which layers are MoE.
            # Mirror to moe_layer_freq so miles' replay_utils registers the
            # rollout routing replay once per real MoE layer, not per transformer layer.
            pattern = getattr(provider, "hybrid_override_pattern", None) or getattr(
                hf, "hybrid_override_pattern", None
            )
            if pattern:
                provider.moe_layer_freq = [1 if ch == "E" else 0 for ch in pattern][: int(provider.num_layers)]

            shared_size = getattr(hf, "moe_shared_expert_intermediate_size", None)
            if shared_size is not None:
                provider.moe_shared_expert_intermediate_size = int(shared_size)
            elif getattr(hf, "n_shared_experts", 0):
                provider.moe_shared_expert_intermediate_size = provider.moe_ffn_hidden_size * int(hf.n_shared_experts)

            for hf_name, prov_name, cast in _NEMOTRONH_MOE_ROUTING_FIELDS:
                val = getattr(hf, hf_name, None)
                if val is not None:
                    setattr(provider, prov_name, cast(val))

            # MoE latent bottleneck (Super-120B-A12B). Upstream NemotronHBridge does
            # not surface this; without it, fc1/fc2 latent projections are not built
            # and routed experts get the wrong input dim (hidden_size vs moe_latent_size).
            # Megatron's moe_layer.preprocess() asserts the two are mutually exclusive,
            # so disable shared-expert overlap whenever a latent bottleneck is in use.
            latent_size = getattr(hf, "moe_latent_size", None)
            if latent_size is not None:
                provider.moe_latent_size = int(latent_size)
                provider.moe_shared_expert_overlap = False

            # MTP head: Super-120B's HF config has num_nextn_predict_layers=1, which
            # the base CONFIG_MAPPING translates to provider.mtp_num_layers=1. Miles'
            # generic training loop only feeds MTP labels when --enable-mtp-training is
            # set, so building the head adds dead weights. Disable unless explicitly
            # requested via the MILES_NEMOTRONH_KEEP_MTP env var.
            import os

            if not os.environ.get("MILES_NEMOTRONH_KEEP_MTP"):
                provider.mtp_num_layers = None
                provider.mtp_hybrid_override_pattern = None
            return provider

        def mapping_registry(self):
            # Append MoE mappings unconditionally. Dense variants do not carry
            # the extra megatron params so these mappings are simply unreferenced.
            base = _registry_mappings(super().mapping_registry())
            extras = [AutoMapping(megatron_param=m, hf_param=h) for m, h in _NEMOTRONH_MOE_MAPPINGS.items()]
            known_names = {mapping.megatron_param for mapping in base}
            mamba = [
                MambaConv1dMapping(
                    megatron_param=f"decoder.layers.*.mixer.conv1d_{suffix}",
                    hf_param=f"backbone.layers.*.mixer.conv1d.{suffix}",
                )
                for suffix in ("weight", "bias")
                if f"decoder.layers.*.mixer.conv1d_{suffix}" not in known_names
            ]
            return MegatronMappingRegistry(*base, *mamba, *extras)

    @MegatronModelBridge.register_bridge(source=NEMOTRON_H_OMNI_ARCHITECTURE, target=MambaModel)
    class MilesNemotronHLanguageOnlyBridge(MilesNemotronHBridge):
        """Language-model RL on a Nemotron-H VL checkpoint; the vision part stays frozen.

        Megatron builds the checkpoint's NemotronH language model only, with
        the same provider and conversions as :class:`MilesNemotronHBridge`; the
        HF names gain the ``language_model.`` prefix. The vision encoder and
        the projector are neither built nor mapped, so they hold no optimizer
        state, get no gradient, are not in the full-parameter manifests and are
        never part of a weight update: the rollout engine serves the VL model
        and keeps its own vision weights. The MTP block follows the text bridge
        (dropped unless ``MILES_NEMOTRONH_KEEP_MTP``); its HF names are under
        ``language_model.`` too.

        Weight updates must therefore reach an engine that serves the VL
        checkpoint (SGLang ``NemotronH_Omni_Reasoning_V3`` strips
        ``language_model.``), and the engine must never drop its weights
        (no colocated offload): nothing re-sends the vision part.
        """

        def provider_bridge(self, hf_pretrained):
            return super().provider_bridge(LanguageModelView(hf_pretrained))

        def mapping_registry(self):
            language = MilesNemotronHBridge()
            language.hf_config = language_model_config(self.hf_config)
            mappings = _registry_mappings(language.mapping_registry())
            return MegatronMappingRegistry(*(prefixed_mapping(m, LANGUAGE_MODEL_PREFIX) for m in mappings))

    return MilesNemotronHBridge


def _registry_mappings(registry) -> list:
    return list(registry.mappings if hasattr(registry, "mappings") else registry._mappings)


def _accept_vl_architecture() -> None:
    """Let ``AutoBridge`` accept the Nemotron-H VL architecture name.

    ``AutoBridge.supports`` / ``_validate_config`` accept only names that end in
    one of ``SUPPORTED_HF_ARCHITECTURES`` (read at call time);
    ``NemotronH_Omni_Reasoning_V3`` does not, although its bridge is registered.
    """
    from megatron.bridge.models.conversion import auto_bridge

    if not NEMOTRON_H_OMNI_ARCHITECTURE.endswith(auto_bridge.SUPPORTED_HF_ARCHITECTURES):
        auto_bridge.SUPPORTED_HF_ARCHITECTURES = (*auto_bridge.SUPPORTED_HF_ARCHITECTURES, NEMOTRON_H_OMNI_ARCHITECTURE)


def _install_nemotronh_hybrid_layer_shims() -> None:
    """Make ``TransformerLayer`` tolerate ``IdentityOp`` self_attention / mlp.

    nemotron_h is a Mamba+Attention hybrid: non-attention positions use
    ``IdentityOp`` for ``self_attention``; attention-only positions use
    ``IdentityOp`` for ``self.mlp``. Both ``TransformerLayer._forward_attention``
    and ``_forward_mlp`` do a 2-tuple unpack that raises ``ValueError`` on the
    raw tensor ``IdentityOp`` returns. Short-circuit both paths.
    """
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if getattr(TransformerLayer, "_miles_nemotron_hybrid_shim_installed", False):
        return

    _orig_fwd_attn = TransformerLayer._forward_attention
    _orig_fwd_mlp = TransformerLayer._forward_mlp

    def _forward_attention(self, *args, **kwargs):
        if type(self.self_attention).__name__ == "IdentityOp":
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            return hidden_states, None
        return _orig_fwd_attn(self, *args, **kwargs)

    def _forward_mlp(self, hidden_states, *args, **kwargs):
        if type(self.mlp).__name__ == "IdentityOp":
            return hidden_states
        return _orig_fwd_mlp(self, hidden_states, *args, **kwargs)

    TransformerLayer._forward_attention = _forward_attention
    TransformerLayer._forward_mlp = _forward_mlp
    TransformerLayer._miles_nemotron_hybrid_shim_installed = True


def _install_mamba_model_loss_mask_shim() -> None:
    """Adapt Miles MTP arguments while preserving the model loss mask.

    The Megatron HybridModel exposed as MambaModel accepts ``loss_mask`` but
    not ``mtp_labels``. Omit absent MTP labels so its forward remains usable
    when MTP is disabled; shift labels for models that support MTP.
    """
    from megatron.core.models.mamba import MambaModel

    if getattr(MambaModel, "_miles_loss_mask_shim_installed", False):
        return

    _orig_forward = MambaModel.forward

    def forward(self, *args, loss_mask=None, mtp_kwargs=None, **kwargs):
        # process_mtp_loss expects next-token-shifted labels; miles passes raw tokens.
        mtp_labels = (mtp_kwargs or {}).get("mtp_labels")
        if mtp_labels is not None:
            # HybridModel.forward does not accept even mtp_labels=None.
            kwargs["mtp_labels"] = torch.roll(mtp_labels, shifts=-1, dims=-1)
        return _orig_forward(self, *args, loss_mask=loss_mask, **kwargs)

    MambaModel.forward = forward
    MambaModel._miles_loss_mask_shim_installed = True


def install() -> None:
    """Register the Nemotron-H bridge subclasses and accept the VL architecture name."""
    for fn in (_build_bridge_subclass, _accept_vl_architecture):
        try:
            fn()
        except Exception as e:  # best-effort; avoid breaking unrelated models
            logger.warning("miles nemotron_h shim %s not applied: %s", fn.__name__, e)


install()
