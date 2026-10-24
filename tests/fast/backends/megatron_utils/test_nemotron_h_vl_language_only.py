"""The Nemotron-H VL checkpoint trains its language model only (vision frozen).

The bridge plugin is loaded against stub ``megatron.bridge`` modules (as in
``test_nemotron_h_conv_mapping.py``); the real bridge was checked on CPU
against the Nemotron-3.5 Super VL 120B weight index (yeto S27 NEMOTRON-120B-PREP).
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Mapping:
    def __init__(self, megatron_param, hf_param) -> None:
        self.megatron_param = megatron_param
        self.hf_param = hf_param


class QKVMapping(Mapping):
    pass


class MambaMapping(Mapping):
    pass


TEXT_MAPPINGS = [
    Mapping("embedding.word_embeddings.weight", "backbone.embeddings.weight"),
    Mapping("output_layer.weight", "lm_head.weight"),
    Mapping("decoder.layers.*.mlp.experts.linear_fc1.weight*", "backbone.layers.*.mixer.experts.*.up_proj.weight"),
    QKVMapping(
        "decoder.layers.*.self_attention.linear_qkv.weight",
        {
            "q": "backbone.layers.*.mixer.q_proj.weight",
            "k": "backbone.layers.*.mixer.k_proj.weight",
            "v": "backbone.layers.*.mixer.v_proj.weight",
        },
    ),
    Mapping("mtp.layers.*.eh_proj.weight", "mtp.layers.*.eh_proj.weight"),
]


def load_plugin(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    class Registry:
        def __init__(self, *mappings) -> None:
            self.mappings = list(mappings)

    class NemotronHBridge:
        hf_config = None

        def provider_bridge(self, hf_pretrained):
            seen["provider_config"] = hf_pretrained.config
            seen["provider_state"] = hf_pretrained.state
            return SimpleNamespace(mtp_num_layers=None)

        def mapping_registry(self):
            seen.setdefault("mapping_configs", []).append(self.hf_config)
            return Registry(*TEXT_MAPPINGS)

    registered = {}

    def register_bridge(**kwargs):
        def decorator(cls):
            registered[kwargs["source"]] = cls
            return cls

        return decorator

    auto_bridge = SimpleNamespace(SUPPORTED_HF_ARCHITECTURES=("ForCausalLM", "NemotronH_Nano_Omni_Reasoning_V3"))
    modules = {
        "megatron.bridge.models.conversion.mapping_registry": {"MegatronMappingRegistry": Registry},
        "megatron.bridge.models.conversion.model_bridge": {
            "MegatronModelBridge": SimpleNamespace(register_bridge=register_bridge)
        },
        "megatron.bridge.models.conversion.param_mapping": {"AutoMapping": Mapping, "MambaConv1dMapping": MambaMapping},
        "megatron.bridge.models.nemotronh.nemotron_h_bridge": {"NemotronHBridge": NemotronHBridge},
        "megatron.bridge.models.conversion": {"auto_bridge": auto_bridge},
        # provider_bridge installs the hybrid-layer and loss-mask shims on these classes
        "megatron.core.models.mamba": {"MambaModel": type("MambaModel", (), {"forward": lambda self, **kw: kw})},
        "megatron.core.transformer.transformer_layer": {
            "TransformerLayer": type(
                "TransformerLayer", (), {"_forward_attention": lambda self: None, "_forward_mlp": lambda self, h: h}
            )
        },
    }
    for name, attributes in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_vl_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # install() runs on import
    return module, registered, seen, auto_bridge


def test_vl_architecture_is_registered_and_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    module, registered, _seen, auto_bridge = load_plugin(monkeypatch)

    assert set(registered) == {"NemotronHForCausalLM", "NemotronH_Omni_Reasoning_V3"}
    assert registered["NemotronH_Omni_Reasoning_V3"].__name__ == "MilesNemotronHLanguageOnlyBridge"
    assert issubclass(registered["NemotronH_Omni_Reasoning_V3"], registered["NemotronHForCausalLM"])
    assert "NemotronH_Omni_Reasoning_V3".endswith(auto_bridge.SUPPORTED_HF_ARCHITECTURES)
    module._accept_vl_architecture()  # idempotent
    assert auto_bridge.SUPPORTED_HF_ARCHITECTURES.count("NemotronH_Omni_Reasoning_V3") == 1


def test_vl_mappings_are_the_text_mappings_under_language_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _module, registered, seen, _auto = load_plugin(monkeypatch)
    llm_config = SimpleNamespace(name="llm")
    bridge = registered["NemotronH_Omni_Reasoning_V3"]()
    bridge.hf_config = SimpleNamespace(llm_config=llm_config)

    mappings = bridge.mapping_registry().mappings

    # The text bridge is asked for its mappings with the language-model config (MTP pattern lives there).
    assert seen["mapping_configs"][-1] is llm_config
    text = {m.megatron_param: m for m in TEXT_MAPPINGS}
    by_name = {m.megatron_param: m for m in mappings}
    assert set(text) <= set(by_name)
    for name, original in text.items():
        mapped = by_name[name]
        assert type(mapped) is type(original) and mapped is not original
        if isinstance(original.hf_param, str):
            assert mapped.hf_param == "language_model." + original.hf_param
        else:
            assert mapped.hf_param == {k: "language_model." + v for k, v in original.hf_param.items()}
    # The originals are untouched, and nothing maps the vision encoder or the projector.
    assert TEXT_MAPPINGS[0].hf_param == "backbone.embeddings.weight"
    hf_names = [h for m in mappings for h in ([m.hf_param] if isinstance(m.hf_param, str) else m.hf_param.values())]
    assert all(h.startswith("language_model.") for h in hf_names)
    assert not any(h.startswith(("vision_model.", "mlp1.", "vision_projector.", "sound")) for h in hf_names)


def test_vl_provider_is_built_from_the_language_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _module, registered, seen, _auto = load_plugin(monkeypatch)
    monkeypatch.delenv("MILES_NEMOTRONH_KEEP_MTP", raising=False)
    llm_config = SimpleNamespace(n_routed_experts=0)
    state = object()
    hf_pretrained = SimpleNamespace(config=SimpleNamespace(llm_config=llm_config), state=state)

    provider = registered["NemotronH_Omni_Reasoning_V3"]().provider_bridge(hf_pretrained)

    assert seen["provider_config"] is llm_config
    assert seen["provider_state"] is state  # weights are read from the VL checkpoint itself
    assert provider.mtp_num_layers is None


def test_vl_config_without_language_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    module, registered, _seen, _auto = load_plugin(monkeypatch)
    with pytest.raises(ValueError, match="llm_config"):
        registered["NemotronH_Omni_Reasoning_V3"]().provider_bridge(SimpleNamespace(config=SimpleNamespace()))
    with pytest.raises(ValueError, match="llm_config"):
        module.language_model_config(SimpleNamespace(llm_config=None))
