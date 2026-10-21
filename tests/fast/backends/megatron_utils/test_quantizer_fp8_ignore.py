from argparse import Namespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8
from miles.utils.fp8_ignore import ignored_modules, is_ignored
from miles.utils.rollout_checkpoint import rollout_hf_checkpoint

ATTN = [f"model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]


def _config(**extra):
    return {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", **extra}


def test_ignored_modules_follows_sglang_key_order_and_prefix_forms() -> None:
    assert ignored_modules(_config(modules_to_not_convert=["lm_head"])) == ("lm_head", "model.lm_head")
    # ignored_layers wins over modules_to_not_convert, as SGLang's get_from_keys_or
    both = _config(ignored_layers=["model.layers.0.self_attn"], modules_to_not_convert=["lm_head"])
    assert ignored_modules(both) == ("layers.0.self_attn", "model.layers.0.self_attn")
    assert ignored_modules(_config()) == ()


def test_is_ignored_matches_on_module_path_boundaries() -> None:
    ignored = ignored_modules(_config(ignored_layers=["mlp.gate", "model.layers.0.self_attn"]))
    assert is_ignored("model.layers.0.self_attn.q_proj.weight", ignored)
    assert is_ignored("model.layers.3.mlp.gate.weight", ignored)
    assert not is_ignored("model.layers.3.mlp.gate_proj.weight", ignored)
    assert not is_ignored("model.layers.1.self_attn.q_proj.weight", ignored)


def _qkv(n=4):
    return [(f"{name}.weight", torch.randn(256, 256, dtype=torch.bfloat16)) for name in ATTN[:3]][:n]


def test_attention_in_the_ignore_list_stays_bf16() -> None:
    args = Namespace()
    params = _qkv()
    out = quantize_params_fp8(args, "module.module.decoder.layers.0.self_attention.linear_qkv.weight", params,
                              _config(ignored_layers=ATTN))
    assert [(n, t.dtype) for n, t in out] == [(n, torch.bfloat16) for n, _ in params]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(out, params))


def test_ffn_outside_the_ignore_list_is_quantized() -> None:
    fc1 = [("model.layers.0.mlp.gate_proj.weight", torch.randn(256, 256, dtype=torch.bfloat16)),
           ("model.layers.0.mlp.up_proj.weight", torch.randn(256, 256, dtype=torch.bfloat16))]
    out = dict(quantize_params_fp8(Namespace(), "module.module.decoder.layers.0.mlp.linear_fc1.weight", fc1,
                                   _config(ignored_layers=ATTN)))
    assert out["model.layers.0.mlp.gate_proj.weight"].dtype == torch.float8_e4m3fn
    assert set(out) == {"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight_scale",
                        "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.weight_scale"}


def test_without_an_ignore_list_attention_is_quantized_as_before() -> None:
    out = dict(quantize_params_fp8(Namespace(), "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
                                   _qkv(), _config()))
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn


def test_rollout_checkpoint_defaults_to_the_trainer_checkpoint() -> None:
    assert rollout_hf_checkpoint(Namespace(hf_checkpoint="/bf16")) == "/bf16"
    assert rollout_hf_checkpoint(Namespace(hf_checkpoint="/bf16", rollout_hf_checkpoint=None)) == "/bf16"
    assert rollout_hf_checkpoint(Namespace(hf_checkpoint="/bf16", rollout_hf_checkpoint="/fp8")) == "/fp8"
