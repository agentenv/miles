from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the exact production exporter without importing the aggregate
# megatron_to_hf package, whose unrelated backends require Megatron at import
# time even though this metadata-only test does not.
_QWEN35_EXPORTER_MODULE = (
    "miles.backends.megatron_utils.megatron_to_hf.qwen3_5"
)
_load_module(
    _QWEN35_EXPORTER_MODULE,
    _REPO_ROOT
    / "miles"
    / "backends"
    / "megatron_utils"
    / "megatron_to_hf"
    / "qwen3_5.py",
)
preflight = _load_module(
    "test_qwen35_sao_checkpoint_preflight_module",
    _REPO_ROOT / "tools" / "probes" / "check_sao_qwen35_raw_checkpoints.py",
)


def _metadata(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16):
    return SimpleNamespace(
        size=torch.Size(shape),
        properties=SimpleNamespace(dtype=dtype),
    )


def _qwen35_08_actor_metadata() -> dict[str, object]:
    # A minimal native-Megatron subset that exercises tied embeddings, the
    # gated full-attention QKV split, and the GatedDeltaNet FP32 contract.
    return {
        "embedding.word_embeddings.weight": _metadata((248320, 1024)),
        "decoder.final_layernorm.weight": _metadata((1024,)),
        "decoder.layers.0.self_attention.linear_qkv.weight": _metadata(
            (5120, 1024)
        ),
        "decoder.layers.0.self_attention.linear_attn.A_log": _metadata(
            (16,), torch.float32
        ),
    }


def _qwen35_08_hf_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "model.language_model.embed_tokens.weight": (248320, 1024),
        "model.language_model.norm.weight": (1024,),
        "model.language_model.layers.0.self_attn.q_proj.weight": (4096, 1024),
        "model.language_model.layers.0.self_attn.k_proj.weight": (512, 1024),
        "model.language_model.layers.0.self_attn.v_proj.weight": (512, 1024),
        "model.language_model.layers.0.linear_attn.A_log": (16,),
    }


def _write_qwen35_08_config(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "attn_output_gate": True,
                    "head_dim": 256,
                    "hidden_size": 1024,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 24,
                    "num_key_value_heads": 2,
                    "tie_word_embeddings": True,
                },
            }
        )
    )


def test_qwen35_08_model_profile_matches_hf_text_config():
    script = (
        _REPO_ROOT / "scripts" / "models" / "qwen3.5-0.8B.sh"
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "${MODEL_ARGS[@]}"',
            "qwen35-profile-test",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "--spec",
        "miles_plugins.models.qwen3_5",
        "get_qwen3_5_spec",
        "--disable-bias-linear",
        "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads",
        "8",
        "--num-query-groups",
        "2",
        "--kv-channels",
        "256",
        "--num-layers",
        "24",
        "--hidden-size",
        "1024",
        "--ffn-hidden-size",
        "3584",
        "--normalization",
        "RMSNorm",
        "--apply-layernorm-1p",
        "--position-embedding-type",
        "rope",
        "--norm-epsilon",
        "1e-6",
        "--rotary-percent",
        "0.25",
        "--swiglu",
        "--vocab-size",
        "248320",
        "--rotary-base",
        "10000000",
        "--attention-output-gate",
    ]


def test_qwen35_08_sao_preflight_derives_layout_from_hf_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_qwen35_08_config(tmp_path)
    actor = _qwen35_08_actor_metadata()
    critic = {
        **actor,
        "output_layer.weight": _metadata((51, 1024)),
        "output_layer.bias": _metadata((51,)),
    }

    monkeypatch.setattr(
        preflight,
        "_tensor_metadata",
        lambda root: actor if root.name == "actor" else critic,
    )
    monkeypatch.setattr(
        preflight,
        "_hf_tensor_shapes",
        lambda root: _qwen35_08_hf_shapes(),
    )

    report = preflight.verify(
        tmp_path / "actor",
        tmp_path / "critic",
        tmp_path,
    )

    assert report == {
        "actor_tensors": 4,
        "actor_scalars": 259_523_600,
        "critic_tensors": 6,
        "hf_text_tensors": 6,
        "fp32_a_log_tensors": 1,
        "hidden_size": 1024,
        "num_layers": 24,
        "value_num_bins": 51,
    }


def test_qwen35_sao_preflight_rejects_export_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_qwen35_08_config(tmp_path)
    actor = _qwen35_08_actor_metadata()
    critic = {
        **actor,
        "output_layer.weight": _metadata((51, 1024)),
        "output_layer.bias": _metadata((51,)),
    }
    hf_shapes = _qwen35_08_hf_shapes()
    hf_shapes["model.language_model.layers.0.self_attn.q_proj.weight"] = (
        2048,
        1024,
    )

    monkeypatch.setattr(
        preflight,
        "_tensor_metadata",
        lambda root: actor if root.name == "actor" else critic,
    )
    monkeypatch.setattr(preflight, "_hf_tensor_shapes", lambda root: hf_shapes)

    with pytest.raises(RuntimeError, match="direct-export shape mismatch"):
        preflight.verify(tmp_path / "actor", tmp_path / "critic", tmp_path)


def test_hf_tensor_shapes_reads_only_safetensors_metadata(tmp_path: Path):
    tensor_name = "model.language_model.norm.weight"
    shard_name = "model.safetensors"
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            tensor_name: {
                "dtype": "BF16",
                "shape": [1024],
                "data_offsets": [0, 2048],
            },
        },
        separators=(",", ":"),
    ).encode()
    (tmp_path / shard_name).write_bytes(struct.pack("<Q", len(header)) + header)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}})
    )

    assert preflight._hf_tensor_shapes(tmp_path) == {tensor_name: (1024,)}
