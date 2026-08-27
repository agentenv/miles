"""Fail closed when the SAO smoke checkpoints do not match raw Qwen3.5.

This is intentionally a metadata-only check: it validates every native actor
tensor against the direct Megatron-to-HF exporter without reading the multi-GB
tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path

import torch

from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import (
    convert_qwen3_5_to_hf,
)


VALUE_HEAD_KEYS = {"output_layer.weight", "output_layer.bias"}
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


def _iteration_dir(root: Path) -> Path:
    marker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
    if marker == "release":
        return root / "release"
    return root / f"iter_{int(marker):07d}"


def _tensor_metadata(root: Path) -> dict[str, object]:
    # Distributed-checkpoint metadata is a trusted, locally generated pickle.
    with (_iteration_dir(root) / ".metadata").open("rb") as source:
        metadata = pickle.load(source)  # noqa: S301
    return {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if hasattr(value, "size")
        and "/" not in key
        and not key.startswith(("optimizer.", "rng_state", "common_state"))
    }


def _hf_tensor_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor shapes from local safetensors headers without loading weights."""

    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid safetensors index: {index_path}")

    shapes: dict[str, tuple[int, ...]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = root / shard_name
        with shard_path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise RuntimeError(f"truncated safetensors header: {shard_path}")
            (header_size,) = struct.unpack("<Q", prefix)
            if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise RuntimeError(
                    f"invalid safetensors header size {header_size} in {shard_path}"
                )
            header_bytes = source.read(header_size)
            if len(header_bytes) != header_size:
                raise RuntimeError(f"truncated safetensors header: {shard_path}")
        header = json.loads(header_bytes)
        present = {name for name in header if name != "__metadata__"}
        assigned = {name for name, mapped_shard in weight_map.items() if mapped_shard == shard_name}
        if present != assigned:
            missing = sorted(assigned - present)
            extra = sorted(present - assigned)
            raise RuntimeError(
                f"safetensors index/header mismatch for {shard_name}: "
                f"missing={missing[:5]!r} extra={extra[:5]!r}"
            )
        for name in assigned:
            shape = header[name].get("shape")
            if not isinstance(shape, list) or not all(
                isinstance(size, int) and size >= 0 for size in shape
            ):
                raise RuntimeError(f"invalid tensor shape for {name!r} in {shard_path}")
            shapes[name] = tuple(shape)
    return shapes


def _metadata_shape(metadata: object) -> tuple[int, ...]:
    return tuple(metadata.size)


def _metadata_dtype(metadata: object) -> torch.dtype:
    return metadata.properties.dtype


def _require_same_backbone_metadata(
    actor: dict[str, object], critic: dict[str, object]
) -> None:
    mismatched = [
        name
        for name, actor_metadata in actor.items()
        if _metadata_shape(critic[name]) != _metadata_shape(actor_metadata)
        or _metadata_dtype(critic[name]) != _metadata_dtype(actor_metadata)
    ]
    if mismatched:
        raise RuntimeError(
            "critic backbone metadata does not exactly match the actor checkpoint: "
            f"{mismatched[:5]!r}"
        )


def _qwen35_text_config(hf_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    config = json.loads((hf_root / "config.json").read_text())
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise RuntimeError("Qwen3.5 config has no valid text_config")
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError(f"expected Qwen3.5 config, got {config.get('model_type')!r}")
    if text_config.get("attn_output_gate") is not True:
        raise RuntimeError("Qwen3.5 config does not enable the attention-output gate")
    tied = config.get("tie_word_embeddings", text_config.get("tie_word_embeddings"))
    if tied is not True:
        raise RuntimeError(
            "SAO raw-checkpoint preflight currently requires tied Qwen3.5 embeddings"
        )
    return config, text_config


def verify(
    actor_root: Path,
    critic_root: Path,
    hf_root: Path,
    *,
    value_num_bins: int = 51,
) -> dict[str, int]:
    actor = _tensor_metadata(actor_root)
    critic = _tensor_metadata(critic_root)
    _, text_config = _qwen35_text_config(hf_root)

    actor_scalars = sum(value.size.numel() for value in actor.values())
    if set(critic) != set(actor) | VALUE_HEAD_KEYS:
        raise RuntimeError("critic backbone does not exactly match the actor checkpoint")
    _require_same_backbone_metadata(actor, critic)

    hidden_size = int(text_config["hidden_size"])
    if value_num_bins <= 0:
        raise RuntimeError("value_num_bins must be positive")
    if _metadata_shape(critic["output_layer.weight"]) != (
        value_num_bins,
        hidden_size,
    ):
        raise RuntimeError(
            "critic value-head weight is not "
            f"{value_num_bins}x{hidden_size}"
        )
    if _metadata_shape(critic["output_layer.bias"]) != (value_num_bins,):
        raise RuntimeError(f"critic value-head bias is not {value_num_bins}")

    converter_args = argparse.Namespace(
        num_attention_heads=int(text_config["num_attention_heads"]),
        num_query_groups=int(text_config["num_key_value_heads"]),
        kv_channels=int(
            text_config.get(
                "head_dim",
                hidden_size // int(text_config["num_attention_heads"]),
            )
        ),
        hidden_size=hidden_size,
    )
    mapped: dict[str, tuple[int, ...]] = {}
    for name, metadata in actor.items():
        parameter = torch.empty(
            _metadata_shape(metadata),
            dtype=_metadata_dtype(metadata),
            device="meta",
        )
        for hf_name, hf_parameter in convert_qwen3_5_to_hf(
            converter_args,
            f"module.module.{name}",
            parameter,
        ):
            if hf_name in mapped:
                raise RuntimeError(
                    f"Qwen3.5 direct exporter produced duplicate HF name {hf_name!r}"
                )
            mapped[hf_name] = tuple(hf_parameter.shape)

    hf_shapes = _hf_tensor_shapes(hf_root)
    expected_text_shapes = {
        name: shape
        for name, shape in hf_shapes.items()
        if name.startswith("model.language_model.")
    }
    if "lm_head.weight" in hf_shapes or "lm_head.weight" in mapped:
        raise RuntimeError("tied Qwen3.5 unexpectedly has a separate LM head")
    if set(mapped) != set(expected_text_shapes):
        missing = sorted(set(expected_text_shapes) - set(mapped))
        extra = sorted(set(mapped) - set(expected_text_shapes))
        raise RuntimeError(
            f"Qwen3.5 direct-export coverage mismatch: missing={missing[:5]!r} "
            f"extra={extra[:5]!r}"
        )
    shape_mismatches = [
        name for name, shape in mapped.items() if shape != expected_text_shapes[name]
    ]
    if shape_mismatches:
        name = shape_mismatches[0]
        raise RuntimeError(
            f"Qwen3.5 direct-export shape mismatch for {name!r}: "
            f"actor={mapped[name]!r} hf={expected_text_shapes[name]!r}"
        )
    hf_text_scalars = sum(
        torch.Size(shape).numel() for shape in expected_text_shapes.values()
    )
    if actor_scalars != hf_text_scalars:
        raise RuntimeError(
            f"actor scalar count mismatch: actor={actor_scalars} hf={hf_text_scalars}"
        )

    a_logs = [
        value
        for name, value in actor.items()
        if name.endswith("linear_attn.A_log")
    ]
    if not a_logs or any(_metadata_dtype(value) != torch.float32 for value in a_logs):
        raise RuntimeError("Qwen3.5 GDN A_log tensors are not all FP32")

    return {
        "actor_tensors": len(actor),
        "actor_scalars": actor_scalars,
        "critic_tensors": len(critic),
        "hf_text_tensors": len(mapped),
        "fp32_a_log_tensors": len(a_logs),
        "hidden_size": hidden_size,
        "num_layers": int(text_config["num_hidden_layers"]),
        "value_num_bins": value_num_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--value-num-bins", type=int, default=51)
    args = parser.parse_args()
    print(
        "SAO_RAW_CHECKPOINT_PREFLIGHT_OK",
        json.dumps(
            verify(
                args.actor,
                args.critic,
                args.hf,
                value_num_bins=args.value_num_bins,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
