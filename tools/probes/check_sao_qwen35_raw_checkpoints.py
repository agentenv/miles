"""Fail closed when the SAO smoke checkpoints do not match raw Qwen3.5.

This is intentionally a metadata-only check: it validates every native actor
tensor against the direct Megatron-to-HF exporter without reading the multi-GB
tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import (
    convert_qwen3_5_to_hf,
)


EXPECTED_ACTOR_TENSORS = 378
EXPECTED_ACTOR_SCALARS = 4_205_751_296
EXPECTED_HF_TEXT_TENSORS = 426
VALUE_HEAD_KEYS = {"output_layer.weight", "output_layer.bias"}


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


def verify(actor_root: Path, critic_root: Path, hf_root: Path) -> dict[str, int]:
    actor = _tensor_metadata(actor_root)
    critic = _tensor_metadata(critic_root)

    if len(actor) != EXPECTED_ACTOR_TENSORS:
        raise RuntimeError(f"actor tensor count mismatch: {len(actor)}")
    actor_scalars = sum(value.size.numel() for value in actor.values())
    if actor_scalars != EXPECTED_ACTOR_SCALARS:
        raise RuntimeError(f"actor scalar count mismatch: {actor_scalars}")
    if set(critic) != set(actor) | VALUE_HEAD_KEYS:
        raise RuntimeError("critic backbone does not exactly match the actor checkpoint")
    if tuple(critic["output_layer.weight"].size) != (51, 2560):
        raise RuntimeError("critic value-head weight is not 51x2560")
    if tuple(critic["output_layer.bias"].size) != (51,):
        raise RuntimeError("critic value-head bias is not 51")

    converter_args = argparse.Namespace(
        num_attention_heads=16,
        num_query_groups=4,
        kv_channels=256,
        hidden_size=2560,
    )
    mapped: list[str] = []
    for name, metadata in actor.items():
        parameter = torch.empty(
            tuple(metadata.size),
            dtype=metadata.properties.dtype,
            device="meta",
        )
        mapped.extend(
            hf_name
            for hf_name, _ in convert_qwen3_5_to_hf(
                converter_args,
                f"module.module.{name}",
                parameter,
            )
        )
    if len(mapped) != len(set(mapped)):
        raise RuntimeError("Qwen3.5 direct exporter produced duplicate HF names")

    config = json.loads((hf_root / "config.json").read_text())
    if config.get("tie_word_embeddings") is not True:
        raise RuntimeError("the rollout checkpoint does not declare tied embeddings")
    hf_names = set(
        json.loads((hf_root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    )
    expected_text_names = {
        name for name in hf_names if name.startswith("model.language_model.")
    }
    if "lm_head.weight" in hf_names or "lm_head.weight" in mapped:
        raise RuntimeError("tied Qwen3.5 unexpectedly has a separate LM head")
    if set(mapped) != expected_text_names:
        missing = sorted(expected_text_names - set(mapped))
        extra = sorted(set(mapped) - expected_text_names)
        raise RuntimeError(
            f"Qwen3.5 direct-export coverage mismatch: missing={missing[:5]!r} "
            f"extra={extra[:5]!r}"
        )
    if len(mapped) != EXPECTED_HF_TEXT_TENSORS:
        raise RuntimeError(f"HF text tensor count mismatch: {len(mapped)}")

    a_logs = [
        value
        for name, value in actor.items()
        if name.endswith("linear_attn.A_log")
    ]
    if not a_logs or any(value.properties.dtype != torch.float32 for value in a_logs):
        raise RuntimeError("Qwen3.5 GDN A_log tensors are not all FP32")

    return {
        "actor_tensors": len(actor),
        "actor_scalars": actor_scalars,
        "critic_tensors": len(critic),
        "hf_text_tensors": len(mapped),
        "fp32_a_log_tensors": len(a_logs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    args = parser.parse_args()
    print(
        "SAO_RAW_CHECKPOINT_PREFLIGHT_OK",
        json.dumps(
            verify(args.actor, args.critic, args.hf),
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
