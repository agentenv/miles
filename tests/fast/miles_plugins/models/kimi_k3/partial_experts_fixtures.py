"""Small synthetic expert selections for the partial-experts CPU tests."""

from __future__ import annotations

import json
import random

from miles.utils.args.configs.partial_experts import ExpertSelection, canonical_layers_sha256


def make_selection_json(
    *, num_experts: int, num_trainable: int, ep_size: int, layers: list[int], seed: int = 0, model: str = "kimi_k31"
) -> dict:
    """A schema-1 selection: per layer a random trainable set and a slot order that front-loads each EP rank."""
    rng = random.Random(seed)
    n_local, k_local = num_experts // ep_size, num_trainable // ep_size
    out_layers = {}
    for layer in layers:
        trainable = sorted(rng.sample(range(num_experts), num_trainable))
        frozen = [e for e in range(num_experts) if e not in set(trainable)]
        rng.shuffle(frozen)
        hot = list(trainable)
        rng.shuffle(hot)
        slot_order = []
        for rank in range(ep_size):
            slot_order += hot[rank * k_local : (rank + 1) * k_local]
            slot_order += sorted(frozen[rank * (n_local - k_local) : (rank + 1) * (n_local - k_local)])
        out_layers[str(layer)] = {"trainable": trainable, "slot_order": slot_order}
    return {
        "schema": 1,
        "model": model,
        "num_experts": num_experts,
        "num_trainable": num_trainable,
        "ep_size": ep_size,
        "layers": out_layers,
        "source": {"method": "routing_frequency", "prompts_sha256": "0" * 64, "tokens": 0},
        "sha256": canonical_layers_sha256(out_layers),
    }


def make_selection(**kwargs) -> ExpertSelection:
    return ExpertSelection.from_json(make_selection_json(**kwargs))


def write_selection(tmp_path, obj: dict, name: str = "expert_selection.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)
