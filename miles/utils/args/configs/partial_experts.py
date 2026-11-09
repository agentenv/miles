"""Partial routed-expert training: train a fixed subset of each MoE layer's experts, freeze the rest.

The selection file names, per MoE layer, the original (HF) expert ids that train and the *slot order*
the trainer loads the experts in. The trainer never rewrites the HF checkpoint; it runs every MoE layer in
slot space, where slot ``g`` holds original expert ``slot_order[g]``, and the router's weight rows and
expert bias are permuted with the same table at load, so the layer computes the same function (top-k over
a permuted score vector selects the permuted set, in the same order). The slot order puts each EP rank's
trainable experts at the front of its local slots, so one ``tokens_per_expert`` prefix splits the
dispatched tokens into a BF16 trainable half and a frozen MXFP4 half.

Everything that crosses the trainer boundary stays in original ids: HF export names, rollout routed
expert ids (R3), routing statistics and checkpoints. This module is pure Python so the orchestrator, the
checkpoint tooling and tests can validate a selection without torch or Megatron.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence

from pydantic import model_validator
from typing_extensions import Self

from miles.utils.args.schema import A, Arg, BaseConfig

SELECTION_SCHEMA = 1
TRAINABLE_DTYPES = ("bf16",)


class PartialExpertsConfig(BaseConfig):
    """Arguments for training only a selected subset of each MoE layer's routed experts."""

    partial_experts_selection: A[
        str | None,
        Arg(
            help=(
                "JSON expert selection (schema 1: per MoE layer the trainable original expert ids and the slot "
                "order the trainer loads the experts in). Trains exactly these routed experts in BF16 and "
                "freezes every other parameter, including the router; the frozen experts stay the HF "
                "checkpoint's MXFP4 bytes. Export, routing replay and checkpoints keep the original ids."
            )
        ),
    ] = None
    partial_experts_trainable_dtype: A[
        str,
        Arg(
            choices=list(TRAINABLE_DTYPES),
            help="Parameter dtype of the trainable experts; the frozen experts keep the checkpoint's MXFP4.",
        ),
    ] = "bf16"

    @model_validator(mode="after")
    def _validate_partial_experts(self) -> Self:
        assert self.partial_experts_trainable_dtype in TRAINABLE_DTYPES, (
            f"--partial-experts-trainable-dtype {self.partial_experts_trainable_dtype!r}: "
            f"supported {list(TRAINABLE_DTYPES)}"
        )
        return self


class ExpertSelectionError(ValueError):
    """The selection file violates the schema or the EP layout constraints."""


@dataclasses.dataclass(frozen=True)
class LayerSelection:
    """One MoE layer: the trainable original ids (ascending) and the slot order (a permutation)."""

    trainable: tuple[int, ...]
    slot_order: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ExpertSelection:
    """A validated ``expert_selection.json``; ``layers`` keys are HF layer indices (0-based)."""

    model: str
    num_experts: int
    num_trainable: int
    ep_size: int
    layers: Mapping[int, LayerSelection]
    source: Mapping[str, object] = dataclasses.field(default_factory=dict)
    sha256: str | None = None

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def num_local_trainable(self) -> int:
        return self.num_trainable // self.ep_size

    @property
    def moe_layers(self) -> list[int]:
        return sorted(self.layers)

    def layers_sha256(self) -> str:
        """sha256 of the canonical JSON of ``layers``: the identity the release manifest carries."""
        return canonical_layers_sha256(
            {str(k): {"trainable": list(v.trainable), "slot_order": list(v.slot_order)} for k, v in self.layers.items()}
        )

    def to_json(self) -> dict:
        return {
            "schema": SELECTION_SCHEMA,
            "model": self.model,
            "num_experts": self.num_experts,
            "num_trainable": self.num_trainable,
            "ep_size": self.ep_size,
            "layers": {
                str(k): {"trainable": list(v.trainable), "slot_order": list(v.slot_order)}
                for k, v in sorted(self.layers.items())
            },
            "source": dict(self.source),
            "sha256": self.layers_sha256(),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, object]) -> ExpertSelection:
        _require(isinstance(obj, Mapping), "selection must be a JSON object")
        _require(obj.get("schema") == SELECTION_SCHEMA, f"schema {obj.get('schema')!r}, expected {SELECTION_SCHEMA}")
        for key in ("num_experts", "num_trainable", "ep_size", "layers"):
            _require(key in obj, f"missing key {key!r}")
        layers_obj = obj["layers"]
        _require(isinstance(layers_obj, Mapping) and layers_obj, "layers must be a non-empty object")
        layers = {}
        for key, layer in layers_obj.items():
            _require(str(key).isdigit(), f"layer key {key!r} is not an HF layer index")
            _require(isinstance(layer, Mapping), f"layer {key}: must be an object")
            _require("trainable" in layer and "slot_order" in layer, f"layer {key}: needs trainable and slot_order")
            layers[int(key)] = LayerSelection(
                trainable=tuple(int(x) for x in layer["trainable"]),
                slot_order=tuple(int(x) for x in layer["slot_order"]),
            )
        selection = cls(
            model=str(obj.get("model", "")),
            num_experts=int(obj["num_experts"]),
            num_trainable=int(obj["num_trainable"]),
            ep_size=int(obj["ep_size"]),
            layers=layers,
            source=dict(obj.get("source", {}) or {}),
            sha256=obj.get("sha256"),
        )
        selection.validate()
        return selection

    def validate(self) -> None:
        n, k, ep = self.num_experts, self.num_trainable, self.ep_size
        _require(n > 0 and k > 0 and ep > 0, f"num_experts={n}, num_trainable={k}, ep_size={ep} must be positive")
        _require(k < n, f"num_trainable {k} must be below num_experts {n}")
        _require(n % ep == 0, f"ep_size {ep} does not divide num_experts {n}")
        _require(k % ep == 0, f"ep_size {ep} does not divide num_trainable {k}: unequal trainable slots per rank")
        n_local, k_local = n // ep, k // ep
        full = list(range(n))
        for layer, sel in sorted(self.layers.items()):
            where = f"layer {layer}"
            _require(len(sel.trainable) == k, f"{where}: {len(sel.trainable)} trainable ids, expected {k}")
            _require(list(sel.trainable) == sorted(set(sel.trainable)), f"{where}: trainable ids must be ascending and unique")
            _require(all(0 <= e < n for e in sel.trainable), f"{where}: trainable id out of range")
            _require(sorted(sel.slot_order) == full, f"{where}: slot_order is not a permutation of range({n})")
            trainable = set(sel.trainable)
            for rank in range(ep):
                local = sel.slot_order[rank * n_local : (rank + 1) * n_local]
                flags = [e in trainable for e in local]
                _require(
                    flags == [True] * k_local + [False] * (n_local - k_local),
                    f"{where}: EP rank {rank} must hold exactly its first {k_local} local slots trainable, "
                    f"got {sum(flags)} trainable slots at positions {[i for i, f in enumerate(flags) if f]}",
                )
        if self.sha256 is not None:
            actual = self.layers_sha256()
            _require(self.sha256 == actual, f"sha256 {self.sha256} does not match layers ({actual})")


def canonical_layers_sha256(layers: Mapping[str, Mapping[str, Sequence[int]]]) -> str:
    canonical = {
        str(k): {"trainable": [int(x) for x in v["trainable"]], "slot_order": [int(x) for x in v["slot_order"]]}
        for k, v in layers.items()
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_expert_selection(path: str) -> ExpertSelection:
    with open(path) as f:
        obj = json.load(f)
    try:
        return ExpertSelection.from_json(obj)
    except ExpertSelectionError as e:
        raise ExpertSelectionError(f"{path}: {e}") from None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpertSelectionError(message)
