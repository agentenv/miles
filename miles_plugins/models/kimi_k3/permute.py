"""Slot permutation tables for partial routed-expert training (torch only, no Megatron).

A selection (``miles.utils.args.configs.partial_experts.ExpertSelection``) fixes, per MoE layer, the slot
order ``perm``: global slot ``g`` holds original HF expert ``perm[g]``. EP rank ``r`` owns global slots
``[r * n_local, (r + 1) * n_local)`` and its first ``k_local`` slots are trainable, so the trainable half is
local slots ``[0, k_local)`` on every rank. The inverse ``inv[original_id]`` is the global slot.

Original ids are what inference engines, HF checkpoints and routing statistics speak; slot ids exist only
inside the trainer. Convert at the boundary with the methods below.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import torch

from miles.utils.args.configs.partial_experts import ExpertSelection, load_expert_selection


@dataclasses.dataclass(frozen=True)
class SlotLayout:
    """Per-layer slot permutation (``perm``: slot -> original id) and its inverse, for one EP size."""

    selection_sha256: str
    ep_size: int
    num_experts: int
    num_trainable: int
    perm: Mapping[int, torch.Tensor]  # HF layer -> int64[num_experts]
    inv: Mapping[int, torch.Tensor]  # HF layer -> int64[num_experts]

    @classmethod
    def from_selection(cls, selection: ExpertSelection) -> SlotLayout:
        perm = {
            layer: torch.tensor(sel.slot_order, dtype=torch.long) for layer, sel in sorted(selection.layers.items())
        }
        return cls(
            selection_sha256=selection.layers_sha256(),
            ep_size=selection.ep_size,
            num_experts=selection.num_experts,
            num_trainable=selection.num_trainable,
            perm=perm,
            inv={layer: torch.argsort(p) for layer, p in perm.items()},
        )

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def num_local_trainable(self) -> int:
        return self.num_trainable // self.ep_size

    @property
    def moe_layers(self) -> list[int]:
        return sorted(self.perm)

    # ---- slot <-> original id -------------------------------------------------------------------

    def is_trainable_slot(self, global_slot: int) -> bool:
        return global_slot % self.num_local_experts < self.num_local_trainable

    def trainable_index(self, global_slot: int) -> int:
        """EP-independent index ``t`` (0..num_trainable-1) of a trainable slot: the checkpoint shard coordinate."""
        rank, local = divmod(global_slot, self.num_local_experts)
        assert local < self.num_local_trainable, f"slot {global_slot} is frozen"
        return rank * self.num_local_trainable + local

    def original_id(self, layer: int, global_slot: int) -> int:
        return int(self.perm[layer][global_slot])

    def local_slot_to_original_id(self, layer: int, ep_rank: int, local_slot: int) -> int:
        return self.original_id(layer, ep_rank * self.num_local_experts + local_slot)

    def original_id_to_rank_slot(self, layer: int, original_id: int) -> tuple[int, int]:
        """``(ep_rank, local_slot)`` that holds an original expert."""
        return divmod(int(self.inv[layer][original_id]), self.num_local_experts)

    def trainable_original_ids(self, layer: int, ep_rank: int) -> list[int]:
        start = ep_rank * self.num_local_experts
        return self.perm[layer][start : start + self.num_local_trainable].tolist()

    # ---- router ---------------------------------------------------------------------------------

    def permute_router_rows(self, layer: int, tensor: torch.Tensor) -> torch.Tensor:
        """HF-order router weight ``[E, H]`` or expert bias ``[E]`` -> slot order."""
        return tensor.index_select(0, self.perm[layer].to(tensor.device))

    def unpermute_router_rows(self, layer: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, self.inv[layer].to(tensor.device))

    # ---- R3 / routing statistics ----------------------------------------------------------------

    def original_to_slot_ids(self, layer: int, ids: torch.Tensor) -> torch.Tensor:
        """Rollout routed expert ids (original) -> slot ids; any shape, any int dtype.

        Negative entries are padding (``fill_replay_data`` pads with -1) and pass through unchanged.
        """
        return _lookup(self.inv[layer], ids)

    def slot_to_original_ids(self, layer: int, ids: torch.Tensor) -> torch.Tensor:
        return _lookup(self.perm[layer], ids)


def _lookup(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    table = table.to(device=ids.device, dtype=ids.dtype)
    # ids.long() returns ids itself for int64 input: never clamp in place
    mapped = table[ids.long().clamp(min=0)]
    return torch.where(ids < 0, ids, mapped)


_ACTIVE: list[SlotLayout] = []


def set_active_slot_layout(layout: SlotLayout | None) -> None:
    """Install the process's layout (one per trainer process); ``None`` clears it."""
    _ACTIVE.clear()
    if layout is not None:
        _ACTIVE.append(layout)


def get_active_slot_layout() -> SlotLayout | None:
    """The layout this process trains with, or ``None`` when partial expert training is off."""
    return _ACTIVE[0] if _ACTIVE else None


def load_slot_layout(path: str, *, num_experts: int, ep_size: int) -> SlotLayout:
    """Load and validate a selection against the run's expert count and EP size; installs it as active."""
    selection = load_expert_selection(path)
    assert selection.num_experts == num_experts, (
        f"{path}: selection has {selection.num_experts} experts, the model {num_experts}"
    )
    assert selection.ep_size == ep_size, (
        f"{path}: slot_order was built for ep_size {selection.ep_size}, the run uses {ep_size}; regenerate "
        "the selection for this EP size (the trainable set does not change)"
    )
    layout = SlotLayout.from_selection(selection)
    set_active_slot_layout(layout)
    return layout
