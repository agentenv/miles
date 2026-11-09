"""Partial routed-expert training glue shared by Miles core (torch only, no Megatron).

The split MoE module (``split_moe.PartiallyTrainableGroupedMLP``) names its halves ``trainable`` and
``frozen``, so every parameter path of a trainable expert contains ``.mlp.experts.trainable.``. Miles
core keys the freeze, the Direct weight export and the HF name conversion off that marker through the
helpers here, and never imports the Megatron-dependent module.
"""

from __future__ import annotations

import torch

from miles_plugins.models.kimi_k3.permute import SlotLayout

TRAINABLE_HALF = "trainable"
FROZEN_HALF = "frozen"
TRAINABLE_EXPERTS_MARK = f".mlp.experts.{TRAINABLE_HALF}."
FROZEN_EXPERTS_MARK = f".mlp.experts.{FROZEN_HALF}."
# set on every trainable expert parameter by freeze_all_but_trainable_experts; the optimizer's
# ParamKey(attr=...) override groups can select the experts by it
TRAINABLE_PARAM_ATTR = "partial_expert"


def is_trainable_expert_param_name(name: str) -> bool:
    return TRAINABLE_EXPERTS_MARK in name


def is_split_expert_param_name(name: str) -> bool:
    return TRAINABLE_EXPERTS_MARK in name or FROZEN_EXPERTS_MARK in name


def freeze_all_but_trainable_experts(model: torch.nn.Module) -> tuple[int, int]:
    """Make the trainable experts the only parameters that require grad; returns (trainable, frozen) counts.

    Must run before DDP wraps the model: Megatron's DDP, grad buffers and optimizer skip
    ``requires_grad=False`` parameters, so the frozen ones get no grad buffer and no optimizer state.
    The router weight is a parameter and freezes here; its expert bias is a buffer, kept fixed by
    ``--moe-router-bias-update-rate 0``.
    """
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        is_trainable = is_trainable_expert_param_name(name)
        parameter.requires_grad_(is_trainable)
        if is_trainable:
            setattr(parameter, TRAINABLE_PARAM_ATTR, True)
            trainable += 1
        else:
            frozen += 1
    return trainable, frozen


def map_replay_ids_to_slots(layout: SlotLayout | None, layer_idx: int, ids: torch.Tensor) -> torch.Tensor:
    """Routed expert ids an inference engine reported (original ids) -> the ids the trainer's router uses.

    Identity when partial expert training is off; padding (-1) passes through.
    """
    if layout is None:
        return ids
    return layout.original_to_slot_ids(layer_idx, ids)


def hf_expert_id_for_export(layout: SlotLayout, layer_idx: int, global_slot: int) -> int:
    """The original expert id an exported trainable expert carries in its HF name.

    ``named_params_and_buffers`` globalizes a local index ``j`` as ``ep_rank * num_local + j``, which for
    the trainable half (local slots ``[0, k_local)``) is the global slot; the HF name takes ``perm[slot]``.
    """
    assert layout.is_trainable_slot(global_slot), f"slot {global_slot} is frozen and never exported"
    return layout.original_id(layer_idx, global_slot)
