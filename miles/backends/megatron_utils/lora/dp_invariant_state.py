"""DP-invariant LoRA training state: optimizer state keyed by parameter name, plus RNG capture/restore.

The default LoRA checkpoint stores ``optimizer.state_dict()`` per global rank, whose ``state`` is keyed by the
optimizer's positional parameter index. That index depends on how the optimizer was built on that rank, so a
checkpoint written at one DP size cannot be loaded at another. Here the state is re-keyed by parameter name and
written once per (tp, pp, dp) coordinate together with the range of the parameter that rank owns; a loader
gathers every DP shard of its own (tp, pp) coordinate into full tensors and slices out the ranges it owns now, so
the DP size may change while TP/PP stay fixed. The fp32 main copies of bf16 training (Float16Optimizer) and the
DP-sharded main params/state of Megatron's DistributedOptimizer are covered (see ``optimizer_slots``).

Everything in this module is pure torch/CPU; Megatron optimizers are accessed by duck typing.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

NAMED_STATE_FORMAT = "lora_named_optimizer_v2"

# What a load does with RNG. "exact": restore this rank's saved RNG and refuse a DP change or a missing coordinate.
# "keep_on_dp_change": restore when the DP size is unchanged, otherwise keep the fresh RNG of this process.
RNG_POLICIES = ("exact", "keep_on_dp_change")


class DpInvariantStateError(RuntimeError):
    """A configuration or checkpoint the DP-invariant LoRA state cannot handle."""


_NAMED_FILE_RE = re.compile(r"^training_state_named_tp(\d+)_pp(\d+)_dp(\d+)\.pt$")


def named_training_state_filename(*, tp_rank: int, pp_rank: int, dp_rank: int) -> str:
    return f"training_state_named_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt"


def dp_invariant_adapter_filename(*, tp_rank: int, pp_rank: int) -> str:
    return f"adapter_dp_invariant_tp{tp_rank}_pp{pp_rank}.pt"


def list_named_training_state_files(checkpoint_dir: Path, *, tp_rank: int, pp_rank: int) -> dict[int, Path]:
    """dp_rank -> file for every DP shard written under this (tp, pp) coordinate."""
    found: dict[int, Path] = {}
    for path in Path(checkpoint_dir).iterdir():
        if (m := _NAMED_FILE_RE.match(path.name)) and (int(m[1]), int(m[2])) == (tp_rank, pp_rank):
            found[int(m[3])] = path
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# optimizer state <-> parameter names (with DP-shard ranges)
# ---------------------------------------------------------------------------
#
# Every optimizer is reduced to "slots": one per model parameter this rank updates, holding the range
# [start, end) of the flattened model parameter that this rank's optimizer owns, plus accessors for the fp32 main
# copy ("param") and the per-element optimizer state (exp_avg, ...). Three optimizer layouts are supported:
#
# * a plain torch optimizer (or Megatron FP32Optimizer): main param == model param, full range;
# * Megatron Float16OptimizerWithFloat16Params (bf16/fp16 without DistOpt): the torch optimizer holds fp32 main
#   copies; they are matched to model params through ``float16_groups``/``fp32_from_float16_groups``; full range;
# * Megatron DistributedOptimizer: each DP rank owns a contiguous slice of every bucket, i.e. a sub-range of some
#   parameters (``gbuf_ranges[..]["param_map"][model_param]["param"]``); main shard + state come from
#   ``_get_main_param_and_optimizer_states``.
# ChainedOptimizer is unwrapped into its members. Saving writes each rank's slots keyed by parameter name with their
# ranges; loading merges every DP shard of one (tp, pp) coordinate into full flat tensors (gather) and slices out
# the ranges the new rank owns (reshard). No collective is needed: the checkpoint directory is the gather point.


class _Slot:
    __slots__ = ("name", "model_param", "start", "end", "get", "set", "group")

    def __init__(self, name, model_param, start, end, get, set_, group):
        self.name, self.model_param, self.start, self.end = name, model_param, start, end
        self.get, self.set, self.group = get, set_, group


def _optimizer_leaves(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        return [leaf for member in chained for leaf in _optimizer_leaves(member)]
    return [optimizer]


def _is_distributed(leaf: Any) -> bool:
    return hasattr(leaf, "gbuf_ranges") and hasattr(leaf, "model_param_group_index_map")


def _torch_slots(torch_optimizer: torch.optim.Optimizer, main_to_model: Mapping[int, torch.nn.Parameter], name_of):
    slots = []
    for group in torch_optimizer.param_groups:
        for main in group["params"]:
            model_param = main_to_model.get(id(main), main)
            if id(model_param) not in name_of:
                raise NotImplementedError(
                    f"optimizer holds a tensor of shape {tuple(main.shape)} that maps to no model parameter"
                )

            def get(main=main):
                tensors = {"param": main}
                tensors.update(
                    {k: v for k, v in torch_optimizer.state.get(main, {}).items() if isinstance(v, torch.Tensor)}
                )
                return tensors

            def set_(tensors, main=main):
                state = torch_optimizer.state[main]
                for key, value in tensors.items():
                    if key == "param":
                        main.data.copy_(value.reshape(main.shape))
                    elif value.dim() == 0:
                        state[key] = value.clone()
                    else:
                        state[key] = value.reshape(main.shape).to(device=main.device).clone()

            slots.append(_Slot(name_of[id(model_param)], model_param, 0, model_param.numel(), get, set_, group))
    return slots


def _distributed_slots(leaf: Any, name_of) -> list[_Slot]:
    if getattr(getattr(leaf, "config", None), "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
        raise NotImplementedError("DP-invariant LoRA state does not support the precision-aware optimizer")
    slots = []
    for gbuf_range_maps in leaf.gbuf_ranges:
        for per_bucket in gbuf_range_maps.values():
            for bucket_range_map in per_bucket:
                for model_param, range_map in bucket_range_map["param_map"].items():
                    if id(model_param) not in name_of:
                        raise NotImplementedError(
                            f"distributed optimizer holds a parameter of shape {tuple(model_param.shape)} that is not "
                            f"a named model parameter"
                        )
                    group_index, _ = leaf.model_param_group_index_map[model_param]

                    def get(model_param=model_param):
                        return leaf._get_main_param_and_optimizer_states(model_param)

                    def set_(tensors, model_param=model_param):
                        current = leaf._get_main_param_and_optimizer_states(model_param)
                        missing = current.keys() - tensors.keys()
                        if missing:
                            raise DpInvariantStateError(
                                f"saved state of {name_of[id(model_param)]!r} lacks {sorted(missing)}"
                            )
                        leaf._set_main_param_and_optimizer_states(model_param, tensors)

                    param_range = range_map["param"]
                    slots.append(
                        _Slot(
                            name_of[id(model_param)],
                            model_param,
                            param_range.start,
                            param_range.end,
                            get,
                            set_,
                            leaf.optimizer.param_groups[group_index],
                        )
                    )
    return slots


def optimizer_slots(optimizer: Any, named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> list[_Slot]:
    """The parameter slots this rank's optimizer updates (see the section comment)."""
    name_of = {}
    for name, param in named_params:
        assert id(param) not in name_of, f"parameter {name!r} is registered twice"
        name_of[id(param)] = name
    slots = []
    for leaf in _optimizer_leaves(optimizer):
        if isinstance(leaf, torch.optim.Optimizer):
            slots += _torch_slots(leaf, {}, name_of)
        elif _is_distributed(leaf):
            slots += _distributed_slots(leaf, name_of)
        elif hasattr(leaf, "float16_groups") and hasattr(leaf, "fp32_from_float16_groups"):
            main_to_model = {
                id(main): model
                for model_group, main_group in zip(leaf.float16_groups, leaf.fp32_from_float16_groups, strict=True)
                for model, main in zip(model_group, main_group, strict=True)
            }
            slots += _torch_slots(leaf.optimizer, main_to_model, name_of)
        elif isinstance(getattr(leaf, "optimizer", None), torch.optim.Optimizer):
            slots += _torch_slots(leaf.optimizer, {}, name_of)
        else:
            raise NotImplementedError(f"DP-invariant LoRA state does not support {type(leaf).__name__}")
    names = [slot.name for slot in slots]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    assert not duplicated, f"parameters {duplicated} appear in more than one optimizer slot"
    return slots


def _hyper(group: Mapping[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}


def export_named_optimizer_state(optimizer: Any, named_params) -> dict[str, Any]:
    """This rank's optimizer state keyed by parameter name, each entry tagged with its range in the parameter."""
    entries = {}
    for slot in optimizer_slots(optimizer, named_params):
        tensors, scalars = {}, {}
        for key, value in slot.get().items():
            value = value.detach().cpu()
            if value.dim() == 0:
                scalars[key] = value.clone()
                continue
            flat = value.reshape(-1).clone()
            if flat.numel() != slot.end - slot.start:
                raise DpInvariantStateError(
                    f"{slot.name!r}: state {key!r} has {flat.numel()} elements but the owned range is "
                    f"[{slot.start}, {slot.end})"
                )
            tensors[key] = flat
        entries[slot.name] = {
            "numel": slot.model_param.numel(),
            "shape": tuple(slot.model_param.shape),
            "start": slot.start,
            "end": slot.end,
            "tensors": tensors,
            "scalars": scalars,
            "hyper": _hyper(slot.group),
        }
    return {"format": NAMED_STATE_FORMAT, "entries": entries}


def merge_named_optimizer_states(shards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gather the DP shards of one (tp, pp) coordinate into full flat tensors per parameter name.

    Ranges written by more than one rank (replicated optimizers) must agree exactly; every parameter must be
    covered completely, otherwise a shard file is missing.
    """
    assert shards, "no shards to merge"
    merged: dict[str, dict[str, Any]] = {}
    covered: dict[str, torch.Tensor] = {}
    for shard in shards:
        if shard.get("format") != NAMED_STATE_FORMAT:
            raise DpInvariantStateError(f"not a {NAMED_STATE_FORMAT} optimizer state: {shard.get('format')!r}")
        for name, entry in shard["entries"].items():
            start, end = entry["start"], entry["end"]
            if name not in merged:
                merged[name] = {
                    "numel": entry["numel"],
                    "shape": tuple(entry["shape"]),
                    "tensors": {
                        k: torch.zeros(entry["numel"], dtype=v.dtype) for k, v in entry["tensors"].items()
                    },
                    "scalars": dict(entry["scalars"]),
                    "hyper": entry["hyper"],
                }
                covered[name] = torch.zeros(entry["numel"], dtype=torch.bool)
            target = merged[name]
            if (target["numel"], target["shape"]) != (entry["numel"], tuple(entry["shape"])):
                raise DpInvariantStateError(f"DP shards disagree on the shape of {name!r}")
            if target["tensors"].keys() != entry["tensors"].keys():
                raise DpInvariantStateError(f"DP shards disagree on the state keys of {name!r}")
            if not _equal(target["scalars"], entry["scalars"]):
                raise DpInvariantStateError(f"DP shards disagree on the scalar state (e.g. step) of {name!r}")
            if not _equal(target["hyper"], entry["hyper"]):
                raise DpInvariantStateError(f"DP shards disagree on the hyperparameters of {name!r}")
            overlap = covered[name][start:end]
            for key, piece in entry["tensors"].items():
                existing = target["tensors"][key][start:end]
                if overlap.any() and not torch.equal(existing[overlap], piece[overlap]):
                    raise DpInvariantStateError(f"DP shards disagree on the optimizer state {key!r} of {name!r}")
                existing.copy_(piece)
            covered[name][start:end] = True
    incomplete = sorted(name for name, mask in covered.items() if not bool(mask.all()))
    if incomplete:
        raise DpInvariantStateError(
            f"optimizer state of {incomplete} is not fully covered; a DP shard file is missing"
        )
    return merged


def check_named_optimizer_state(optimizer: Any, named_params, merged: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate that ``merged`` fits this rank's optimizer without writing anything (names, sizes, groups)."""
    slots = optimizer_slots(optimizer, named_params)
    missing = sorted(slot.name for slot in slots if slot.name not in merged)
    if missing:
        raise KeyError(f"named optimizer state lacks parameters {missing}")
    for slot in slots:
        if merged[slot.name]["numel"] != slot.model_param.numel():
            raise DpInvariantStateError(
                f"{slot.name!r} has {slot.model_param.numel()} elements, the checkpoint {merged[slot.name]['numel']}"
            )
    for names in _names_by_group(slots).values():
        hypers = [merged[n]["hyper"] for n in names[1]]
        if any(not _equal(h, hypers[0]) for h in hypers):
            raise DpInvariantStateError(f"parameters {names[1]} come from groups with different hyperparameters")


def _names_by_group(slots) -> dict[int, tuple[dict, list[str]]]:
    groups: dict[int, tuple[dict, list[str]]] = {}
    for slot in slots:
        groups.setdefault(id(slot.group), (slot.group, []))[1].append(slot.name)
    return groups


def load_named_optimizer_state(optimizer: Any, named_params, merged: Mapping[str, Mapping[str, Any]]) -> None:
    """Scatter the merged (gathered) state onto the ranges this rank's optimizer owns.

    Everything is validated (``check_named_optimizer_state``) before the optimizer is touched.
    """
    named_params = list(named_params)
    check_named_optimizer_state(optimizer, named_params, merged)
    for leaf in _optimizer_leaves(optimizer):
        if _is_distributed(leaf) and not leaf.optimizer.state:
            # A freshly built DistOpt has no state tensors yet; Megatron's own loaders initialize them first.
            leaf._init_optimizer_states_with_dummy_values()
    slots = optimizer_slots(optimizer, named_params)
    for slot in slots:
        full = merged[slot.name]
        tensors = {k: v[slot.start : slot.end] for k, v in full["tensors"].items()}
        tensors.update({k: v.clone() for k, v in full["scalars"].items()})
        slot.set(tensors)
    for group, names in _names_by_group(slots).values():
        group.update(copy.deepcopy(merged[names[0]]["hyper"]))


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
        state["megatron_cuda_tracker"] = _megatron_tracker_states()
    return state


def _megatron_tracker_states() -> dict | None:
    """Megatron's model-parallel CUDA RNG tracker (dropout etc.); None when Megatron is absent. GPU-only path."""
    try:
        from megatron.core import tensor_parallel
    except ImportError:
        return None
    return tensor_parallel.get_cuda_rng_tracker().get_states()


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if (cuda_state := state.get("torch_cuda")) is not None:
        assert torch.cuda.is_available(), "the checkpoint holds CUDA RNG state but CUDA is unavailable"
        assert len(cuda_state) == torch.cuda.device_count(), (
            f"the checkpoint holds CUDA RNG state for {len(cuda_state)} devices, this process sees "
            f"{torch.cuda.device_count()}"
        )
        torch.cuda.set_rng_state_all(cuda_state)
    if (tracker_states := state.get("megatron_cuda_tracker")) is not None:
        from megatron.core import tensor_parallel

        tensor_parallel.get_cuda_rng_tracker().set_states(tracker_states)
