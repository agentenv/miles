"""DP-invariant LoRA training state: optimizer state keyed by parameter name, plus RNG capture/restore.

The default LoRA checkpoint stores ``optimizer.state_dict()`` per global rank, whose ``state`` is keyed by the
optimizer's positional parameter index. That index depends on how the optimizer was built on that rank, so a
checkpoint written at one DP size cannot be loaded at another. Here the state is re-keyed by parameter name and
written once per (tp, pp, dp) coordinate; a loader merges every DP shard of its own (tp, pp) coordinate and picks
the names its optimizer holds, so the DP size may change while TP/PP stay fixed.

Everything in this module is pure torch/CPU and does not touch Megatron. A Megatron distributed optimizer keeps its
state in flattened, DP-sharded fp32 buffers that do not map 1:1 onto model parameters; re-keying it requires a DP
gather of the parameter state first, which is not implemented here (see ``unwrap_name_mappable_optimizer``).
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

NAMED_STATE_FORMAT = "lora_named_optimizer_v1"
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
# optimizer state <-> parameter names
# ---------------------------------------------------------------------------


def param_names_per_group(optimizer: torch.optim.Optimizer, named_params: Iterable[tuple[str, torch.nn.Parameter]]):
    """For each param group, the names of its parameters in optimizer order (matched by tensor identity)."""
    name_of = {}
    for name, param in named_params:
        assert id(param) not in name_of, f"parameter {name!r} is registered twice"
        name_of[id(param)] = name
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = []
        for param in group["params"]:
            if id(param) not in name_of:
                raise NotImplementedError(
                    f"optimizer param group {group_index} holds a tensor that is not a model parameter (shape "
                    f"{tuple(param.shape)}); a distributed/mixed-precision optimizer keeps sharded main copies and "
                    f"needs a DP gather before its state can be keyed by name"
                )
            names.append(name_of[id(param)])
        groups.append(names)
    return groups


def optimizer_state_to_named(state_dict: Mapping[str, Any], group_names: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Re-key a torch ``optimizer.state_dict()`` by parameter name."""
    groups = state_dict["param_groups"]
    assert len(groups) == len(group_names), f"{len(groups)} param groups but names for {len(group_names)}"
    index_to_name: dict[int, str] = {}
    named_groups = []
    for group, names in zip(groups, group_names, strict=True):
        assert len(group["params"]) == len(names), f"group has {len(group['params'])} params, {len(names)} names"
        index_to_name.update(zip(group["params"], names, strict=True))
        named_groups.append({**{k: v for k, v in group.items() if k != "params"}, "param_names": list(names)})
    state = {index_to_name[index]: copy.deepcopy(value) for index, value in state_dict["state"].items()}
    return {"format": NAMED_STATE_FORMAT, "param_groups": named_groups, "state": state}


def named_to_optimizer_state(named: Mapping[str, Any], group_names: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Build a torch ``state_dict`` for an optimizer whose groups hold ``group_names``.

    Hyperparameters of each target group come from the saved group holding its first name; a target whose names are
    split across saved groups with different hyperparameters is rejected.
    """
    assert named.get("format") == NAMED_STATE_FORMAT, f"not a named optimizer state: {named.get('format')!r}"
    group_of_name = {}
    for saved_group in named["param_groups"]:
        for name in saved_group["param_names"]:
            group_of_name[name] = saved_group
    wanted = [name for names in group_names for name in names]
    missing = sorted(set(wanted) - set(group_of_name))
    if missing:
        raise KeyError(f"named optimizer state lacks parameters {missing}")

    param_groups, state, index = [], {}, 0
    for names in group_names:
        hyper = [{k: v for k, v in group_of_name[n].items() if k != "param_names"} for n in names]
        assert all(
            h == hyper[0] for h in hyper
        ), f"parameters {list(names)} come from groups with different hyperparameters"
        params = []
        for name in names:
            if name in named["state"]:
                state[index] = copy.deepcopy(named["state"][name])
            params.append(index)
            index += 1
        param_groups.append({**(hyper[0] if hyper else {}), "params": params})
    return {"state": state, "param_groups": param_groups}


def merge_named_optimizer_states(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge the named states of several DP shards of the same (tp, pp) coordinate.

    Replicated (non-distributed) optimizers write identical entries on every DP rank; they must agree exactly.
    """
    assert shards, "no shards to merge"
    merged_state: dict[str, Any] = {}
    merged_groups: dict[str, dict] = {}
    for shard in shards:
        assert shard.get("format") == NAMED_STATE_FORMAT, f"not a named optimizer state: {shard.get('format')!r}"
        for name, value in shard["state"].items():
            if name in merged_state:
                if not _equal(merged_state[name], value):
                    raise ValueError(f"DP shards disagree on the optimizer state of {name!r}")
            else:
                merged_state[name] = value
        for group in shard["param_groups"]:
            hyper = {k: v for k, v in group.items() if k != "param_names"}
            for name in group["param_names"]:
                if name in merged_groups and merged_groups[name] != hyper:
                    raise ValueError(f"DP shards disagree on the hyperparameters of {name!r}")
                merged_groups[name] = hyper
    groups_by_hyper: list[tuple[dict, list[str]]] = []
    for name, hyper in merged_groups.items():
        for existing, names in groups_by_hyper:
            if existing == hyper:
                names.append(name)
                break
        else:
            groups_by_hyper.append((hyper, [name]))
    return {
        "format": NAMED_STATE_FORMAT,
        "param_groups": [{**hyper, "param_names": names} for hyper, names in groups_by_hyper],
        "state": merged_state,
    }


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def unwrap_name_mappable_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    """The plain torch optimizer behind ``optimizer`` whose params are model parameters, or NotImplementedError."""
    seen = 0
    current = optimizer
    while not isinstance(current, torch.optim.Optimizer):
        seen += 1
        inner = getattr(current, "optimizer", None)
        if inner is None or seen > 4:
            raise NotImplementedError(
                f"{type(optimizer).__name__} exposes no plain torch optimizer; DP-invariant LoRA state is "
                f"implemented only for non-distributed optimizers"
            )
        if type(current).__name__ in ("DistributedOptimizer", "ChainedOptimizer"):
            raise NotImplementedError(
                f"{type(current).__name__} shards its state over DP; keying it by name needs a DP gather that is "
                f"not implemented (disable --use-distributed-optimizer for DP-invariant LoRA state)"
            )
        current = inner
    return current


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
