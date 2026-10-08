"""Distributed (broadcast) LoRA sync with --megatron-to-hf-mode raw: the protocol asks for a PP gather, so the
whole adapter is sent by global rank 0 over one group "miles-pp_0" and the rank-0 checksum manifest covers every
PP stage. CPU only; collectives are faked."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import hashlib
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight import hf_weight_iterator as megatron_iter
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    get_hf_weight_iterator,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size, record_lora_checksums

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"
_PARALLEL = SimpleNamespace(
    pp=SimpleNamespace(size=4, rank=0), tp=SimpleNamespace(rank=0), intra_dp_cp=SimpleNamespace(rank=0)
)


def _protocol(**args) -> UpdateWeightFromDistributed:
    with (
        patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=_PARALLEL),
        patch(f"{_BROADCAST_MODULE}.create_world_ticket_lock", return_value=MagicMock()),
    ):
        return UpdateWeightFromDistributed(Namespace(**args))


class TestRequiredPlacement:
    def test_lora_requires_a_pp_gather(self) -> None:
        assert _protocol(lora_rank=16).required_placement == WeightUpdatePlacement(gather_pp=True)

    def test_full_weight_sync_keeps_pp_local_senders(self) -> None:
        assert _protocol().required_placement == WeightUpdatePlacement(gather_pp=False)
        # The class default is untouched by a LoRA instance.
        _protocol(lora_rank=16)
        assert UpdateWeightFromDistributed.required_placement == WeightUpdatePlacement(gather_pp=False)

    def test_train_only_lora_does_not_sync_adapters(self) -> None:
        assert _protocol(lora_rank=16, lora_train_only=True).required_placement.gather_pp is False

    def test_raw_iterator_resolves_to_a_pp_gather(self) -> None:
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        captured = {}

        def _init(self, args, model, *, placement, model_name, quantization_config):
            captured["placement"] = placement

        with patch.object(HfWeightIteratorDirect, "__init__", _init):
            get_hf_weight_iterator(
                Namespace(megatron_to_hf_mode="raw"),
                [MagicMock()],
                required_placement=_protocol(lora_rank=16).required_placement,
                model_name="qwen4_exp",
                quantization_config=None,
            )
        assert captured["placement"].gather_pp is True


class TestSingleSender:
    @pytest.mark.parametrize("global_rank, pp_rank, sender", [(0, 0, True), (2, 1, False), (7, 3, False)])
    def test_only_global_rank_zero_sends_over_miles_pp_0(self, global_rank, pp_rank, sender) -> None:
        protocol = _protocol(lora_rank=16)
        placement = protocol.required_placement
        with patch("miles.backends.training_utils.weight_update.utils.dist") as dist:
            dist.get_rank.return_value = global_rank
            dist.get_world_size.return_value = 8
            replica_rank, replica_size = get_data_replica_rank_and_size(SimpleNamespace(), placement)
        assert (replica_rank == 0) is sender and replica_size == 8
        with (
            patch(f"{_BROADCAST_MODULE}.get_data_replica_rank_and_size", return_value=(replica_rank, replica_size)),
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed") as connect,
        ):
            protocol.connect(
                [MagicMock()],
                [8],
                [0],
                SimpleNamespace(pp=SimpleNamespace(rank=pp_rank)),
                placement,
                "all",
            )
        assert protocol.is_sender is sender
        if sender:
            assert protocol.group_name == "miles-pp_0"
            assert connect.call_args.args[1] == "miles-pp_0"
        else:
            connect.assert_not_called()


# Two PP stages, each owning its layers' adapter (Megatron global layer numbers, so no name clashes).
_STAGE_ADAPTERS = {
    0: [
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.full((2, 3), 1.0, dtype=torch.bfloat16)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.full((3, 2), 2.0, dtype=torch.bfloat16)),
    ],
    1: [
        ("model.layers.2.mlp.experts.gate_up_proj.lora_A.weight", torch.full((4, 2, 3), 3.0, dtype=torch.bfloat16)),
        ("model.layers.2.mlp.experts.gate_up_proj.lora_B.weight", torch.full((4, 6, 2), 4.0, dtype=torch.bfloat16)),
        ("model.layers.3.linear_attn.in_proj_qkvz.lora_A.weight", torch.full((2, 3), 5.0, dtype=torch.float32)),
    ],
}


class _FakeIterator(MegatronHfWeightIteratorBase):
    def __init__(self, pp_rank: int) -> None:  # bypass the model-reading base init
        self.args = Namespace(update_weight_buffer_size=1 << 30)
        self.placement = WeightUpdatePlacement(gather_pp=True)
        self.pp_rank = pp_rank

    def _iter_hf_param_units(self, weights, *, materialize):
        raise AssertionError("LoRA sync must not stream base weights")

    def _export_pp_local_lora(self, adapter):
        return [(n, t.clone()) for n, t in _STAGE_ADAPTERS[self.pp_rank]]


def _fake_pp_collectives(pp_rank: int):
    """all_gather_object / broadcast of a 2-stage PP group seen from ``pp_rank``; the other stage's flat buffer is
    filled from _STAGE_ADAPTERS, as its broadcast would deliver it."""
    pp = SimpleNamespace(size=2, rank=pp_rank, group=object())
    dist = MagicMock()
    dist.get_process_group_ranks.return_value = [0, 4]

    def all_gather_object(out, obj, group):
        for stage in range(2):
            out[stage] = obj if stage == pp_rank else [(n, tuple(t.shape), t.dtype) for n, t in _STAGE_ADAPTERS[stage]]

    def broadcast(flat, src, group):
        stage = [0, 4].index(src)
        if stage != pp_rank:
            parts = [t.reshape(-1) for _, t in _STAGE_ADAPTERS[stage] if t.dtype == flat.dtype]
            flat.copy_(torch.cat(parts))

    dist.all_gather_object.side_effect = all_gather_object
    dist.broadcast.side_effect = broadcast
    return pp, dist


class TestGatheredAdapterStream:
    @pytest.mark.parametrize("pp_rank", [0, 1])
    def test_every_stage_holds_the_full_prefixed_adapter_and_rank0_checksums_cover_it(self, pp_rank) -> None:
        pp, dist = _fake_pp_collectives(pp_rank)
        with (
            patch.object(megatron_iter, "get_parallel_state", return_value=SimpleNamespace(pp=pp)),
            patch.object(megatron_iter, "dist", dist),
            patch.object(megatron_iter.torch.cuda, "current_device", return_value="cpu"),
        ):
            buckets = list(
                _FakeIterator(pp_rank).iter_hf_weights(None, include_base=False, adapters=[("lora", None)])
            )
        sent = [(n, t) for bucket in buckets for n, t in bucket]
        expected = {f"lora:{n}": t for stage in (0, 1) for n, t in _STAGE_ADAPTERS[stage]}
        assert sorted(n for n, _ in sent) == sorted(expected)
        for name, tensor in sent:
            assert tensor.dtype == expected[name].dtype and torch.equal(tensor, expected[name]), name

        checksums = {"lora": {}}
        for bucket in buckets:
            record_lora_checksums(bucket, checksums)
        assert set(checksums["lora"]) == {n.split(":", 1)[1] for n in expected}
        name = "model.layers.2.mlp.experts.gate_up_proj.lora_B.weight"
        want = hashlib.sha256(
            _STAGE_ADAPTERS[1][1][1].contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        assert checksums["lora"][name] == want

    def test_non_sender_ranks_join_the_gather_but_yield_nothing(self) -> None:
        pp, dist = _fake_pp_collectives(1)
        with (
            patch.object(megatron_iter, "get_parallel_state", return_value=SimpleNamespace(pp=pp)),
            patch.object(megatron_iter, "dist", dist),
            patch.object(megatron_iter.torch.cuda, "current_device", return_value="cpu"),
        ):
            buckets = list(
                _FakeIterator(1).iter_hf_weights(
                    None, include_base=False, adapters=[("lora", None)], materialize=False
                )
            )
        assert buckets == []
        assert dist.all_gather_object.call_count == 1 and dist.broadcast.call_count == 3  # 2 dtypes + 1 dtype
