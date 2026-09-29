"""CPU tests for the DP-invariant LoRA training state (name-keyed optimizer state + RNG)."""

import random
import shutil
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils.lora import dp_invariant_state as dps
from miles.backends.megatron_utils.lora import utils as lora_utils

_NAMES = ["layers.0.lora_A.weight", "layers.0.lora_B.weight", "layers.1.lora_A.weight"]


class _Model(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.params = {name: torch.nn.Parameter(torch.randn(3, 2, generator=g)) for name in _NAMES}
        for index, param in enumerate(self.params.values()):
            self.register_parameter(f"p{index}", param)

    def named_parameters(self, *args, **kwargs):
        return list(self.params.items())


def _optimizer(model: _Model, order: list[str]) -> torch.optim.Optimizer:
    params = model.params
    return torch.optim.AdamW(
        [
            {"params": [params[n] for n in order if n.startswith("layers.0")], "lr": 0.1},
            {"params": [params[n] for n in order if n.startswith("layers.1")], "lr": 0.01, "weight_decay": 0.0},
        ]
    )


def _step(model: _Model, optimizer: torch.optim.Optimizer, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    optimizer.zero_grad()
    loss = sum((p * torch.randn(p.shape, generator=g)).sum() for p in model.params.values())
    loss.backward()
    optimizer.step()


def _group_names(model, optimizer):
    return dps.param_names_per_group(optimizer, model.named_parameters())


class TestNamedOptimizerState:
    def test_state_is_keyed_by_name_and_survives_a_different_param_order(self):
        """Resuming from name-keyed state gives the same next step as never stopping."""
        reference, source = _Model(), _Model()
        ref_opt, src_opt = _optimizer(reference, _NAMES), _optimizer(source, _NAMES)
        for seed in range(3):
            _step(reference, ref_opt, seed)
            _step(source, src_opt, seed)
        named = dps.optimizer_state_to_named(src_opt.state_dict(), _group_names(source, src_opt))
        assert set(named["state"]) == set(_NAMES)

        target = _Model(seed=1)
        with torch.no_grad():
            for name in _NAMES:
                target.params[name].copy_(source.params[name])
        tgt_opt = _optimizer(target, list(reversed(_NAMES)))
        tgt_opt.load_state_dict(dps.named_to_optimizer_state(named, _group_names(target, tgt_opt)))

        _step(reference, ref_opt, 99)
        _step(target, tgt_opt, 99)
        for name in _NAMES:
            torch.testing.assert_close(target.params[name], reference.params[name], rtol=0, atol=0)

    def test_disjoint_dp_shards_merge_and_load_at_a_smaller_dp(self):
        model = _Model()
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 0)
        full = dps.optimizer_state_to_named(opt.state_dict(), _group_names(model, opt))
        shard0 = {**full, "state": {n: v for n, v in full["state"].items() if n != _NAMES[2]}}
        shard1 = {**full, "state": {_NAMES[2]: full["state"][_NAMES[2]]}}

        merged = dps.merge_named_optimizer_states([shard0, shard1])

        assert set(merged["state"]) == set(_NAMES)
        fresh = _optimizer(_Model(), _NAMES)
        fresh.load_state_dict(dps.named_to_optimizer_state(merged, _group_names(model, opt)))
        assert fresh.param_groups[1]["lr"] == 0.01

    def test_replicated_shards_must_agree(self):
        model = _Model()
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 0)
        a = dps.optimizer_state_to_named(opt.state_dict(), _group_names(model, opt))
        b = dps.optimizer_state_to_named(opt.state_dict(), _group_names(model, opt))
        b["state"][_NAMES[0]]["exp_avg"] = b["state"][_NAMES[0]]["exp_avg"] + 1

        assert set(dps.merge_named_optimizer_states([a, a])["state"]) == set(_NAMES)
        with pytest.raises(ValueError, match="disagree on the optimizer state"):
            dps.merge_named_optimizer_states([a, b])

    def test_missing_names_and_mixed_hyperparameters_are_refused(self):
        model = _Model()
        opt = _optimizer(model, _NAMES)
        named = dps.optimizer_state_to_named(opt.state_dict(), _group_names(model, opt))

        with pytest.raises(KeyError, match="layers.9"):
            dps.named_to_optimizer_state(named, [["layers.9.lora_A.weight"]])
        with pytest.raises(AssertionError, match="different hyperparameters"):
            dps.named_to_optimizer_state(named, [[_NAMES[0], _NAMES[2]]])

    def test_an_optimizer_over_non_model_tensors_is_not_name_mappable(self):
        model = _Model()
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
        with pytest.raises(NotImplementedError, match="DP gather"):
            dps.param_names_per_group(opt, model.named_parameters())

    def test_a_distributed_optimizer_wrapper_is_refused(self):
        class DistributedOptimizer:
            def __init__(self, inner):
                self.optimizer = inner

        inner = _optimizer(_Model(), _NAMES)
        assert dps.unwrap_name_mappable_optimizer(SimpleNamespace(optimizer=inner)) is inner
        with pytest.raises(NotImplementedError, match="DistributedOptimizer shards its state"):
            dps.unwrap_name_mappable_optimizer(DistributedOptimizer(inner))


class TestRngState:
    def test_capture_then_restore_replays_every_cpu_generator(self):
        state = dps.capture_rng_state()
        first = (random.random(), np.random.rand(), torch.rand(2))

        dps.restore_rng_state(state)
        second = (random.random(), np.random.rand(), torch.rand(2))

        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_the_state_survives_torch_save(self, tmp_path):
        state = dps.capture_rng_state()
        expected = torch.rand(3)
        torch.save(state, tmp_path / "rng.pt")

        dps.restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=False))

        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)


def _patch_parallel_state(monkeypatch, *, dp_rank: int):
    rank = lambda r: SimpleNamespace(rank=r, size=1)  # noqa: E731
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=rank(0), pp=rank(0), intra_dp_cp=rank(dp_rank), ep=rank(0)),
    )


class _AdapterModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.full((2, 2), value))
        self.lora_B = torch.nn.Parameter(torch.full((2, 2), value + 1))


class TestSaveLoadDpInvariant:
    def _save(self, model, optimizer, path, *, flag: bool):
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=flag)
        lora_utils.save_lora_checkpoint(
            [model],
            args,
            str(path),
            publisher=SimpleNamespace(write_adapter=lambda *_: None),
            optimizer=optimizer,
            opt_param_scheduler=SimpleNamespace(state_dict=lambda: {"lr": 0.5}),
            iteration=4,
        )

    def test_default_format_writes_no_dp_invariant_files(self, tmp_path, monkeypatch):
        _patch_parallel_state(monkeypatch, dp_rank=0)
        model = _AdapterModel(1.0)
        self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=False)

        assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == [
            "adapter_megatron_rank0.pt",
            "training_state_rank0.pt",
        ]

    def test_a_dp2_checkpoint_restores_at_dp1_with_optimizer_and_rng(self, tmp_path, monkeypatch):
        """Two DP ranks save; a single rank (another global layout) loads by name and gets its RNG back."""
        model = _AdapterModel(1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.lora_A.grad = torch.ones(2, 2)
        model.lora_B.grad = torch.ones(2, 2)
        optimizer.step()

        merged_dir = tmp_path / "merged"
        rng_expected = None
        for dp_rank in (1, 0):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank)
            torch.manual_seed(100 + dp_rank)
            if dp_rank == 0:
                rng_expected = torch.rand(1)
                torch.manual_seed(100)
            self._save(model, optimizer, tmp_path / f"dp{dp_rank}", flag=True)
            merged_dir.mkdir(exist_ok=True)
            for f in (tmp_path / f"dp{dp_rank}").iterdir():
                if "dp_invariant" in f.name or "named" in f.name:
                    shutil.copy(f, merged_dir / f.name)
        assert sorted(p.name for p in merged_dir.iterdir()) == [
            "adapter_dp_invariant_tp0_pp0.pt",
            "training_state_named_tp0_pp0_dp0.pt",
            "training_state_named_tp0_pp0_dp1.pt",
        ]

        _patch_parallel_state(monkeypatch, dp_rank=0)
        target = _AdapterModel(0.0)
        target_opt = torch.optim.Adam(target.parameters(), lr=0.1)
        scheduler_loads = []
        torch.manual_seed(7)

        loaded, iteration, restored = lora_utils.load_lora_adapter(
            [target],
            str(merged_dir),
            optimizer=target_opt,
            opt_param_scheduler=SimpleNamespace(load_state_dict=scheduler_loads.append),
            dp_invariant=True,
        )

        assert (loaded, iteration, restored) == (True, 4, True)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)
        assert scheduler_loads == [{"lr": 0.5}]
        torch.testing.assert_close(torch.rand(1), rng_expected, rtol=0, atol=0)
        saved_state = optimizer.state_dict()["state"]
        loaded_state = target_opt.state_dict()["state"]
        for index in saved_state:
            torch.testing.assert_close(loaded_state[index]["exp_avg"], saved_state[index]["exp_avg"])

    def test_dp_invariant_load_falls_back_to_rank_shards(self, tmp_path, monkeypatch):
        _patch_parallel_state(monkeypatch, dp_rank=0)
        model = _AdapterModel(3.0)
        self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=False)
        target = _AdapterModel(0.0)

        loaded, iteration, _ = lora_utils.load_lora_adapter([target], str(tmp_path / "ckpt"), dp_invariant=True)

        assert (loaded, iteration) == (True, None)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)

    def test_expert_parallelism_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            lora_utils,
            "get_parallel_state",
            lambda: SimpleNamespace(
                tp=SimpleNamespace(rank=0),
                pp=SimpleNamespace(rank=0),
                intra_dp_cp=SimpleNamespace(rank=0),
                ep=SimpleNamespace(rank=0, size=2),
            ),
        )
        model = _AdapterModel(1.0)
        with pytest.raises(AssertionError, match="EP>1"):
            self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=True)
