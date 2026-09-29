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


def _named(model):
    return model.named_parameters()


def _export(model, optimizer):
    return dps.export_named_optimizer_state(optimizer, _named(model))


def _load(model, optimizer, shards):
    dps.load_named_optimizer_state(optimizer, _named(model), dps.merge_named_optimizer_states(shards))


class TestNamedOptimizerState:
    def test_state_is_keyed_by_name_and_survives_a_different_param_order(self):
        src = _Model(0)
        src_opt = _optimizer(src, _NAMES)
        _step(src, src_opt, 1)
        named = _export(src, src_opt)

        tgt = _Model(0)
        tgt_opt = _optimizer(tgt, list(reversed(_NAMES)))
        _load(tgt, tgt_opt, [named])
        for name in _NAMES:
            s, t = src_opt.state[src.params[name]], tgt_opt.state[tgt.params[name]]
            torch.testing.assert_close(t["exp_avg"], s["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(t["step"], s["step"], rtol=0, atol=0)
            torch.testing.assert_close(tgt.params[name].data, src.params[name].data, rtol=0, atol=0)
        assert tgt_opt.param_groups[1]["lr"] == 0.01

    def test_replicated_shards_must_agree(self):
        model = _Model(0)
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 1)
        a = _export(model, opt)
        _step(model, opt, 2)
        b = _export(model, opt)
        assert set(dps.merge_named_optimizer_states([a, a])) == set(_NAMES)
        with pytest.raises(dps.DpInvariantStateError, match="disagree"):
            dps.merge_named_optimizer_states([a, b])

    def test_missing_names_and_mixed_hyperparameters_are_refused(self):
        model = _Model(0)
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 1)
        merged = dps.merge_named_optimizer_states([_export(model, opt)])
        merged.pop(_NAMES[0])
        with pytest.raises(KeyError, match="lacks parameters"):
            dps.load_named_optimizer_state(opt, _named(model), merged)

        merged = dps.merge_named_optimizer_states([_export(model, opt)])
        mixed = torch.optim.AdamW([{"params": [model.params[_NAMES[0]], model.params[_NAMES[2]]]}])
        with pytest.raises(dps.DpInvariantStateError, match="different hyperparameters"):
            dps.load_named_optimizer_state(mixed, [(n, model.params[n]) for n in (_NAMES[0], _NAMES[2])], merged)

    def test_an_optimizer_over_non_model_tensors_is_refused(self):
        model = _Model(0)
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))])
        with pytest.raises(NotImplementedError, match="maps to no model parameter"):
            _export(model, opt)

    def test_an_unknown_optimizer_wrapper_is_refused(self):
        with pytest.raises(NotImplementedError, match="does not support"):
            _export(_Model(0), SimpleNamespace())


class _Float16Optimizer:
    """The attribute layout of Megatron Float16OptimizerWithFloat16Params (whose ctor needs CUDA tensors)."""

    def __init__(self, model, lr=0.1):
        self.float16_groups = [[p for p in model.parameters()]]
        self.fp32_from_float16_groups = [[p.detach().clone().float() for p in self.float16_groups[0]]]
        self.fp32_from_fp32_groups = [[]]
        self.optimizer = torch.optim.Adam(self.fp32_from_float16_groups[0], lr=lr, foreach=False)

    def step(self, grads):
        for main, g in zip(self.fp32_from_float16_groups[0], grads, strict=True):
            main.grad = g.clone()
        self.optimizer.step()
        for model_p, main in zip(self.float16_groups[0], self.fp32_from_float16_groups[0], strict=True):
            model_p.data.copy_(main)


class _Bf16Model(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = torch.nn.Parameter(torch.randn(3, 5, generator=g).bfloat16())
        self.lora_B = torch.nn.Parameter(torch.randn(7, generator=g).bfloat16())


class TestFloat16Optimizer:
    def test_fp32_main_params_and_state_roundtrip_by_name(self):
        src = _Bf16Model(0)
        opt = _Float16Optimizer(src)
        opt.step([torch.randn(3, 5), torch.randn(7)])
        named = _export(src, opt)
        assert named["entries"]["lora_A"]["tensors"]["param"].dtype == torch.float32

        tgt = _Bf16Model(1)
        tgt_opt = _Float16Optimizer(tgt)
        _load(tgt, tgt_opt, [named])
        for s_main, t_main in zip(opt.fp32_from_float16_groups[0], tgt_opt.fp32_from_float16_groups[0], strict=True):
            torch.testing.assert_close(t_main, s_main, rtol=0, atol=0)  # full fp32 precision, not the bf16 copy
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(tgt_opt.optimizer.state[t_main][key], opt.optimizer.state[s_main][key])
        grads = [torch.randn(3, 5), torch.randn(7)]
        opt.step(grads)
        tgt_opt.step(grads)
        for s_main, t_main in zip(opt.fp32_from_float16_groups[0], tgt_opt.fp32_from_float16_groups[0], strict=True):
            torch.testing.assert_close(t_main, s_main, rtol=0, atol=0)


# --- DistributedOptimizer: the real Megatron range/state accessors on a CPU-constructible object -------------------

_dist = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")


class _DistOpt:
    """One DP rank of a Megatron DistributedOptimizer over one flat bf16 buffer holding ``params`` in order.

    The ranges come from ``DistributedOptimizer._build_model_gbuf_param_range_map`` and the state accessors are the
    real DistributedOptimizer methods; only the CUDA grad buffer is replaced by the index arithmetic it performs.
    """

    _get_main_param_and_optimizer_states = _dist.DistributedOptimizer._get_main_param_and_optimizer_states
    _set_main_param_and_optimizer_states = _dist.DistributedOptimizer._set_main_param_and_optimizer_states
    _init_optimizer_states_with_dummy_values = _dist.DistributedOptimizer._init_optimizer_states_with_dummy_values

    def __init__(self, params, full_main, *, dp_rank, dp_size, lr=0.1):
        index_map, offset = {}, 0
        for p in params:
            index_map[p] = (offset, offset + p.numel(), 0)
            offset += p.numel()
        shard = -(-offset // dp_size)
        world = _dist.Range(dp_rank * shard, min((dp_rank + 1) * shard, offset))
        param_map = _dist.DistributedOptimizer._build_model_gbuf_param_range_map(index_map, world, 0)
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map, mains = {}, []
        for p, ranges in param_map.items():
            r = ranges["param"]
            self.model_param_group_index_map[p] = (0, len(mains))
            mains.append(full_main[p].reshape(-1)[r.start : r.end].clone())
        self.param_map = param_map
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = torch.optim.AdamW(
            [{"params": mains, "lr": lr, "weight_decay": 0.01}] if mains else [{"params": [torch.zeros(0)]}],
            foreach=False,
        )

    def step(self, full_grads):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            self.optimizer.param_groups[gi]["params"][go].grad = full_grads[p].reshape(-1)[r.start : r.end].clone()
        self.optimizer.step()

    def gather_into(self, full):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            full[p].reshape(-1)[r.start : r.end] = self.optimizer.param_groups[gi]["params"][go].detach()

    def state_dict(self):
        return {}


def _params_of(model):
    return [p for _, p in model.named_parameters()]


def _grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(p.shape, generator=g) for p in _params_of(model)}


class TestDistributedOptimizerGatherReshard:
    """Save at one DP size, gather through the files, reshard at another; the next step must match exactly."""

    def _reference(self, model, init):
        mains = [init[p].clone().reshape(-1) for p in _params_of(model)]
        opt = torch.optim.AdamW([{"params": mains, "lr": 0.1, "weight_decay": 0.01}], foreach=False)

        def step(grads):
            for m, p in zip(mains, _params_of(model), strict=True):
                m.grad = grads[p].reshape(-1).clone()
            opt.step()

        return mains, step

    @pytest.mark.parametrize("save_dp, load_dp", [(2, 1), (2, 3), (1, 4), (4, 2)])
    def test_the_next_step_after_a_dp_change_matches_an_unsharded_run(self, save_dp, load_dp):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        ref_mains, ref_step = self._reference(model, init)

        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=save_dp) for r in range(save_dp)]
        for seed in (1, 2):
            for opt in savers:
                opt.step(_grads(model, seed))
            ref_step(_grads(model, seed))
        shards = [_export(model, opt) for opt in savers]

        zeros = {p: torch.zeros(p.shape) for p in _params_of(model)}
        loaders = [_DistOpt(_params_of(model), zeros, dp_rank=r, dp_size=load_dp) for r in range(load_dp)]
        for opt in loaders:
            _load(model, opt, shards)
            opt.step(_grads(model, 3))
        ref_step(_grads(model, 3))

        gathered = {p: torch.full(p.shape, float("nan")) for p in _params_of(model)}
        for opt in loaders:
            opt.gather_into(gathered)
        for p, ref in zip(_params_of(model), ref_mains, strict=True):
            torch.testing.assert_close(gathered[p].reshape(-1), ref.detach(), rtol=0, atol=0)

    def test_a_missing_dp_shard_is_detected(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=2) for r in range(2)]
        for opt in savers:
            opt.step(_grads(model, 1))
        with pytest.raises(dps.DpInvariantStateError, match="not fully covered"):
            dps.merge_named_optimizer_states([_export(model, savers[0])])

    def test_chained_optimizers_are_unwrapped(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        dist = _DistOpt(_params_of(model), init, dp_rank=0, dp_size=1)
        dist.step(_grads(model, 1))
        chained = SimpleNamespace(chained_optimizers=[dist])
        assert set(_export(model, chained)["entries"]) == {"lora_A", "lora_B"}


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


def _patch_parallel_state(monkeypatch, *, dp_rank: int, dp_size: int = 1, cp_size: int = 1, ep_size: int = 1):
    group = lambda r, n=1: SimpleNamespace(rank=r, size=n)  # noqa: E731
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(
            tp=group(0), pp=group(0), intra_dp=group(dp_rank, dp_size), cp=group(0, cp_size), ep=group(0, ep_size)
        ),
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

    def _save_dp2(self, tmp_path, monkeypatch):
        """Two DP ranks save into one directory; returns (model, optimizer, dir, RNG draw expected on dp rank 0)."""
        model = _AdapterModel(1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.lora_A.grad = torch.ones(2, 2)
        model.lora_B.grad = torch.ones(2, 2)
        optimizer.step()

        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        expected = None
        for dp_rank in (1, 0):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            torch.manual_seed(100 + dp_rank)
            if dp_rank == 0:
                expected = torch.rand(1)
                torch.manual_seed(100)
            self._save(model, optimizer, tmp_path / f"dp{dp_rank}", flag=True)
            for f in (tmp_path / f"dp{dp_rank}").iterdir():
                if "dp_invariant" in f.name or "named" in f.name:
                    shutil.copy(f, merged_dir / f.name)
        return model, optimizer, merged_dir, expected

    def _load(self, merged_dir, *, rng_policy="exact"):
        target = _AdapterModel(0.0)
        target_opt = torch.optim.Adam(target.parameters(), lr=0.1)
        scheduler_loads = []
        result = lora_utils.load_lora_adapter(
            [target],
            str(merged_dir),
            optimizer=target_opt,
            opt_param_scheduler=SimpleNamespace(load_state_dict=scheduler_loads.append),
            dp_invariant=True,
            rng_policy=rng_policy,
        )
        return result, target, target_opt, scheduler_loads

    def test_same_dp_size_restores_optimizer_and_exact_rng(self, tmp_path, monkeypatch):
        model, optimizer, merged_dir, expected = self._save_dp2(tmp_path, monkeypatch)
        assert sorted(p.name for p in merged_dir.iterdir()) == [
            "adapter_dp_invariant_tp0_pp0.pt",
            "training_state_named_tp0_pp0_dp0.pt",
            "training_state_named_tp0_pp0_dp1.pt",
        ]
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=2)
        torch.manual_seed(7)

        result, target, target_opt, scheduler_loads = self._load(merged_dir)

        assert result == (True, 4, True)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)
        assert scheduler_loads == [{"lr": 0.5}]
        torch.testing.assert_close(torch.rand(1), expected, rtol=0, atol=0)

    def test_a_dp_change_loads_optimizer_by_name_under_keep_on_dp_change(self, tmp_path, monkeypatch):
        """DP=2 -> DP=1: optimizer state comes back by name; RNG stays the fresh one of the new process."""
        model, optimizer, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        torch.manual_seed(7)
        fresh = torch.rand(1)
        torch.manual_seed(7)

        result, _, target_opt, _ = self._load(merged_dir, rng_policy="keep_on_dp_change")

        assert result == (True, 4, True)
        torch.testing.assert_close(torch.rand(1), fresh, rtol=0, atol=0)
        saved_state = optimizer.state_dict()["state"]
        loaded_state = target_opt.state_dict()["state"]
        for index in saved_state:
            torch.testing.assert_close(loaded_state[index]["exp_avg"], saved_state[index]["exp_avg"])

    def test_a_dp_change_is_refused_under_the_exact_rng_policy(self, tmp_path, monkeypatch):
        _, _, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)

        with pytest.raises(dps.DpInvariantStateError, match="explicit policy"):
            self._load(merged_dir, rng_policy="exact")

    def test_an_unknown_rng_policy_is_refused(self, tmp_path, monkeypatch):
        _, _, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="unknown RNG policy"):
            self._load(merged_dir, rng_policy="whatever")

    def test_dp_invariant_load_falls_back_to_rank_shards(self, tmp_path, monkeypatch):
        _patch_parallel_state(monkeypatch, dp_rank=0)
        model = _AdapterModel(3.0)
        self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=False)
        target = _AdapterModel(0.0)

        loaded, iteration, _ = lora_utils.load_lora_adapter([target], str(tmp_path / "ckpt"), dp_invariant=True)

        assert (loaded, iteration) == (True, None)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)

    @pytest.mark.parametrize("sizes", [dict(ep_size=2), dict(cp_size=2)])
    def test_expert_and_context_parallelism_are_refused_at_runtime(self, tmp_path, monkeypatch, sizes):
        _patch_parallel_state(monkeypatch, dp_rank=0, **sizes)
        model = _AdapterModel(1.0)
        with pytest.raises(dps.DpInvariantStateError, match="EP=1 and CP=1"):
            self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=True)


class TestDistributedOptimizerEndToEnd:
    """save_lora_checkpoint at DP=2 with a bf16 model under a DistributedOptimizer, load_lora_adapter at DP=1."""

    def test_dp2_to_dp1_through_the_checkpoint_directory(self, tmp_path, monkeypatch):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=2) for r in range(2)]
        for opt in savers:
            opt.step(_grads(model, 1))
        full = {p: torch.zeros(p.shape) for p in _params_of(model)}
        for opt in savers:
            opt.gather_into(full)
        for p in _params_of(model):
            p.data.copy_(full[p])

        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        for dp_rank, opt in enumerate(savers):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
            out = tmp_path / f"dp{dp_rank}"
            lora_utils.save_lora_checkpoint(
                [model], args, str(out), publisher=SimpleNamespace(write_adapter=lambda *_: None),
                optimizer=opt, opt_param_scheduler=None, iteration=9,
            )
            for f in out.iterdir():
                if "dp_invariant" in f.name or "named" in f.name:
                    shutil.copy(f, merged_dir / f.name)

        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        target = _Bf16Model(5)
        zeros = {p: torch.zeros(p.shape) for p in _params_of(target)}
        loader = _DistOpt(_params_of(target), zeros, dp_rank=0, dp_size=1)
        result = lora_utils.load_lora_adapter(
            [target], str(merged_dir), optimizer=loader, dp_invariant=True, rng_policy="keep_on_dp_change"
        )

        assert result == (True, 9, True)
        gathered = {p: torch.zeros(p.shape) for p in _params_of(target)}
        loader.gather_into(gathered)
        for src_p, tgt_p in zip(_params_of(model), _params_of(target), strict=True):
            torch.testing.assert_close(gathered[tgt_p], full[src_p], rtol=0, atol=0)
            torch.testing.assert_close(tgt_p.data, src_p.data, rtol=0, atol=0)

    def test_the_exact_policy_refuses_a_dp_change_before_touching_the_model(self, tmp_path, monkeypatch):
        model = _AdapterModel(1.0)
        opt = torch.optim.Adam(model.parameters())
        for dp_rank in (0, 1):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
            lora_utils.save_lora_checkpoint(
                [model],
                args,
                str(tmp_path / f"dp{dp_rank}"),
                publisher=SimpleNamespace(write_adapter=lambda *_: None),
                optimizer=opt, opt_param_scheduler=None, iteration=1,
            )
        merged = tmp_path / "dp0"
        shutil.copy(tmp_path / "dp1" / "training_state_named_tp0_pp0_dp1.pt", merged)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        target = _AdapterModel(0.0)
        with pytest.raises(dps.DpInvariantStateError, match="explicit policy"):
            lora_utils.load_lora_adapter([target], str(merged), dp_invariant=True, rng_policy="exact")
        assert torch.equal(target.lora_A.data, torch.zeros(2, 2))


class TestLoadValidatesBeforeWriting:
    def _save_bf16_dist(self, tmp_path, monkeypatch):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        opt = _DistOpt(_params_of(model), init, dp_rank=0, dp_size=1)
        opt.step(_grads(model, 1))
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
        lora_utils.save_lora_checkpoint(
            [model],
            args,
            str(tmp_path / "ckpt"),
            publisher=SimpleNamespace(write_adapter=lambda *_: None),
            optimizer=opt,
            opt_param_scheduler=None,
            iteration=1,
        )
        return tmp_path / "ckpt"

    def test_an_optimizer_that_does_not_fit_leaves_the_adapter_untouched(self, tmp_path, monkeypatch):
        ckpt = self._save_bf16_dist(tmp_path, monkeypatch)
        target = _Bf16Model(5)
        before = {n: p.detach().clone() for n, p in target.named_parameters()}
        # the target optimizer also updates a parameter the checkpoint does not have
        extra = torch.nn.Parameter(torch.zeros(3))
        target.register_parameter("lora_C", extra)
        opt = torch.optim.AdamW(list(target.parameters()))

        with pytest.raises(KeyError, match="lacks parameters"):
            lora_utils.load_lora_adapter([target], str(ckpt), optimizer=opt, dp_invariant=True)
        for n, p in target.named_parameters():
            if n in before:
                assert torch.equal(p, before[n])

    def test_a_failing_optimizer_write_restores_the_adapter(self, tmp_path, monkeypatch):
        ckpt = self._save_bf16_dist(tmp_path, monkeypatch)
        target = _Bf16Model(5)
        before = {n: p.detach().clone() for n, p in target.named_parameters()}
        zeros = {p: torch.zeros(p.shape) for p in _params_of(target)}
        opt = _DistOpt(_params_of(target), zeros, dp_rank=0, dp_size=1)
        monkeypatch.setattr(
            dps, "load_named_optimizer_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("write failed"))
        )

        with pytest.raises(RuntimeError, match="write failed"):
            lora_utils.load_lora_adapter([target], str(ckpt), optimizer=opt, dp_invariant=True)
        for n, p in target.named_parameters():
            assert torch.equal(p, before[n])

    def test_tensor_valued_hyperparameters_are_compared_by_value(self):
        model = _Model(0)
        opt = torch.optim.Adam([{"params": list(model.params.values()), "lr": torch.tensor(0.1)}])
        _step(model, opt, 1)
        a, b = _export(model, opt), _export(model, opt)  # separate tensor objects, equal values
        merged = dps.merge_named_optimizer_states([a, b])
        dps.load_named_optimizer_state(opt, _named(model), merged)


class TestArgumentFailFast:
    """Validation runs on the args as ``parse_args`` leaves them, i.e. after ``set_default_megatron_args``."""

    @staticmethod
    def _args(**overrides):
        pytest.importorskip("megatron.training.arguments")
        from miles.backends.megatron_utils.arguments import set_default_megatron_args

        base = dict(
            lora_dp_invariant_state=True,
            lora_rank=8,
            optimizer="adam",
            fp16=False,
            seq_length=None,
            vocab_size=None,
            padded_vocab_size=None,
            tokenizer_model="tok",
            tokenizer_type="HuggingFaceTokenizer",
            multi_latent_attention=False,
            rope_type="rope",
            spec=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            use_precision_aware_optimizer=False,
            num_distributed_optimizer_instances=1,
        )
        return set_default_megatron_args(Namespace(**{**base, **overrides}))

    def test_the_default_megatron_config_is_accepted(self):
        from miles.utils.lora.arguments import validate_lora_dp_invariant_args

        args = self._args()
        assert args.bf16 and args.use_distributed_optimizer  # what a real launch gets
        validate_lora_dp_invariant_args(args)
        validate_lora_dp_invariant_args(self._args(optimizer="sgd"))  # bf16 without DistOpt
        validate_lora_dp_invariant_args(self._args(lora_dp_invariant_state=False, fp16=True))

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(fp16=True), "loss-scaler"),
            (dict(use_precision_aware_optimizer=True), "precision-aware"),
            (dict(num_distributed_optimizer_instances=2), "partial DP group"),
            (dict(context_parallel_size=2), "CP>1"),
            (dict(expert_model_parallel_size=4), "EP>1"),
            (dict(lora_rank=0), "needs LoRA"),
        ],
    )
    def test_unsupported_configs_fail_at_parse_time(self, overrides, match):
        from miles.utils.lora.arguments import validate_lora_dp_invariant_args

        with pytest.raises(ValueError, match=match):
            validate_lora_dp_invariant_args(self._args(**overrides))
