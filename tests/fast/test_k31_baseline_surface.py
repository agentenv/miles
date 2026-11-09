"""The surface an out-of-tree driver relies on from the Kimi-K3.1 baseline.

A driver that runs Miles as a library probes these names at start-up (its runtime
manifest) and refuses to start on a tree that lacks one. Keep this test in step with
that manifest: every public name here is part of the contract, so a rename needs the
driver updated in the same change. Nothing here needs a GPU, Ray, Megatron or SGLang;
the pieces that import them are checked by module path or skipped.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# CLI flags of the K3.1 baseline that the driver passes or inspects. --partial-experts-selection
# is not part of this surface: it belongs to the 128-experts change that lands on top of it.
K31_BASELINE_FLAGS = (
    "--worker-dynamic-port-start",
    "--miles-dsa-topk-backend",
    "--use-indexer-replay",
    "--use-rollout-indexer-replay",
    "--qkv-format",
    "--allgather-cp",
    "--over-sampling-batch-size",
    "--megatron-to-hf-mode",
    "--experts-shared-outer-loras",
)

YETO_EXT_LEAVES = (
    "AllConfig",
    "OrchestratorConfig",
    "TrainerConfig",
    "RolloutConfig",
    "InferenceControllerConfig",
    "MultiLoraConfig",
)


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class TestRunPlugin:
    def test_trainer_controller_and_train_actor_expose_run_plugin(self) -> None:
        pytest.importorskip("ray")
        group = importlib.import_module("miles.ray.train.group")
        train_actor = importlib.import_module("miles.ray.train_actor")
        for owner in (group.TrainerController, train_actor.TrainRayActor):
            run_plugin = getattr(owner, "run_plugin", None)
            assert callable(run_plugin), f"{owner.__name__}.run_plugin is missing"
            params = list(inspect.signature(run_plugin).parameters)
            assert params[1:3] == ["fn_path", "kwargs"], (owner.__name__, params)
        assert inspect.iscoroutinefunction(group.TrainerController.run_plugin)


class TestYetoExtension:
    def test_trait_module_and_field(self) -> None:
        module = importlib.import_module("miles.utils.args.configs.yeto_extension")
        assert "yeto_ext" in module.YetoExtensionConfig.model_fields
        assert module.YetoExtensionConfig.model_validate({}).yeto_ext == {}

    def test_every_leaf_carries_yeto_ext(self) -> None:
        runtime = pytest.importorskip("miles.utils.args.runtime")
        missing = [name for name in YETO_EXT_LEAVES if "yeto_ext" not in getattr(runtime, name).model_fields]
        assert missing == []

    def test_yeto_ext_has_no_cli_flag(self) -> None:
        runtime = pytest.importorskip("miles.utils.args.runtime")
        parser = argparse.ArgumentParser()
        runtime.AllConfig.add_arguments(parser)
        assert "--yeto-ext" not in {option for action in parser._actions for option in action.option_strings}


class TestRolloutRuntimeStats:
    def test_publish_and_take(self) -> None:
        stats = importlib.import_module("miles.rollout.rollout_runtime_stats")
        assert callable(stats.publish) and callable(stats.take)
        stats.publish("abort_discard", {"groups": 1})
        assert stats.take("abort_discard") == {"groups": 1}
        assert stats.take("abort_discard") is None


class TestKimiK31Registration:
    def test_model_spec_entry_point(self) -> None:
        module = importlib.import_module("miles_plugins.models.kimi_k31")
        assert callable(module.get_kimi_k31_spec)
        assert _module_exists("miles_plugins.models.kimi_k31.model")

    def test_mbridge_plugin_module_exists_and_is_exported(self) -> None:
        # find_spec would import miles_plugins.mbridge, which needs mbridge; check the files instead
        package = REPO_ROOT / "miles_plugins" / "mbridge"
        assert (package / "kimi_k31.py").is_file()
        source = (package / "__init__.py").read_text()
        tree = ast.parse(source)
        imported = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "kimi_k31" for alias in node.names}
        assert "KimiK31Bridge" in imported

    def test_lora_registries_know_kimi_k31(self) -> None:
        hf_targets = importlib.import_module("miles.utils.lora.hf_lora_targets")
        assert "kimi_k31" in hf_targets._HF_LORA_MODELS
        registry = importlib.import_module("miles.backends.megatron_utils.lora.native.registry")
        assert "kimi_k31" in registry.MODEL_SPECS

    def test_tito_has_the_kimi_k3_family(self) -> None:
        tito = pytest.importorskip("miles.utils.chat_template_utils.tito_tokenizer")  # needs sglang
        assert tito.TITOTokenizerType("kimi_k3") is tito.TITOTokenizerType.KIMI_K3

    def test_model_args_script_and_snapshot(self) -> None:
        assert (REPO_ROOT / "scripts" / "models" / "kimi-k31.py").is_file()
        snapshot = REPO_ROOT / "tests" / "snapshots" / "model_args" / "kimi-k31.txt"
        assert "get_kimi_k31_spec" in snapshot.read_text()

    def test_support_modules(self) -> None:
        mxfp4 = importlib.import_module("miles.utils.mxfp4")
        assert callable(mxfp4.pack_mxfp4_rows) and callable(mxfp4.split_mxfp4_rows)
        replay = importlib.import_module("miles.utils.replay_base")
        assert issubclass(replay.IndexerReplayManager, replay.BaseReplayManager)
        cp_utils = pytest.importorskip("miles_plugins.models.cp_utils")
        assert callable(cp_utils.zigzag_gather_layout)


class TestBaselineFlags:
    def test_k31_baseline_flags_are_declared(self) -> None:
        runtime = pytest.importorskip("miles.utils.args.runtime")
        parser = argparse.ArgumentParser()
        runtime.AllConfig.add_arguments(parser)
        options = {option for action in parser._actions for option in action.option_strings}
        assert [flag for flag in K31_BASELINE_FLAGS if flag not in options] == []


class TestLightMegatronUtils:
    def test_runtime_patches_split_out_of_the_package_init(self) -> None:
        assert _module_exists("miles.backends.megatron_utils.runtime_patches")
        init = (REPO_ROOT / "miles" / "backends" / "megatron_utils" / "__init__.py").read_text()
        assert "deep_ep" not in init
