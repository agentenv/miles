import argparse
import json

import pytest

from miles.utils.args import runtime
from miles.utils.args.configs.data import DataConfig
from miles.utils.args.configs.yeto_extension import YetoExtensionConfig
from miles.utils.args.custom_function import CustomFunctionConfig, resolve_custom_function_configs
from miles.utils.args.custom_view import ImmutableNamespace, compute_custom_function_config
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.arguments import _validate_argument_ownership


_LEAVES = [
    value
    for value in vars(runtime).values()
    if isinstance(value, type) and issubclass(value, BaseLeafConfig) and value.__module__ == runtime.__name__
]


class _SourceLeaf(BaseLeafConfig, YetoExtensionConfig, DataConfig):
    pass


class _RolloutLikeLeaf(BaseLeafConfig, YetoExtensionConfig):
    pass


class _Hookless(BaseLeafConfig):
    pass


def _source(ext):
    defaults = {name: field.get_default(call_default_factory=True) for name, field in DataConfig.model_fields.items()}
    return _SourceLeaf.model_validate(defaults | {"yeto_ext": ext})


class TestYetoExtensionTrait:
    def test_every_runtime_leaf_and_all_config_carry_the_field(self) -> None:
        """No component slice can drop the orchestrator context."""
        assert {cls.__name__ for cls in _LEAVES} >= {
            "AllConfig",
            "OrchestratorConfig",
            "TrainerConfig",
            "RolloutConfig",
            "InferenceControllerConfig",
            "MultiLoraConfig",
        }
        assert [cls.__name__ for cls in _LEAVES if "yeto_ext" not in cls.model_fields] == []

    def test_declares_no_cli_flag(self) -> None:
        """The context never reaches argv, so CLI goldens and the parser schema stay unchanged."""
        parser = argparse.ArgumentParser()
        before = len(parser._actions)
        YetoExtensionConfig.add_arguments(parser)
        assert len(parser._actions) == before

    def test_slicing_keeps_the_value_and_json_round_trip_restores_tuples(self) -> None:
        """A slice and its wire copy carry the same context."""
        ext = {"schema": "yeto-ext-v1", "attrs": {"yeto_rl_learner_id": 3, "yeto_rl_unverified_mechanisms": ("a", "b")}}
        sliced = _RolloutLikeLeaf.slice_from(_source(ext))
        restored = _RolloutLikeLeaf.model_validate_json(sliced.model_dump_json())
        assert sliced.yeto_ext == ext
        assert restored.yeto_ext == ext

    def test_field_is_immutable_but_replaceable_before_slicing(self) -> None:
        """Assignment is refused; a pre-slice model_copy is the only way to set it after parsing."""
        source = _source({})
        with pytest.raises(TypeError, match="immutable"):
            source.yeto_ext = {"x": 1}
        updated = source.model_copy(update={"yeto_ext": {"x": 1}})
        assert updated.yeto_ext == {"x": 1}
        assert source.yeto_ext == {}

    def test_unknown_attribute_assignment_still_fails(self) -> None:
        """The old setattr(args, 'yeto_*') pattern is rejected by every leaf."""
        with pytest.raises(ValueError):
            _source({}).yeto_rl_learner_id = 1

    def test_hook_view_exposes_the_context(self) -> None:
        """A custom hook's frozen view (compute_custom_function_config) carries the context."""
        args = _RolloutLikeLeaf.model_validate({"yeto_ext": {"attrs": {"yeto_rl_learner_id": 3}}})
        view = compute_custom_function_config(args, CustomFunctionConfig(path="json.dumps"))
        assert isinstance(view, ImmutableNamespace)
        assert view.yeto_ext["attrs"]["yeto_rl_learner_id"] == 3
        with pytest.raises(ValueError):
            view.yeto_scratch = 1

    def test_preprocess_set_value_is_owned(self) -> None:
        """Setting the field on the raw namespace in preprocess_args passes the ownership check."""
        args = argparse.Namespace(yeto_ext={"attrs": {}})
        _validate_argument_ownership(args, parser=argparse.ArgumentParser(), training_backend_arg_names=set())
        with pytest.raises(ValueError, match="no configuration owner.*yeto_rl_learner_id"):
            _validate_argument_ownership(
                argparse.Namespace(yeto_rl_learner_id=1),
                parser=argparse.ArgumentParser(),
                training_backend_arg_names=set(),
            )


class _MisConfig(BaseConfig):
    tis_mode: str = "truncate"
    tis_upper_bound: float = 2.0
    flagged: A[int, Arg()] = 0


def mis_hook(args, **kwargs):
    return args.tis_mode


mis_hook.config_class = _MisConfig


class _HookLeaf(BaseConfig):
    custom_tis_function_path: CustomFunctionConfig | None = None


class TestHookScopedConfig:
    def test_preprocess_values_reach_a_hook_config_without_argv(self) -> None:
        """Non-Arg hook fields set before resolution land in the hook config and leave the namespace."""
        args = argparse.Namespace(
            custom_tis_function_path=f"{__name__}.mis_hook", tis_mode="clip", tis_upper_bound=3.0, flagged=1
        )
        resolve_custom_function_configs(args, config_class=_HookLeaf)
        hook = args.custom_tis_function_path
        assert isinstance(hook, CustomFunctionConfig)
        assert dict(hook.config) == {"tis_mode": "clip", "tis_upper_bound": 3.0, "flagged": 1}
        assert not {"tis_mode", "tis_upper_bound", "flagged"} & vars(args).keys()
        view = compute_custom_function_config(
            _Hookless.model_validate({}), hook
        )
        assert mis_hook(view) == "clip"
        json.dumps(hook.model_dump(mode="json"))
