"""SAO faster value update decision (port of agentenv/miles 16a9bea4 train_critic)."""

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

from miles.backends.training_utils.sao_critic import refresh_values_after_critic_update

ACTOR = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/actor.py"


def _reference(args, rollout_id):
    # upstream: sao_faster_value_update and not offline_value_pretraining and rollout_id >= num_critic_only_steps
    # (this fork has no offline value pretraining)
    return getattr(args, "policy_objective", "ppo") == "sao_dis" and rollout_id >= args.num_critic_only_steps


def test_matches_reference_grid():
    for objective, only, rid in itertools.product(["ppo", "sao_dis", None], [0, 1, 3], range(5)):
        args = SimpleNamespace(num_critic_only_steps=only)
        if objective is not None:
            args.policy_objective = objective
        assert refresh_values_after_critic_update(args, rid) == _reference(args, rid)


def test_default_ppo_never_refreshes():
    assert not refresh_values_after_critic_update(SimpleNamespace(num_critic_only_steps=0), 0)
    assert not refresh_values_after_critic_update(SimpleNamespace(), 7)


def _critic_body():
    tree = ast.parse(ACTOR.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_train_critic":
            return node
    raise AssertionError("_train_critic missing")


def test_refresh_runs_after_updates_and_before_shipping_values():
    body = _critic_body().body
    kinds = []
    for stmt in body:
        src = ast.unparse(stmt)
        if isinstance(stmt, ast.For) and "train(" in src:
            kinds.append("train_loop")
        elif isinstance(stmt, ast.If) and "refresh_values_after_critic_update" in src:
            assert "forward_only" in src and "get_values" in src and "get_data_iterator" in src
            kinds.append("refresh")
        elif "object_store" in src or (isinstance(stmt, ast.If) and "is_pp_last_stage" in src):
            kinds.append("ship")
    assert kinds == ["train_loop", "refresh", "ship"], kinds
