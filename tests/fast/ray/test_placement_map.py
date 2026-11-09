from __future__ import annotations

import json
from argparse import Namespace

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import (
    STANDBY_PG_NAME,
    PlacementGroupInfo,
    PlacementMap,
    RolloutCellDecl,
    create_placement_groups,
    parse_placement_map,
    rollout_cell_binding,
    validate_placement_map,
)


def _args(**overrides) -> Namespace:
    defaults = dict(
        debug_train_only=False,
        debug_rollout_only=False,
        rollout_external=False,
        colocate=False,
        use_critic=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        rollout_num_gpus=3,
        eval_num_gpus=0,
        megatron_config=None,
        deploy_component="all",
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


@pytest.fixture
def requested(monkeypatch) -> list[int]:
    calls: list[int] = []

    def _fake_create(num_gpus):
        calls.append(num_gpus)
        return PlacementGroupInfo(
            pg="pg-sentinel",
            pg_reordered_bundle_indices=[(index * 3 + 1) % num_gpus for index in range(num_gpus)],
            pg_reordered_gpu_ids=[100 + index for index in range(num_gpus)],
        )

    monkeypatch.setattr(placement_group_module, "_create_placement_group", _fake_create)
    return calls


class TestExplicitPlacementMap:
    def test_default_args_carry_no_map_and_keep_the_offset_split(self, requested):
        """The flag defaults to None, so the layout stays trainer-then-rollout with no standby entry."""
        args = _args()
        assert args.yeto_placement_map is None

        pgs = create_placement_groups(args)

        assert requested == [5]
        assert sorted(pgs) == ["actor", "rollout"]
        assert pgs["rollout"].pg_reordered_gpu_ids == [102, 103, 104]

    def test_each_role_gets_exactly_the_bundles_the_map_names(self, requested):
        """Rollout may sit before the trainer and standby bundles are reserved in the same PG."""
        pm = PlacementMap(trainer=(3, 4), rollout=(0, 1, 2), standby=(5,))

        pgs = create_placement_groups(_args(), placement_map=pm)

        assert requested == [6]
        assert {info.pg for info in pgs.values()} == {"pg-sentinel"}
        assert pgs["actor"].pg_reordered_gpu_ids == [103, 104]
        assert pgs["rollout"].pg_reordered_gpu_ids == [100, 101, 102]
        assert pgs[STANDBY_PG_NAME].pg_reordered_gpu_ids == [105]
        full_bundles = [(index * 3 + 1) % 6 for index in range(6)]
        assert pgs["actor"].pg_reordered_bundle_indices == [full_bundles[3], full_bundles[4]]
        assert pgs[STANDBY_PG_NAME].pg_reordered_bundle_indices == [full_bundles[5]]

    def test_the_flag_takes_a_json_object(self, requested):
        args = _args(use_critic=False, yeto_placement_map=json.dumps({"trainer": [1, 0], "rollout": [2, 3, 4]}))

        pgs = create_placement_groups(args)

        assert pgs["actor"].pg_reordered_gpu_ids == [101, 100]
        assert pgs[STANDBY_PG_NAME].pg_reordered_gpu_ids == []

    def test_a_critic_shares_the_mapped_trainer_bundles(self, requested):
        args = _args(use_critic=True, critic_num_nodes=1, critic_num_gpus_per_node=1)

        pgs = create_placement_groups(args, placement_map={"trainer": [4, 3], "rollout": [0, 1, 2]})

        assert pgs["critic"] == pgs["actor"]

    def test_colocate_rejects_the_map(self, requested):
        with pytest.raises(AssertionError, match="--colocate"):
            create_placement_groups(_args(colocate=True), placement_map={"trainer": [0, 1], "rollout": [2, 3, 4]})
        assert requested == []

    @pytest.mark.parametrize(
        "raw, match",
        [
            ({"trainer": [0, 0], "rollout": [1, 2, 3]}, "repeats"),
            ({"trainer": [0, 1], "rollout": [2, 3, 7]}, "outside"),
            ({"trainer": [0, 1], "rollout": [1, 2, 3]}, "outside|share"),
            ({"trainer": [0, 1], "rollout": [2, 3], "standby": [3]}, "share"),
            ({"trainer": [0], "rollout": [1, 2, 3]}, "trainer 1 bundles"),
            ({"trainer": [0, 1], "rollout": [2, 3]}, "rollout 2 bundles"),
        ],
    )
    def test_invalid_maps_are_rejected_before_any_pg_is_created(self, requested, raw, match):
        with pytest.raises(AssertionError, match=match):
            create_placement_groups(_args(), placement_map=raw)
        assert requested == []

    def test_an_overlap_inside_range_is_named_as_an_overlap(self):
        pm = PlacementMap(trainer=(0, 1), rollout=(1, 2), standby=(3,))
        with pytest.raises(AssertionError, match="share bundles \\[1\\]"):
            validate_placement_map(pm, trainer_num_gpus=2, rollout_num_gpus=2)

    def test_unknown_roles_and_non_int_indices_are_rejected(self):
        with pytest.raises(AssertionError, match="unknown roles"):
            parse_placement_map({"trainer": [0], "eval": [1]})
        with pytest.raises(AssertionError, match="list of ints"):
            parse_placement_map({"trainer": ["0"]})
        assert parse_placement_map(None) is None


class TestRolloutCells:
    _MAP = {
        "trainer": [0, 1],
        "rollout": [2, 3, 4],
        "standby": [5],
        "rollout_cells": [
            {"name": "c0", "bundles": [2]},
            {"name": "c1", "bundles": [3]},
            {"name": "c2", "bundles": [4]},
            {"name": "s0", "bundles": [5], "start": False},
            {"name": "t0", "start": False},
        ],
    }

    def test_cells_are_parsed_and_own_no_extra_bundle(self, requested):
        pm = parse_placement_map(self._MAP)
        assert pm.num_bundles == 6
        assert pm.rollout_cells[3] == RolloutCellDecl(name="s0", bundles=(5,), start=False)
        assert pm.rollout_cells[4] == RolloutCellDecl(name="t0", bundles=(), start=False)
        assert [rollout_cell_binding(pm, c) for c in pm.rollout_cells] == [
            ("rollout", 0),
            ("rollout", 1),
            ("rollout", 2),
            ("standby", 0),
            None,
        ]
        create_placement_groups(_args(), placement_map=pm)
        assert requested == [6]

    def test_no_cells_is_the_default(self):
        assert parse_placement_map({"trainer": [0], "rollout": [1]}).rollout_cells == ()

    @pytest.mark.parametrize(
        "cells, match",
        [
            ([{"name": "a", "bundles": [2]}, {"name": "a", "bundles": [3]}], "repeat names"),
            ([{"name": "a", "bundles": [2]}, {"name": "b", "bundles": [2]}], "share bundle 2"),
            ([{"name": "a", "bundles": [0]}], "not in the rollout or standby role"),
            ([{"name": "a", "bundles": [2, 4]}], "not a consecutive run"),
            ([{"name": "a", "start": False}, {"name": "b", "bundles": [2]}], "before the stopped ones"),
            ([{"name": "a"}], "names no bundles"),
        ],
    )
    def test_bad_cells_are_rejected_before_any_pg_is_created(self, requested, cells, match):
        raw = {"trainer": [0, 1], "rollout": [2, 3, 4], "standby": [5], "rollout_cells": cells}
        with pytest.raises(AssertionError, match=match):
            create_placement_groups(_args(), placement_map=raw)
        assert requested == []

    @pytest.mark.parametrize(
        "cells",
        [{"name": "a"}, [{"name": ""}], [{"name": "a", "bundles": ["2"]}], [{"name": "a", "start": 0}], [{"id": "a"}]],
    )
    def test_malformed_cells_are_rejected_when_parsed(self, cells):
        with pytest.raises(AssertionError):
            parse_placement_map({"trainer": [0], "rollout": [1], "rollout_cells": cells})


class TestSlicePgInfo:
    def test_the_public_slicer_keeps_the_given_order(self):
        from miles.ray.placement_group import _slice_pg_info, slice_pg_info

        info = PlacementGroupInfo(pg="pg", pg_reordered_bundle_indices=[7, 8, 9], pg_reordered_gpu_ids=[0, 1, 2])
        assert slice_pg_info(info, [2, 0]) == PlacementGroupInfo("pg", [9, 7], [2, 0])
        assert _slice_pg_info is slice_pg_info
