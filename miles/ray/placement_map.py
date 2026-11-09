"""Explicit role -> bundle placement and caller-declared rollout cells.

A placement map names, by logical bundle position (the (node, gpu) sorted order of the one placement group the
run creates), which bundles the trainer, the rollout engines and a reserved ``standby`` set use. An optional
``rollout_cells`` list declares the engine cells of the single engine pool by the caller's names: each one bound to
a consecutive run of ``rollout`` or ``standby`` bundles, or unbound, and either started at startup or declared
stopped (``start: false``) until the caller starts it.

Pure python and import light: both ``miles.ray.placement_group`` and ``miles.ray.specs.inference`` read it.
"""

import json
from typing import NamedTuple

STANDBY_PG_NAME = "standby"
PLACEMENT_MAP_ROLES = ("trainer", "rollout", "standby")
# Not a role: an optional explicit list of the rollout engine cells, see ``RolloutCellDecl``.
ROLLOUT_CELLS_KEY = "rollout_cells"
# role -> name of the placement group view a cell bound to that role schedules on
_CELL_ROLE_VIEWS = {"rollout": "rollout", STANDBY_PG_NAME: STANDBY_PG_NAME}


class RolloutCellDecl(NamedTuple):
    """One rollout engine cell the caller declares in the placement map.

    ``name`` is the caller's id for the cell. Miles keeps its own cell id (``compute_cell_id(pool, index)`` with
    ``index`` the position in ``rollout_cells``) and reports both (``RayWorkerManager.describe_cells``).
    ``bundles`` are logical bundle positions, a consecutive run of the ``rollout`` or ``standby`` role, or empty for an
    unbound cell. ``start`` False declares the cell stopped: it is not started at startup and takes no GPU, no traffic
    and no weights until the caller starts it (after ``rebind_cell`` if it is unbound).
    """

    name: str
    bundles: tuple[int, ...]
    start: bool = True


class PlacementMap(NamedTuple):
    trainer: tuple[int, ...]
    rollout: tuple[int, ...]
    standby: tuple[int, ...] = ()
    # explicit rollout engine cells; empty keeps the engine pool's own layout over the rollout role
    rollout_cells: tuple[RolloutCellDecl, ...] = ()

    @property
    def num_bundles(self) -> int:
        return len(self.trainer) + len(self.rollout) + len(self.standby)

    def role_positions(self) -> dict[str, tuple[int, ...]]:
        return {"trainer": self.trainer, "rollout": self.rollout, STANDBY_PG_NAME: self.standby}


def parse_placement_map(raw: "str | dict | PlacementMap | None") -> PlacementMap | None:
    if raw is None or isinstance(raw, PlacementMap):
        return raw
    data = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(data, dict), f"a placement map is a JSON object keyed by role, got {data!r}"
    known = [*PLACEMENT_MAP_ROLES, ROLLOUT_CELLS_KEY]
    unknown = sorted(set(data) - set(known))
    assert not unknown, f"placement map names unknown keys {unknown}; known: {known}"
    roles = {}
    for role in PLACEMENT_MAP_ROLES:
        indices = data.get(role, [])
        assert isinstance(indices, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in indices
        ), f"placement map role {role!r} must be a list of ints, got {indices!r}"
        roles[role] = tuple(indices)
    return PlacementMap(**roles, rollout_cells=_parse_rollout_cells(data.get(ROLLOUT_CELLS_KEY, [])))


def _parse_rollout_cells(raw) -> tuple[RolloutCellDecl, ...]:
    assert isinstance(raw, list), f"placement map {ROLLOUT_CELLS_KEY!r} must be a list, got {raw!r}"
    cells = []
    for entry in raw:
        assert isinstance(entry, dict) and not (
            set(entry) - {"name", "bundles", "start"}
        ), f"a rollout cell is {{'name', 'bundles', 'start'}}, got {entry!r}"
        name, bundles, start = entry.get("name"), entry.get("bundles", []), entry.get("start", True)
        assert isinstance(name, str) and name, f"rollout cell name must be a non-empty string, got {entry!r}"
        assert isinstance(bundles, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in bundles
        ), f"rollout cell {name!r} bundles must be a list of ints, got {bundles!r}"
        assert isinstance(start, bool), f"rollout cell {name!r} start must be a bool, got {start!r}"
        cells.append(RolloutCellDecl(name=name, bundles=tuple(bundles), start=start))
    return tuple(cells)


def rollout_cell_binding(pm: PlacementMap, cell: RolloutCellDecl) -> tuple[str, int] | None:
    """``(view name, slot offset)`` of a declared cell's bundles, or None for an unbound cell."""
    if not cell.bundles:
        return None
    positions_of = pm.role_positions()
    for role, view in _CELL_ROLE_VIEWS.items():
        positions = positions_of[role]
        if cell.bundles[0] in positions:
            offset = positions.index(cell.bundles[0])
            assert tuple(positions[offset : offset + len(cell.bundles)]) == cell.bundles, (
                f"rollout cell {cell.name!r} bundles {list(cell.bundles)} are not a consecutive run of the {role!r} "
                f"role {list(positions)}"
            )
            return view, offset
    raise AssertionError(
        f"rollout cell {cell.name!r} bundles {list(cell.bundles)} are not in the rollout or standby role"
    )


def validate_placement_map(pm: PlacementMap, *, trainer_num_gpus: int, rollout_num_gpus: int) -> None:
    """Reject duplicates, out-of-range or overlapping indices, gaps, and role sizes that disagree with the args."""
    n = pm.num_bundles
    positions_of = pm.role_positions()
    for role, indices in positions_of.items():
        dups = sorted({i for i in indices if indices.count(i) > 1})
        assert not dups, f"placement map role {role!r} repeats bundles {dups}"
        out_of_range = sorted(i for i in indices if not 0 <= i < n)
        assert not out_of_range, f"placement map role {role!r} names bundles {out_of_range} outside 0..{n - 1}"
    for a_index, a in enumerate(PLACEMENT_MAP_ROLES):
        for b in PLACEMENT_MAP_ROLES[a_index + 1 :]:
            overlap = sorted(set(positions_of[a]) & set(positions_of[b]))
            assert not overlap, f"placement map roles {a!r} and {b!r} share bundles {overlap}"
    assert (
        len(pm.trainer) == trainer_num_gpus
    ), f"placement map gives the trainer {len(pm.trainer)} bundles but the args ask for {trainer_num_gpus}"
    assert (
        len(pm.rollout) == rollout_num_gpus
    ), f"placement map gives rollout {len(pm.rollout)} bundles but the args ask for {rollout_num_gpus}"
    _validate_rollout_cells(pm)


def _validate_rollout_cells(pm: PlacementMap) -> None:
    cells = pm.rollout_cells
    names = [c.name for c in cells]
    dups = sorted({n for n in names if names.count(n) > 1})
    assert not dups, f"placement map rollout cells repeat names {dups}"
    starts = [c.start for c in cells]
    assert starts == sorted(starts, reverse=True), (
        f"placement map rollout cells must list the cells started at startup before the stopped ones, got "
        f"{[(c.name, c.start) for c in cells]}"
    )
    owner: dict[int, str] = {}
    for cell in cells:
        assert cell.bundles or not cell.start, f"rollout cell {cell.name!r} starts at startup but names no bundles"
        rollout_cell_binding(pm, cell)
        for b in cell.bundles:
            assert b not in owner, f"rollout cells {owner[b]!r} and {cell.name!r} share bundle {b}"
            owner[b] = cell.name
