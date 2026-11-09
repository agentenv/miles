"""--partial-experts-selection: schema validation and the config trait."""

import copy

import pytest
from pydantic import ValidationError
from tests.fast.miles_plugins.models.kimi_k3.partial_experts_fixtures import make_selection_json, write_selection

from miles.utils.args.configs.partial_experts import (
    ExpertSelection,
    ExpertSelectionError,
    PartialExpertsConfig,
    load_expert_selection,
)

E, K, EP, LAYERS = 64, 16, 4, [1, 2, 3]


def _valid():
    return make_selection_json(num_experts=E, num_trainable=K, ep_size=EP, layers=LAYERS)


def test_valid_selection_loads_and_round_trips(tmp_path):
    obj = _valid()
    selection = load_expert_selection(write_selection(tmp_path, obj))
    assert selection.num_local_experts == E // EP and selection.num_local_trainable == K // EP
    assert selection.moe_layers == LAYERS
    assert selection.layers_sha256() == obj["sha256"]
    assert ExpertSelection.from_json(selection.to_json()) == selection


def test_sha256_is_optional_and_ignores_source():
    obj = _valid()
    del obj["sha256"]
    obj["source"] = {"method": "other", "tokens": 12345}
    assert ExpertSelection.from_json(obj).layers_sha256() == _valid()["sha256"]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda o: o.update(schema=2), "schema"),
        (lambda o: o["layers"]["2"]["trainable"].pop(), "trainable ids, expected"),
        (lambda o: o["layers"]["2"].update(trainable=o["layers"]["2"]["trainable"][::-1]), "ascending"),
        (lambda o: o["layers"]["3"]["slot_order"].__setitem__(0, o["layers"]["3"]["slot_order"][1]), "permutation"),
        (lambda o: o.update(ep_size=32), "does not divide num_trainable"),
        (lambda o: o.update(ep_size=3), "does not divide num_experts"),
        (lambda o: o.update(sha256="0" * 64), "sha256"),
        (lambda o: o.update(num_trainable=E), "below num_experts"),
    ],
)
def test_invalid_selection_is_rejected(mutate, message):
    obj = _valid()
    mutate(obj)
    if "sha256" in obj and message != "sha256":
        del obj["sha256"]  # the mutations above change the layers; the sha check is tested on its own
    with pytest.raises(ExpertSelectionError, match=message):
        ExpertSelection.from_json(obj)


def test_unequal_trainable_slots_per_rank_is_rejected():
    """Swapping a trainable slot of rank 0 with a frozen slot of rank 1 keeps slot_order a permutation and the
    trainable set intact, but rank 0 then holds k_local - 1 trainable slots and rank 1 k_local + 1."""
    obj = _valid()
    del obj["sha256"]
    order = obj["layers"]["1"]["slot_order"]
    n_local = E // EP
    order[0], order[n_local + n_local - 1] = order[n_local + n_local - 1], order[0]
    with pytest.raises(ExpertSelectionError, match="EP rank 0 must hold exactly its first"):
        ExpertSelection.from_json(obj)


def test_trainable_slot_not_at_front_is_rejected():
    obj = _valid()
    del obj["sha256"]
    order = obj["layers"]["1"]["slot_order"]
    k_local = K // EP
    order[0], order[k_local] = order[k_local], order[0]  # a frozen expert in a trainable slot of rank 0
    with pytest.raises(ExpertSelectionError, match="EP rank 0"):
        ExpertSelection.from_json(obj)


def test_load_reports_the_path(tmp_path):
    obj = _valid()
    obj["layers"]["1"]["slot_order"][0] = obj["layers"]["1"]["slot_order"][1]
    path = write_selection(tmp_path, obj)
    with pytest.raises(ExpertSelectionError, match="expert_selection.json"):
        load_expert_selection(path)


def test_config_trait_defaults_and_dtype_choice():
    config = PartialExpertsConfig.model_validate({})
    assert config.partial_experts_selection is None and config.partial_experts_trainable_dtype == "bf16"
    with pytest.raises(ValidationError, match="partial-experts-trainable-dtype"):
        PartialExpertsConfig.model_validate({"partial_experts_trainable_dtype": "fp16"})


def test_trainer_config_carries_the_trait(tmp_path):
    pytest.importorskip("sglang")
    from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

    from miles.utils.args.trainer_utils import compute_trainer_config

    path = write_selection(tmp_path, copy.deepcopy(_valid()))
    config = parse_megatron_test_config("--partial-experts-selection", path)
    trainer = compute_trainer_config(all_config=config, trainer=config.raw_megatron.trainers[0])
    assert trainer.partial_experts_selection == path
    assert trainer.partial_experts_trainable_dtype == "bf16"
