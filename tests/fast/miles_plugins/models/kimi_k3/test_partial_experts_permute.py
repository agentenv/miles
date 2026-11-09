"""The slot permutation keeps a latent MoE layer's function bitwise; R3 ids and the token split map back exactly."""

import pytest
import torch
from tests.fast.miles_plugins.models.kimi_k3.partial_experts_fixtures import make_selection, make_selection_json, write_selection

from miles_plugins.models.kimi_k3.partial_experts import map_replay_ids_to_slots
from miles_plugins.models.kimi_k3.permute import (
    SlotLayout,
    get_active_slot_layout,
    load_slot_layout,
    set_active_slot_layout,
)

E, K, TOPK, H, LATENT, FFN = 64, 16, 8, 32, 24, 16
LAYER = 3


@pytest.fixture(autouse=True)
def _no_active_layout():
    set_active_slot_layout(None)
    yield
    set_active_slot_layout(None)


def _latent_moe(params, x, *, replay_ids=None):
    """K3-style layer: sigmoid router + expert bias for selection, renormalized top-k scores,
    latent down-proj -> per-expert gated MLP -> latent up-proj, plus a shared expert."""
    logits = x @ params["router"].T
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + params["bias"], TOPK, dim=-1).indices if replay_ids is None else replay_ids
    probs = scores.gather(-1, ids)
    probs = probs / probs.sum(-1, keepdim=True)
    lat = x @ params["down"].T
    lat = lat / lat.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    out = torch.zeros_like(lat)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            gate, up = (lat[t] @ params["w13"][e].T).chunk(2)
            out[t] += probs[t, j] * ((torch.nn.functional.silu(gate) * up) @ params["w2"][e].T)
    return out @ params["up"].T + x @ params["shared"].T, ids


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    return {
        "router": r(E, H),
        "bias": r(E) * 0.1,
        "down": r(LATENT, H) / H**0.5,
        "up": r(H, LATENT) / LATENT**0.5,
        "w13": r(E, 2 * FFN, LATENT) / LATENT**0.5,
        "w2": r(E, LATENT, FFN) / FFN**0.5,
        "shared": r(H, H) / H**0.5,
    }


def _layout(ep_size):
    return SlotLayout.from_selection(make_selection(num_experts=E, num_trainable=K, ep_size=ep_size, layers=[LAYER]))


def _to_slots(params, layout):
    perm = layout.perm[LAYER]
    return {
        **params,
        "router": layout.permute_router_rows(LAYER, params["router"]),
        "bias": layout.permute_router_rows(LAYER, params["bias"]),
        "w13": params["w13"][perm],
        "w2": params["w2"][perm],
    }


@pytest.mark.parametrize("ep_size", [1, 2, 4, 8, 16])
def test_every_rank_front_loads_the_same_number_of_trainable_slots(ep_size):
    layout = _layout(ep_size)
    trainable = set(layout.trainable_original_ids(LAYER, 0))
    for rank in range(1, ep_size):
        trainable |= set(layout.trainable_original_ids(LAYER, rank))
    n_local, k_local = E // ep_size, K // ep_size
    assert len(trainable) == K
    for rank in range(ep_size):
        flags = [layout.local_slot_to_original_id(LAYER, rank, j) in trainable for j in range(n_local)]
        assert flags == [True] * k_local + [False] * (n_local - k_local)
        for j in range(n_local):
            original = layout.local_slot_to_original_id(LAYER, rank, j)
            assert layout.original_id_to_rank_slot(LAYER, original) == (rank, j)
    assert torch.equal(layout.perm[LAYER][layout.inv[LAYER]], torch.arange(E))
    assert [layout.trainable_index(s) for s in range(E) if layout.is_trainable_slot(s)] == list(range(K))


@pytest.mark.parametrize("ep_size", [1, 4, 16])
def test_permuted_layer_is_bitwise_equivalent(ep_size):
    layout, params = _layout(ep_size), _params()
    x = torch.randn(40, H, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    ref, ref_ids = _latent_moe(params, x)
    out, slot_ids = _latent_moe(_to_slots(params, layout), x)
    # same experts in the same top-k order, so the weighted sum runs in the same order: bitwise equal
    assert torch.equal(layout.slot_to_original_ids(LAYER, slot_ids), ref_ids)
    assert torch.equal(out, ref)


def test_r3_replay_of_original_ids_matches_the_unpermuted_model():
    layout, params = _layout(4), _params()
    x = torch.randn(24, H, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    ref, engine_ids = _latent_moe(params, x)  # the inference engine routes the original model
    replayed, _ = _latent_moe(_to_slots(params, layout), x, replay_ids=map_replay_ids_to_slots(layout, LAYER, engine_ids))
    assert torch.equal(replayed, ref)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_r3_mapping_round_trips_and_keeps_padding(dtype):
    layout = _layout(8)
    ids = torch.randint(0, E, (5, 7, TOPK), dtype=dtype)
    ids[0, 0] = -1  # fill_replay_data pads the final token with -1 on every stream
    slots = layout.original_to_slot_ids(LAYER, ids)
    assert slots.dtype == dtype and slots.shape == ids.shape
    assert torch.equal(slots[0, 0], ids[0, 0])
    assert torch.equal(layout.slot_to_original_ids(LAYER, slots), ids)
    valid = ids[1:]
    assert torch.equal(layout.inv[LAYER].to(dtype)[valid.long()], slots[1:])
    assert map_replay_ids_to_slots(None, LAYER, ids) is ids


def test_prefix_split_grouped_gemm_matches_the_unsplit_one():
    """What PartiallyTrainableGroupedMLP does: tokens arrive sorted by local slot, the first
    sum(tokens_per_expert[:k]) rows belong to the trainable half, including when a trainable slot has none."""
    g = torch.Generator().manual_seed(4)
    n_local, k_local = 8, 2
    tokens_per_expert = torch.tensor([3, 0, 5, 1, 0, 2, 4, 1])
    w = torch.randn(n_local, FFN, LATENT, generator=g, dtype=torch.float64)
    x = torch.randn(int(tokens_per_expert.sum()), LATENT, generator=g, dtype=torch.float64)

    def grouped(x, w, counts):
        return torch.cat([xe @ we.T for xe, we in zip(x.split(counts.tolist()), w, strict=True)]) if len(counts) else x[:0]

    ref = grouped(x, w, tokens_per_expert)
    hot = int(tokens_per_expert[:k_local].sum())
    x_hot, x_cold = x.split([hot, x.shape[0] - hot])
    out = torch.cat([grouped(x_hot, w[:k_local], tokens_per_expert[:k_local]), grouped(x_cold, w[k_local:], tokens_per_expert[k_local:])])
    assert torch.equal(out, ref)


def test_load_slot_layout_installs_the_active_layout_and_checks_ep(tmp_path):
    path = write_selection(tmp_path, make_selection_json(num_experts=E, num_trainable=K, ep_size=4, layers=[LAYER]))
    assert get_active_slot_layout() is None
    layout = load_slot_layout(path, num_experts=E, ep_size=4)
    assert get_active_slot_layout() is layout and layout.ep_size == 4
    with pytest.raises(AssertionError, match="ep_size"):
        load_slot_layout(path, num_experts=E, ep_size=8)
    with pytest.raises(AssertionError, match="experts"):
        load_slot_layout(path, num_experts=E * 2, ep_size=4)


def test_each_layer_has_its_own_trainable_set_and_tables():
    """92 layers x 128 experts: the selection is per layer; nothing is shared between layers."""
    layout = SlotLayout.from_selection(make_selection(num_experts=E, num_trainable=K, ep_size=4, layers=[1, 2, 3], seed=7))
    sets = {layer: {e for r in range(4) for e in layout.trainable_original_ids(layer, r)} for layer in (1, 2, 3)}
    assert all(len(s) == K for s in sets.values())
    assert sets[1] != sets[2] and sets[2] != sets[3]
    ids = torch.arange(E).view(1, E)
    for layer in (1, 2, 3):
        slots = layout.original_to_slot_ids(layer, ids)
        assert torch.equal(layout.perm[layer][slots[0]], ids[0])  # this layer's own inverse
        # a trainable id of layer 1 only lands in a trainable slot of layer 1 if layer 1 trains it
        for e in sets[1]:
            assert layout.is_trainable_slot(int(layout.inv[layer][e])) == (e in sets[layer])
