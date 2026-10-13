from __future__ import annotations

from types import SimpleNamespace

from miles.backends.megatron_utils import full_parameter_plugin as plugin
from miles.backends.megatron_utils import full_parameter_state as state


def _actor():
    return SimpleNamespace(role="critic", args=SimpleNamespace(start_rollout_id=None))


def _patch(monkeypatch, *, topology="t0", calls=None):
    monkeypatch.setattr(state, "full_parameter_topology", lambda: topology)
    monkeypatch.setattr(
        state, "initialize_full_parameter_tracking", lambda actor, version: setattr(actor, "_external_full_policy_version", version)
    )

    def install(actor, plan, *, role):
        calls.append((plan, role))
        return 2

    monkeypatch.setitem(plugin._ROLE_OPERATIONS, "install_full_parameter_fragment_plan", install)


def test_rank_call_picks_its_own_topology_and_passes_the_actor_role(monkeypatch):
    calls = []
    _patch(monkeypatch, topology="t1", calls=calls)
    actor = _actor()

    result = plugin.full_parameter_rank_call(
        actor,
        operation="install_full_parameter_fragment_plan",
        by_topology={"t0": {"plan": "p0"}, "t1": {"plan": "p1"}},
    )

    assert result == 2
    assert calls == [("p1", "critic")]
    assert actor._external_full_policy_version == 0  # tracking bound from a fresh start


def test_rank_call_returns_failures_instead_of_raising(monkeypatch):
    _patch(monkeypatch, calls=[])
    error = RuntimeError("prepare rejected")

    def finalize(actor, *, commit_token):
        raise error

    monkeypatch.setitem(plugin._PLAIN_OPERATIONS, "finalize_full_parameter_commit", finalize)

    assert plugin.full_parameter_rank_call(
        _actor(), operation="finalize_full_parameter_commit", kwargs={"commit_token": "c"}
    ) is error
