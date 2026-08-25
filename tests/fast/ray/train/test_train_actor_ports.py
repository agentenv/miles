from types import SimpleNamespace

from miles.ray.train_actor import TrainRayActor


def test_configured_train_master_base_port_is_used(monkeypatch):
    observed = {}

    def allocate(*, start_port, consecutive=1):
        observed["start_port"] = start_port
        return "127.0.0.1", start_port

    monkeypatch.setattr("miles.ray.train_actor.configure_logger", lambda *_a, **_k: None)
    monkeypatch.setattr("miles.ray.train_actor.get_local_gpu_id", lambda: 0)
    monkeypatch.setattr(
        TrainRayActor,
        "_get_current_node_ip_and_free_port",
        staticmethod(allocate),
    )

    actor = TrainRayActor(
        SimpleNamespace(train_master_base_port=21002),
        world_size=1,
        rank=0,
        master_addr=None,
        master_port=None,
        indep_dp_store_addr="",
        role="actor",
        cell_index=0,
    )

    assert observed["start_port"] == 21002
    assert actor.master_port == 21002


def test_configured_train_master_ports_separate_actor_critic_and_cells(monkeypatch):
    observed = []

    def allocate(*, start_port, consecutive=1):
        observed.append(start_port)
        return "127.0.0.1", start_port

    monkeypatch.setattr("miles.ray.train_actor.configure_logger", lambda *_a, **_k: None)
    monkeypatch.setattr("miles.ray.train_actor.get_local_gpu_id", lambda: 0)
    monkeypatch.setattr(
        TrainRayActor,
        "_get_current_node_ip_and_free_port",
        staticmethod(allocate),
    )

    common = dict(
        args=SimpleNamespace(train_master_base_port=25000),
        world_size=1,
        rank=0,
        master_addr=None,
        master_port=None,
        indep_dp_store_addr="",
    )
    actor = TrainRayActor(**common, role="actor", cell_index=1)
    critic = TrainRayActor(**common, role="critic", cell_index=1)

    assert observed == [25100, 26100]
    assert actor.master_port == 25100
    assert critic.master_port == 26100
