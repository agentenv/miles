import ast
from pathlib import Path


def test_sglang_engine_normalizes_mutable_and_derived_server_hosts():
    source = (
        Path(__file__).parents[2]
        / "miles/backends/sglang_utils/sglang_engine.py"
    ).read_text()

    assert 'if hasattr(server_args, "derive"):' in source
    assert 'server_args = server_args.derive(' in source
    assert 'host=server_args.host.strip("[]")' in source
    assert 'server_args.host = server_args.host.strip("[]")' in source


def test_broadcast_weight_updates_lazily_import_the_p2p_backend():
    source = (
        Path(__file__).parents[2]
        / "miles/backends/megatron_utils/actor.py"
    ).read_text()
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "update_weight.update_weight_from_distributed.p2p"
        for node in tree.body
    )

    actor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    init = next(
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name == "init"
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "update_weight.update_weight_from_distributed.p2p"
        for node in ast.walk(init)
    )
