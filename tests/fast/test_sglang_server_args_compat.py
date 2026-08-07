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
