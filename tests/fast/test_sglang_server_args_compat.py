from pathlib import Path


def test_sglang_engine_derives_the_normalized_server_host():
    source = (
        Path(__file__).parents[2]
        / "miles/backends/sglang_utils/sglang_engine.py"
    ).read_text()

    assert 'server_args = server_args.derive(' in source
    assert 'host=server_args.host.strip("[]")' in source
    assert 'server_args.host = server_args.host.strip("[]")' not in source
