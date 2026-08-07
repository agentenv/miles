from pathlib import Path


SGLANG_REPOSITORY = "https://github.com/agentenv/sglang.git"
SGLANG_COMMIT = "f187485a79b98a1f2c3aade37ce72c2e1f3e89f5"


def test_cuda_dockerfile_pins_the_yeto_sglang_revision():
    dockerfile = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()

    assert f"ARG SGLANG_REPOSITORY={SGLANG_REPOSITORY}" in dockerfile
    assert f"ARG SGLANG_COMMIT={SGLANG_COMMIT}" in dockerfile
    assert "git fetch --depth 1 ${SGLANG_REPOSITORY} ${SGLANG_COMMIT}" in dockerfile
    assert "git checkout --detach ${SGLANG_COMMIT}" in dockerfile
    assert 'test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}"' in dockerfile
