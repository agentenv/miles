from pathlib import Path


SGLANG_REPOSITORY = "https://github.com/agentenv/sglang.git"
SGLANG_COMMIT = "b34df47cb760b5210c4861de8d2d3a54ea3695a2"
SGLANG_IMAGE_TAG = "v0.5.16"


def test_cuda_dockerfile_pins_the_yeto_sglang_revision():
    dockerfile = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()

    assert f"ARG SGLANG_IMAGE_TAG={SGLANG_IMAGE_TAG}" in dockerfile
    assert f"ARG SGLANG_REPOSITORY={SGLANG_REPOSITORY}" in dockerfile
    assert f"ARG SGLANG_COMMIT={SGLANG_COMMIT}" in dockerfile
    assert "git fetch --depth 1 ${SGLANG_REPOSITORY} ${SGLANG_COMMIT}" in dockerfile
    assert "git checkout --detach ${SGLANG_COMMIT}" in dockerfile
    assert 'test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}"' in dockerfile
