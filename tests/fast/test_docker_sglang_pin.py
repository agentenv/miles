from pathlib import Path


SGLANG_REPOSITORY = "https://github.com/agentenv/sglang.git"
SGLANG_COMMIT = "b34df47444271ebda0673d68fe000399804c181b"
SGLANG_IMAGE_TAG = "v0.5.16"
SGLANG_TVM_FFI = "apache-tvm-ffi==0.1.11"


def test_cuda_dockerfile_pins_the_yeto_sglang_revision():
    dockerfile = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()

    assert f"ARG SGLANG_IMAGE_TAG={SGLANG_IMAGE_TAG}" in dockerfile
    assert f"ARG SGLANG_REPOSITORY={SGLANG_REPOSITORY}" in dockerfile
    assert f"ARG SGLANG_COMMIT={SGLANG_COMMIT}" in dockerfile
    assert "git fetch --depth 1 ${SGLANG_REPOSITORY} ${SGLANG_COMMIT}" in dockerfile
    assert "git checkout --detach ${SGLANG_COMMIT}" in dockerfile
    assert 'test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}"' in dockerfile


def test_cuda_dockerfile_restores_sglang_tvm_ffi_after_flashqla():
    dockerfile = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()

    flashqla_install = 'RUN pip install -v --no-build-isolation "git+https://github.com/QwenLM/FlashQLA.git"'
    tvm_ffi_install = f'RUN pip install --force-reinstall --no-deps "{SGLANG_TVM_FFI}"'

    assert flashqla_install in dockerfile
    assert tvm_ffi_install in dockerfile
    assert dockerfile.index(flashqla_install) < dockerfile.index(tvm_ffi_install)
    assert "map_dataclass_to_tuple" in dockerfile
