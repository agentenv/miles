"""Full-parameter chunks travel through Ray as one out-of-band numpy buffer."""

from __future__ import annotations

import pickle

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as state


def _round_trip(value: torch.Tensor) -> tuple[object, list[pickle.PickleBuffer]]:
    buffers: list[pickle.PickleBuffer] = []
    payload = pickle.dumps(state._ObjectStoreChunk(value), protocol=5, buffer_callback=buffers.append)
    return pickle.loads(payload, buffers=buffers), buffers


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_chunk_pickles_as_one_out_of_band_buffer_and_restores_the_tensor(dtype):
    value = torch.arange(1 << 12).to(dtype)
    restored, buffers = _round_trip(value)
    assert len(buffers) == 1 and buffers[0].raw().nbytes == value.numel() * value.element_size()
    assert isinstance(restored, torch.Tensor)
    assert restored.dtype == dtype and restored.ndim == 1 and restored.is_contiguous()
    assert torch.equal(restored, value)


def test_restored_chunk_passes_the_reader_checks():
    value = torch.linspace(-1, 1, 4096, dtype=torch.float32)
    restored, _ = _round_trip(value)
    descriptor = state.FullParameterChunkDescriptor(
        chunk_index=0,
        flat_offset=0,
        numel=value.numel(),
        payload_hash=state._chunk_hash(0, 0, 0, value, parameter_layout_hash="a" * 64, plan_hash="b" * 64),
    )
    checked = state._validate_cpu_chunk(
        restored, descriptor, fragment_id=0, verify_values=True, parameter_layout_hash="a" * 64, plan_hash="b" * 64
    )
    assert torch.equal(checked, value)


class _Recorder:
    def __init__(self, name: str) -> None:
        self.__name__ = name
        self.stored: list[object] = []

    def put(self, value):
        self.stored.append(value)
        return len(self.stored) - 1


def test_only_the_real_ray_module_gets_the_zero_copy_form(monkeypatch):
    value = torch.ones(8)
    ray_like, fake = _Recorder("ray"), _Recorder("fake_ray")
    state.put_full_parameter_chunk(ray_like, value)
    state.put_full_parameter_chunk(fake, value)
    assert isinstance(ray_like.stored[0], state._ObjectStoreChunk)
    assert fake.stored[0] is value
    monkeypatch.setenv("MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS", "0")
    state.put_full_parameter_chunk(ray_like, value)
    assert ray_like.stored[1] is value


def test_transport_advertises_zero_copy_chunks():
    transport = pytest.importorskip("miles.ray.full_parameter_transport")
    assert "zero_copy_chunks" in transport.FULL_PARAMETER_TRANSPORT_FEATURES
