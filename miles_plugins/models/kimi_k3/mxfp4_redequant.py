"""Frozen MXFP4 experts: save the packed bytes, not the dequantized BF16 weights, for the backward.

``MXFP4*GroupedLinear._get_weight_tensors()`` dequantizes every local frozen expert into a fresh BF16 tensor
for each GEMM, and TE's ``_GroupedLinear.forward`` (release_v2.17, ``pytorch/module/grouped_linear.py``) hands
exactly those tensors to ``ctx.save_for_backward`` for the dgrad GEMM: ``cast_if_needed`` returns its input when
the dtype already matches (legacy path ``weights_fp8`` at :574, grouped-tensor path ``weights_for_gemm`` at :204
and :337), and ``prepare_for_saving`` passes plain tensors through (``quantized_tensor.py:136``). The input rows
are not saved (no wgrad: ``weights[0].requires_grad`` is False). So without recompute every MoE layer keeps its
frozen experts in BF16 from the forward until its backward: ``n_frozen_local x 33.0M x 2 B``, 2.95 GiB per layer
per GPU at EP16 with 128 trainable experts (48 frozen local experts), 6.4 GiB at EP8 with 64.

With ``MILES_K31_MXFP4_REDEQUANT_BACKWARD=1`` the module forward runs under
``torch.autograd.graph.saved_tensors_hooks``: a saved tensor that is one of this call's dequantized weights is
packed as a reference to its uint8 parameter (and the parameter's version counter); every other saved tensor
passes through untouched. The unpack hook dequantizes the parameter again. ``dequantize_mxfp4_rows`` is exact and
deterministic, so the backward GEMM reads bitwise the same weights and every gradient is unchanged. The BF16
copy then lives only around the forward GEMM and the backward GEMM of one projection; the cost is one more
dequantization per projection in the backward (elementwise over the 0.53 B/param bytes).

Under ``--recompute-granularity full`` the first forward runs without grad (nothing is saved, the hooks are not
installed) and the backward recompute dequantizes once more, so a recomputed layer dequantizes three times
instead of twice; its peak still drops by the experts the backward is not currently using.

Every weight fetch dequantizes with the fp32 chain of ``dequantize_mxfp4`` (about 77 B of memory traffic per
weight: 122 GB per MoE layer and fetch at EP16, two fetches per layer under full recompute, three with this switch);
``MILES_K31_MXFP4_LUT_DEQUANT=1`` uses ``dequantize_mxfp4_lut`` instead, the same bits through a byte table at
about 9 B per weight.

Default off: the module forward is TE's, unchanged. Torch-only (plus ``miles.utils.mxfp4``) so the CPU tests load
it without Megatron or TE.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator, Sequence

import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows

REDEQUANT_ENV = "MILES_K31_MXFP4_REDEQUANT_BACKWARD"
# dequantize through the byte table (miles.utils.mxfp4.dequantize_mxfp4_lut): the same bits, about 1/8 of the traffic
LUT_DEQUANT_ENV = "MILES_K31_MXFP4_LUT_DEQUANT"

logger = logging.getLogger(__name__)
_reported = False


def redequant_enabled() -> bool:
    return os.environ.get(REDEQUANT_ENV, "0") == "1"


def lut_dequant_enabled() -> bool:
    return os.environ.get(LUT_DEQUANT_ENV, "0") == "1"


def _key(tensor: torch.Tensor) -> tuple:
    # identity of a dequantized weight while its forward runs: a fresh allocation, so (address, geometry) is unique
    return (tensor.device, tensor.dtype, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))


class PackedWeightRef:
    """What the autograd graph holds instead of a dequantized expert: its packed parameter."""

    __slots__ = ("rows", "version", "shape", "dtype", "group_size")

    def __init__(self, rows: torch.Tensor, shape: torch.Size, dtype: torch.dtype, group_size: int):
        self.rows = rows
        self.version = rows._version
        self.shape = shape
        self.dtype = dtype
        self.group_size = group_size

    def dequantize(self) -> torch.Tensor:
        assert self.rows._version == self.version, (
            "a frozen MXFP4 expert's packed bytes changed between its forward and its backward"
        )
        weight = dequantize_mxfp4_rows(self.rows, self.group_size, lut=lut_dequant_enabled())
        assert weight.shape == self.shape and weight.dtype == self.dtype, (weight.shape, self.shape, weight.dtype)
        return weight


class RedequantScope:
    """The dequantized weights of one module forward, and the pack/unpack hooks that swap them for their bytes."""

    def __init__(self, group_size: int = MXFP4_GROUP_SIZE):
        self.group_size = group_size
        self._packed: dict[tuple, torch.Tensor] = {}
        self.registered = 0
        self.packed = 0

    def register(self, weights: Sequence[torch.Tensor], rows: Sequence[torch.Tensor]) -> None:
        """``weights[i]`` was dequantized from ``rows[i]`` for this forward."""
        assert len(weights) == len(rows)
        for weight, packed in zip(weights, rows, strict=True):
            self._packed[_key(weight)] = packed
        self.registered += len(weights)

    def pack(self, tensor: torch.Tensor):
        packed = self._packed.get(_key(tensor)) if isinstance(tensor, torch.Tensor) else None
        if packed is None:
            return tensor
        self.packed += 1
        return PackedWeightRef(packed, tensor.shape, tensor.dtype, self.group_size)

    @staticmethod
    def unpack(saved):
        return saved.dequantize() if isinstance(saved, PackedWeightRef) else saved

    def close(self) -> None:
        # drop the address table: the weights it named are freed after the forward
        self._packed.clear()


@contextlib.contextmanager
def redequant_saved_weights(group_size: int = MXFP4_GROUP_SIZE) -> Iterator[RedequantScope]:
    """Run a forward whose ``RedequantScope.register``-ed weights the graph keeps as packed bytes."""
    scope = RedequantScope(group_size)
    try:
        with torch.autograd.graph.saved_tensors_hooks(scope.pack, scope.unpack):
            yield scope
    finally:
        scope.close()
        _report_once(scope)


def _report_once(scope: RedequantScope) -> None:
    global _reported
    if _reported or scope.registered == 0:
        return
    _reported = True
    if scope.packed == 0:
        logger.warning(
            "%s=1 but the grouped linear saved none of the %d dequantized MXFP4 experts it was given: this TE "
            "keeps other tensors for the backward, so nothing is saved and nothing changes",
            REDEQUANT_ENV,
            scope.registered,
        )
    else:
        logger.info(
            "%s=1: frozen MXFP4 experts are re-dequantized in the backward (%d of %d packed on the first call)",
            REDEQUANT_ENV,
            scope.packed,
            scope.registered,
        )


class RedequantFrozenExperts:
    """Mixin over a grouped linear whose ``weight{i}`` are packed MXFP4 rows: ``_get_weight_tensors`` returns their
    exact BF16 dequantization, and with the switch on the forward saves the packed rows, not the BF16 copies.

    Listed before the TE grouped linear in the bases: the host provides ``num_gemms`` and ``weight0..weightN-1``,
    and the next class in the MRO the forward that calls ``_get_weight_tensors()`` and saves what it returns.
    """

    _redequant_scope: RedequantScope | None = None

    def _packed_rows(self) -> list[torch.Tensor]:
        return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]  # config-access-exempt: TE weight0..N-1

    def _get_weight_tensors(self):
        rows = self._packed_rows()
        lut = lut_dequant_enabled()
        weights = [dequantize_mxfp4_rows(packed, MXFP4_GROUP_SIZE, lut=lut) for packed in rows]
        if self._redequant_scope is not None:
            self._redequant_scope.register(weights, rows)
        return weights

    def forward(self, *args, **kwargs):
        if not redequant_enabled() or not torch.is_grad_enabled():
            return super().forward(*args, **kwargs)
        assert self._redequant_scope is None, "re-entrant forward of one frozen MXFP4 grouped linear"
        with redequant_saved_weights(MXFP4_GROUP_SIZE) as scope:
            self._redequant_scope = scope
            try:
                return super().forward(*args, **kwargs)
            finally:
                self._redequant_scope = None
