"""Hide the MoE expert-parallel all-to-alls of one microbatch behind work of the same layer that does not need them.

``MILES_K3_MOE_A2A_OVERLAP=1`` (default off) builds K3's MoE layers as :class:`A2AOverlapMoELayer`. Megatron's
``MoELayer.forward`` still drives the layer (routing, padding mask, logging, latent projections); only three
steps change where they run:

* dispatch: the token all-to-all goes on a side communication stream; while it is in flight the current stream
  dequantizes the frozen MXFP4 experts' fc1 weights (they depend on the weights only, not on the tokens), then
  the probs all-to-all follows on the side stream. ``MILES_K3_MOE_A2A_OVERLAP_PREFETCH=0`` skips the prefetch.
* combine: the combine all-to-all goes on the side stream and the shared experts (``shared_experts_compute``,
  which Megatron runs before the router) run on the current stream while it is in flight.
* backward: autograd runs each node's backward on its forward stream, so the all-to-all gradients also run on
  the side stream. The engine runs ready nodes in decreasing sequence number. The shared experts' nodes are
  created after the combine all-to-all node and before the unpermute that consumes it. Their backward therefore
  runs right after the unpermute's backward has queued the combine-backward all-to-all's input, and overlaps it.

Every kernel sees the same inputs as without the switch, and the dequantization is the same function of the
same bytes. Only the stream and launch order change. One sum would otherwise change: the layer input's
gradient, whose terms arrive in a new order. :class:`_ForkSharedInput` keeps Megatron's order (routed path
first, shared experts last). So the outputs and every gradient are bit-identical.

Launch order is chosen for ``CUDA_DEVICE_MAX_CONNECTIONS=1``, which yeto sets, where every stream shares one
hardware queue. A queued command that waits on unfinished work blocks everything queued behind it. So
nothing that depends on an in-flight all-to-all is queued before the work meant to overlap it: token a2a, then
the dequantization, then the probs a2a (the same reason Megatron's ``token_dispatch`` puts shared fc1 between
its two all-to-alls).

The NCCL collectives on each communicator keep their order, the same on every rank: tokens, probs, combine
on the EP group, and the shared experts' TP collectives after the dispatch instead of before the router. A rank
only waits on another stream for work it queued itself.

Not used (``super()`` paths) under Megatron's own overlaps (``moe_shared_expert_overlap``,
``overlap_moe_expert_parallel_comm``, ``overlap_dispatch_backward_with_experts_wgrad``), MoE-module
recompute, CUDA graphs, the inference dispatchers or any dispatcher other than all-to-all, partial
``fwd_execution_map`` calls, or EP 1. On CPU the reordering still happens and the stream steps are no-ops,
so the gloo tests check the reordered computation.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os

import torch
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

logger = logging.getLogger(__name__)

A2A_OVERLAP_ENV = "MILES_K3_MOE_A2A_OVERLAP"
PREFETCH_ENV = "MILES_K3_MOE_A2A_OVERLAP_PREFETCH"
_FULL_EXECUTION = ["route", "expert_compute", "postprocess"]
_comm_streams: dict[int, torch.cuda.Stream] = {}
_reported = False


def a2a_overlap_enabled() -> bool:
    return os.environ.get(A2A_OVERLAP_ENV, "0") == "1"


def prefetch_enabled() -> bool:
    return os.environ.get(PREFETCH_ENV, "1") == "1"


def _comm_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _comm_streams:
        # high priority: the NCCL CTAs get SMs ahead of the compute kernels they overlap
        _comm_streams[index] = torch.cuda.Stream(device=index, priority=-1)
    return _comm_streams[index]


@contextlib.contextmanager
def _on_comm_stream(*inputs: torch.Tensor, wait: bool = True):
    """Run the body on the side stream. With ``wait`` the stream first waits for the work already queued on
    the current stream (the producers of ``inputs``). ``inputs`` are recorded on the side stream, so their
    memory is not reused while it reads them. Yields the side stream (None off CUDA)."""
    if not inputs[0].is_cuda:
        yield None
        return
    comm = _comm_stream(inputs[0].device)
    if wait:
        comm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(comm):
        yield comm
    for tensor in inputs:
        tensor.record_stream(comm)


def _join(comm: torch.cuda.Stream | None, *outputs: torch.Tensor) -> None:
    """The current stream waits for the side stream. ``outputs``, which the side stream allocated, are
    recorded on the current stream that reads them from here on."""
    if comm is None:
        return
    current = torch.cuda.current_stream()
    current.wait_stream(comm)
    for tensor in outputs:
        tensor.record_stream(current)


def prefetch_frozen_expert_weights(experts) -> bool:
    """Dequantize the frozen MXFP4 experts' fc1 weights now; their grouped linear's next weight fetch
    takes them. Returns whether there was anything to prefetch."""
    from miles_plugins.models.kimi_k3.mxfp4_redequant import RedequantFrozenExperts

    frozen = getattr(experts, "frozen", experts)  # PartiallyTrainableGroupedMLP or a whole-MXFP4 TEGroupedMLP
    fc1 = getattr(frozen, "linear_fc1", None)
    if not isinstance(fc1, RedequantFrozenExperts):
        return False
    fc1.prefetch_weights()
    return True


class _ForkSharedInput(torch.autograd.Function):
    """The MoE input, once for the routed path (router, latent projection, permute) and once for the shared
    experts. Backward: the routed path's gradient (accumulated in its own arrival order) plus the shared
    experts'. That is exactly the sum Megatron's graph forms at the layer input: there the shared experts,
    created before the router, deliver last. Moving them later in creation order would otherwise make them
    deliver first, and the bf16/fp32 sum would round differently."""

    @staticmethod
    def forward(ctx, hidden_states):
        return hidden_states.view_as(hidden_states), hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_routed, grad_shared):
        if grad_routed is None:
            return grad_shared
        if grad_shared is None:
            return grad_routed
        return grad_routed + grad_shared


class A2AOverlapMoELayer(MoELayer):
    """``MoELayer`` whose EP all-to-alls overlap the shared experts and the frozen experts' dequantization."""

    _a2a_overlap = False
    _shared_input: torch.Tensor | None = None
    _deferred_shared_input: torch.Tensor | None = None
    _deferred_shared_output: torch.Tensor | None = None

    def _a2a_overlap_applies(self, intermediate_tensors) -> bool:
        dispatcher = self.token_dispatcher
        config = self.config
        return (
            intermediate_tensors is None
            and isinstance(dispatcher, MoEAlltoAllTokenDispatcher)
            # the reimplemented dispatch below is MoEAlltoAllTokenDispatcher.token_dispatch's body
            and type(dispatcher).token_dispatch is MoEAlltoAllTokenDispatcher.token_dispatch
            and type(dispatcher).token_combine is MoEAlltoAllTokenDispatcher.token_combine
            and dispatcher.ep_size > 1
            and dispatcher.shared_experts is None
            and not dispatcher.use_nccl_stream
            and not dispatcher.drop_and_pad
            and not self.shared_expert_overlap
            and not self.moe_layer_recompute
            and not config.overlap_moe_expert_parallel_comm
            and not config.overlap_dispatch_backward_with_experts_wgrad
            and config.cuda_graph_impl == "none"
            and not hasattr(self, "_inference_token_dispatcher")
            and list(self.fwd_execution_map) == _FULL_EXECUTION
        )

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None, input_ids=None, **kwargs):
        self._a2a_overlap = self._a2a_overlap_applies(intermediate_tensors)
        _report_once(self._a2a_overlap)
        self._shared_input = None
        if self._a2a_overlap and self.use_shared_expert and torch.is_grad_enabled() and hidden_states.requires_grad:
            hidden_states, self._shared_input = _ForkSharedInput.apply(hidden_states)
        try:
            return super().forward(
                hidden_states,
                intermediate_tensors=intermediate_tensors,
                padding_mask=padding_mask,
                input_ids=input_ids,
                **kwargs,
            )
        finally:
            self._a2a_overlap = False
            self._shared_input = None
            self._deferred_shared_input = None
            self._deferred_shared_output = None

    def shared_experts_compute(self, hidden_states):
        if not self._a2a_overlap:
            return super().shared_experts_compute(hidden_states)
        # Megatron calls this before the router; the shared experts run in combine(), under the combine a2a
        self._deferred_shared_input = hidden_states if self._shared_input is None else self._shared_input
        return None

    def dispatch(self, hidden_states, probs):
        if not self._a2a_overlap:
            return super().dispatch(hidden_states, probs)
        dispatcher = self.token_dispatcher
        # MoEAlltoAllTokenDispatcher.token_dispatch with the all-to-alls on the side stream
        dispatcher.tokens_per_expert = dispatcher._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", dispatcher.tokens_per_expert
        )
        with _on_comm_stream(hidden_states, probs) as comm:
            global_tokens = all_to_all(
                dispatcher.ep_group, hidden_states, dispatcher.output_splits, dispatcher.input_splits
            )
        if prefetch_enabled():
            prefetch_frozen_expert_weights(self.experts)
        # probs were ready with the tokens: no second wait, which would queue this behind the dequantization
        with _on_comm_stream(probs, wait=False):
            global_probs = all_to_all(dispatcher.ep_group, probs, dispatcher.output_splits, dispatcher.input_splits)
        _join(comm, global_tokens, global_probs)
        return global_tokens, global_probs

    def combine(self, output):
        if not self._a2a_overlap:
            return super().combine(output)
        with _on_comm_stream(output) as comm:
            combined = super().combine(output)
        shared_input, self._deferred_shared_input = self._deferred_shared_input, None
        if shared_input is not None:
            self._deferred_shared_output = super().shared_experts_compute(shared_input)
        _join(comm, combined)
        return combined

    def postprocess(self, output, shared_expert_output):
        if self._a2a_overlap:
            assert shared_expert_output is None, "A2AOverlapMoELayer defers the shared experts to combine()"
            shared_expert_output, self._deferred_shared_output = self._deferred_shared_output, None
        return super().postprocess(output, shared_expert_output)


def _report_once(active: bool) -> None:
    global _reported
    if _reported:
        return
    _reported = True
    if active:
        print(f"[k3] {A2A_OVERLAP_ENV}=1: MoE EP all-to-alls overlap shared experts / expert dequant", flush=True)
    else:
        logger.warning(
            "%s=1 but this MoE layer's configuration is not supported; Megatron's order is kept", A2A_OVERLAP_ENV
        )


def a2a_overlap_moe_spec(mlp_spec):
    """Swap ``partial(MoELayer, ...)`` (or a ModuleSpec of MoELayer) for :class:`A2AOverlapMoELayer`."""
    if isinstance(mlp_spec, functools.partial):
        assert mlp_spec.func is MoELayer, mlp_spec.func
        return functools.partial(A2AOverlapMoELayer, *mlp_spec.args, **mlp_spec.keywords)
    assert getattr(mlp_spec, "module", None) is MoELayer, mlp_spec  # config-access-exempt: ModuleSpec
    mlp_spec.module = A2AOverlapMoELayer
    return mlp_spec
