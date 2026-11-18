"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn


try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(
        module, "cp_group", None
    )  # config-access-exempt: context parallel groups are attached only to participating model modules
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _zigzag_maps(cu_seqlens, cp_size, total):
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding). Returns ``(zigzag_position, packed_position)``:
    ``zigzag_position[p]`` is the row of packed position ``p`` in the rank-concatenated zigzag shards,
    and ``packed_position`` is its inverse."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    zigzag_position = owner * shard + start // cp_size + offset % chunk + mirrored * chunk
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = position
    return zigzag_position, packed_position


def _relayout_indices(cu_seqlens, cp_rank, cp_size, total):
    """``(to_packed, to_zigzag)``: ``to_packed`` selects this rank's contiguous shard from the
    rank-concatenated zigzag shards, ``to_zigzag`` selects this rank's zigzag shard from the
    rank-concatenated contiguous shards (the packed stream itself)."""
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, total)
    rows = slice(cp_rank * (total // cp_size), (cp_rank + 1) * (total // cp_size))
    return zigzag_position[rows], packed_position[rows]


def zigzag_gather_layout(cu_seqlens, cp_rank, cp_size, local_len):
    """For attention over the whole packed stream from a rank's zigzag shard of ``local_len`` rows:
    ``(gather_order, positions)``. ``gather_order`` puts the rank-concatenated zigzag shards (a CP
    all-gather) in packed order; ``positions`` are this rank's rows' packed positions."""
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, local_len * cp_size)
    return zigzag_position, packed_position[cp_rank * local_len : (cp_rank + 1) * local_len]


def _gather_select(x, index, cp_group):
    gathered = x.new_empty((x.shape[0] * dist.get_world_size(group=cp_group), *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=cp_group)
    return gathered.index_select(0, index)


class _Relayout(torch.autograd.Function):
    """All-gather over CP, then select this rank's rows; the gradient is the same with the inverse map."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group):
        ctx.cp_group = cp_group
        ctx.save_for_backward(inverse_index)
        return _gather_select(x, index, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)


def tp_slice_all_gather(x: torch.Tensor, cp_group, tp_group) -> torch.Tensor:
    """``all_gather_into_tensor`` of ``x`` over ``cp_group`` (rank-concatenated rows) for an ``x`` identical on
    every rank of ``tp_group``: each TP rank moves only its 1 / TP slice over CP, the TP ranks all-gather the
    slices. Same bytes out; 1 / TP of the cross-CP traffic. No autograd: the caller owns the backward."""
    cp_size = dist.get_world_size(group=cp_group)
    tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
    rows, rest = x.shape[0], x.shape[1:]
    assert rows % tp_size == 0, (rows, tp_size)
    part = rows // tp_size
    slices = x.new_empty((cp_size * part, *rest))  # [cp, part, ...]
    dist.all_gather_into_tensor(slices, x[tp_rank * part : (tp_rank + 1) * part].contiguous(), group=cp_group)
    full = x.new_empty((tp_size * cp_size * part, *rest))  # [tp, cp, part, ...]
    dist.all_gather_into_tensor(full, slices, group=tp_group)
    return full.view(tp_size, cp_size, part, *rest).transpose(0, 1).reshape(cp_size * rows, *rest)


class _TPReplicatedRelayout(torch.autograd.Function):
    """``_Relayout`` of a tensor that is identical on every TP rank (the sequence-parallel-gathered layer input).

    ``_Relayout`` all-gathers the whole [rows, ...] tensor over CP on every TP rank: TP copies of the same bytes
    over the (cross-node) CP link. Here TP rank t sends only its 1 / TP slice of rows over CP, then the TP ranks
    all-gather the peers' slices over the (intra-node) TP link. Pure data movement, so the output is bit-identical.
    The gradient is TP-distinct (the consumers are column-parallel), so the backward is ``_Relayout``'s."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group, tp_group):
        ctx.cp_group, ctx.tp_group = cp_group, tp_group
        ctx.save_for_backward(inverse_index)
        cp_size, cp_rank = dist.get_world_size(group=cp_group), dist.get_rank(group=cp_group)
        tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
        rows, rest = x.shape[0], x.shape[1:]
        part = rows // tp_size
        # [cp, part, ...]: slice tp_rank of every CP rank's rows
        slices = x.new_empty((cp_size * part, *rest))
        dist.all_gather_into_tensor(slices, x[tp_rank * part : (tp_rank + 1) * part].contiguous(), group=cp_group)
        # the peers' slices only: this CP rank's own rows are x itself
        peers = slices.view(cp_size, part, *rest)[[c for c in range(cp_size) if c != cp_rank]].reshape(-1, *rest)
        remote = x.new_empty((tp_size * (cp_size - 1) * part, *rest))  # [tp, cp - 1, part, ...]
        dist.all_gather_into_tensor(remote, peers, group=tp_group)
        # row g of the rank-concatenated zigzag shards: CP rank g // rows, TP slice (g % rows) // part
        owner, local = index // rows, index % rows
        peer = owner - (owner > cp_rank).to(owner.dtype)
        remote_row = rows + ((local // part) * (cp_size - 1) + peer) * part + local % part
        return torch.cat((x, remote)).index_select(0, torch.where(owner == cp_rank, local, remote_row))

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        if _relayout_bwd_reduce_on():
            grad = _tp_reduced_gather_select(grad_output, inverse_index, ctx.cp_group, ctx.tp_group)
            return grad, None, None, None, None
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None, None


RELAYOUT_BWD_REDUCE_ENV = "MILES_K31_KDA_RELAYOUT_BWD_REDUCE"


def _relayout_bwd_reduce_on() -> bool:
    """MILES_K31_KDA_RELAYOUT_BWD_REDUCE=1 (default off; set it the same on every rank, the backward's collectives
    differ): ``_TPReplicatedRelayout``'s backward sums the gradient over TP before it crosses CP."""
    return os.environ.get(RELAYOUT_BWD_REDUCE_ENV, "0") == "1"


def _reduce_scatter_rows(x: torch.Tensor, group) -> torch.Tensor:
    """Sum ``x`` over ``group`` and keep this rank's 1 / size block of rows (gloo, the CPU tests' backend, has no
    reduce-scatter: all-reduce and slice there)."""
    size, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
    part = x.shape[0] // size
    if dist.get_backend(group) == "gloo":
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x[rank * part : (rank + 1) * part].contiguous()
    out = x.new_empty((part, *x.shape[1:]))
    dist.reduce_scatter_tensor(out, x, group=group)
    return out


def _tp_reduced_gather_select(grad, inverse_index, cp_group, tp_group):
    """The gradient of ``_TPReplicatedRelayout`` with 1 / TP of the cross-CP bytes.

    ``_gather_select`` all-gathers each TP rank's whole TP-distinct partial gradient over CP (the n1 trace: 19.8 ms
    per KDA layer and microbatch for [131072, 7168] bf16, against 2.6-3.2 ms for the forward's TP-slice gather).
    The layer input is the sequence-parallel gather (``tensor_parallel_output_grad=True``), whose backward sums its
    gradient over TP, so any split of the true gradient over the TP ranks is as good as the partials. Here the TP
    ranks first reduce-scatter the packed-layout partials (rank t gets packed rows [t * part, (t + 1) * part) summed,
    over the intra-node link), the CP ranks all-gather those sums (1 / TP of today's cross-CP bytes), and rank t
    returns, for every zigzag row, the summed gradient if its packed row is in slice t, else zero: the TP sum of the
    returned tensors is the summed gradient, exactly once per row. The values differ from the default only in the
    order of the TP sum (bf16 reduce-scatter in packed instead of zigzag row blocks), not in what is summed."""
    cp_size = dist.get_world_size(group=cp_group)
    tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
    rows, rest = grad.shape[0], grad.shape[1:]
    part = rows // tp_size
    summed = _reduce_scatter_rows(grad.contiguous(), tp_group)  # [part, ...] packed rows of slice tp_rank
    slices = grad.new_empty((cp_size * part, *rest))  # [cp, part, ...]
    dist.all_gather_into_tensor(slices, summed, group=cp_group)
    # zigzag row i takes packed row inverse_index[i] of the rank-concatenated packed shards
    owner, local = inverse_index // rows, inverse_index % rows
    mine = local // part == tp_rank
    src = torch.where(mine, owner * part + local - tp_rank * part, torch.zeros_like(local))
    picked = slices.index_select(0, src)
    return torch.where(mine.view(-1, *([1] * len(rest))), picked, torch.zeros_like(picked))


def zigzag_to_packed_shard_tp_replicated(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, tp_group):
    """``zigzag_to_packed_shard`` of a tensor identical on every rank of ``tp_group``: the same result and
    gradient, with 1 / TP of its cross-CP forward bytes. Rows must split evenly over TP (they do under SP)."""
    assert hidden_states.size(0) % tp_group.size() == 0, (hidden_states.size(0), tp_group.size())
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _TPReplicatedRelayout.apply(hidden_states, to_packed, to_zigzag, cp_group, tp_group)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)


# ---------------------------------------------------------------------------------------------------------------
# MILES_K31_KDA_CP_RELAYOUT_A2A: the same relayouts by an all-to-all of only the rows each rank needs.
#
# A relayout is a permutation of the rank-concatenated rows: every source row lands on exactly one rank. The
# all-gather above moves every rank's whole shard to every other rank and then keeps the rows it needs. At
# CP2 that is half of the peer's shard (a rank keeps half its own zigzag rows in its contiguous shard), so
# half the bytes are thrown away, and more at higher CP. Sending only the needed rows moves the same values
# into the same places. It is a pure permutation, so it is bit-identical, forward and backward (the backward
# is the inverse permutation).
#
# The all-to-all split sizes must be known on the host. They depend only on ``cu_seqlens`` and are computed
# once per microbatch (one small DtoH copy) and cached with the index tensors. Every KDA layer, the recompute
# and the backward reuse them: ``cu_seqlens`` is the same tensor object for all of a microbatch's layers.
# ---------------------------------------------------------------------------------------------------------------

CP_RELAYOUT_A2A_ENV = "MILES_K31_KDA_CP_RELAYOUT_A2A"
_PLAN_CACHE: list = []
_PLAN_CACHE_SIZE = 8


def cp_relayout_a2a_on() -> bool:
    return os.environ.get(CP_RELAYOUT_A2A_ENV, "0") == "1"


class _CPPermutation:
    """One rank's side of a permutation of the rank-concatenated rows, ``out_global[g] = in_global[src_of[g]]``.

    ``own_pos`` / ``own_src``: output rows taken from this rank's own input rows. ``recv_pos``: the output rows
    filled by the all-to-all, grouped by source rank (ascending), each group in output order. ``send_idx``: this
    rank's input rows the others need, grouped by destination rank, each group in the destination's order.
    ``counts[r][s]``: rows rank ``r`` takes from rank ``s`` (host ints, the same table on every rank)."""

    def __init__(self, src_of: torch.Tensor, cp_rank: int, cp_size: int, counts: list[list[int]]):
        shard = src_of.numel() // cp_size
        self.cp_rank, self.cp_size, self.shard, self.counts = cp_rank, cp_size, shard, counts
        mine = src_of[cp_rank * shard : (cp_rank + 1) * shard]
        owner = mine // shard
        self.own_pos = torch.nonzero(owner == cp_rank).flatten()
        self.own_src = mine[self.own_pos] - cp_rank * shard
        others = [s for s in range(cp_size) if s != cp_rank]
        self.recv_pos = torch.cat([torch.nonzero(owner == s).flatten() for s in others])
        sends = []
        for r in others:
            theirs = src_of[r * shard : (r + 1) * shard]
            sends.append(theirs[theirs // shard == cp_rank] - cp_rank * shard)
        self.send_idx = torch.cat(sends)
        self.recv_splits = [0 if s == cp_rank else counts[cp_rank][s] for s in range(cp_size)]
        self.send_splits = [0 if r == cp_rank else counts[r][cp_rank] for r in range(cp_size)]

    def _place(self, own: torch.Tensor, received: torch.Tensor) -> torch.Tensor:
        out = own.new_empty((self.shard, *own.shape[1:]))
        out.index_copy_(0, self.own_pos, own)
        out.index_copy_(0, self.recv_pos, received)
        return out

    def forward(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        received = x.new_empty((sum(self.recv_splits), *x.shape[1:]))
        dist.all_to_all_single(
            received,
            x.index_select(0, self.send_idx),
            output_split_sizes=self.recv_splits,
            input_split_sizes=self.send_splits,
            group=cp_group,
        )
        return self._place(x.index_select(0, self.own_src), received)

    def backward(self, grad: torch.Tensor, cp_group) -> torch.Tensor:
        returned = grad.new_empty((sum(self.send_splits), *grad.shape[1:]))
        dist.all_to_all_single(
            returned,
            grad.index_select(0, self.recv_pos),
            output_split_sizes=self.send_splits,
            input_split_sizes=self.recv_splits,
            group=cp_group,
        )
        grad_x = grad.new_empty((self.shard, *grad.shape[1:]))
        grad_x.index_copy_(0, self.own_src, grad.index_select(0, self.own_pos))
        grad_x.index_copy_(0, self.send_idx, returned)
        return grad_x

    def forward_tp_replicated(self, x: torch.Tensor, cp_group, tp_group) -> torch.Tensor:
        """``forward`` of an ``x`` identical on every rank of ``tp_group``: TP rank ``t`` carries the ``t``-th
        ``1 / TP`` of every rank pair's rows over CP, then the TP ranks all-gather what they received."""
        tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)

        def piece(n: int, t: int) -> tuple[int, int]:  # rows [lo, hi) of n that TP rank t carries
            k = -(-n // tp_size)
            return min(n, t * k), min(n, (t + 1) * k)

        send_parts, send_splits, recv_splits = [], [], []
        offset = 0
        for r in range(self.cp_size):
            n = 0 if r == self.cp_rank else self.counts[r][self.cp_rank]
            lo, hi = piece(n, tp_rank)
            send_parts.append(self.send_idx[offset + lo : offset + hi])
            send_splits.append(hi - lo)
            offset += n
        widths = []  # per source rank: the padded rows every TP rank holds for it
        for s in range(self.cp_size):
            m = 0 if s == self.cp_rank else self.counts[self.cp_rank][s]
            lo, hi = piece(m, tp_rank)
            recv_splits.append(hi - lo)
            widths.append(-(-m // tp_size))
        received = x.new_empty((sum(recv_splits), *x.shape[1:]))
        dist.all_to_all_single(
            received,
            x.index_select(0, torch.cat(send_parts)),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=cp_group,
        )
        # pad each source's piece to its width so the TP all-gather is uniform: [tp, sum(widths), ...]
        padded = x.new_zeros((sum(widths), *x.shape[1:]))
        index, start, pos = [], 0, 0
        for s in range(self.cp_size):
            padded[start : start + recv_splits[s]] = received[pos : pos + recv_splits[s]]
            start += widths[s]
            pos += recv_splits[s]
        gathered = x.new_empty((tp_size * sum(widths), *x.shape[1:]))
        dist.all_gather_into_tensor(gathered, padded, group=tp_group)
        # back to the plain all-to-all's received order: source by source, TP piece by TP piece
        start = 0
        for s in range(self.cp_size):
            m = 0 if s == self.cp_rank else self.counts[self.cp_rank][s]
            for t in range(tp_size):
                lo, hi = piece(m, t)
                base = t * sum(widths) + start
                index.append(torch.arange(base, base + hi - lo, device=x.device))
            start += widths[s]
        full = gathered.index_select(0, torch.cat(index))
        return self._place(x.index_select(0, self.own_src), full)


def _relayout_plans(cu_seqlens: torch.Tensor, cp_rank: int, cp_size: int, total: int):
    """``(to_packed, to_zigzag)`` :class:`_CPPermutation` of one microbatch, cached on its ``cu_seqlens``."""
    for entry in _PLAN_CACHE:
        cu, version, key, plans = entry
        if cu is cu_seqlens and version == cu_seqlens._version and key == (cp_rank, cp_size, total):
            return plans
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, total)
    shard = total // cp_size
    dest = torch.arange(total, device=zigzag_position.device) // shard
    # one DtoH for both directions: counts[r][s] of rank r's rows taken from rank s
    both = torch.stack(
        [
            torch.bincount(dest * cp_size + src // shard, minlength=cp_size * cp_size)
            for src in (zigzag_position, packed_position)
        ]
    ).tolist()
    counts = [[row[r * cp_size : (r + 1) * cp_size] for r in range(cp_size)] for row in both]
    plans = (
        _CPPermutation(zigzag_position, cp_rank, cp_size, counts[0]),
        _CPPermutation(packed_position, cp_rank, cp_size, counts[1]),
    )
    _PLAN_CACHE.insert(0, (cu_seqlens, cu_seqlens._version, (cp_rank, cp_size, total), plans))
    del _PLAN_CACHE[_PLAN_CACHE_SIZE:]
    return plans


class _A2ARelayout(torch.autograd.Function):
    """``_Relayout`` / ``_TPReplicatedRelayout`` by all-to-all; the gradient is the inverse permutation."""

    @staticmethod
    def forward(ctx, x, plan, cp_group, tp_group):
        ctx.plan, ctx.cp_group = plan, cp_group
        if tp_group is not None:
            return plan.forward_tp_replicated(x, cp_group, tp_group)
        return plan.forward(x, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.plan.backward(grad_output.contiguous(), ctx.cp_group), None, None, None


def zigzag_to_packed_shard_a2a(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, tp_group=None):
    """``zigzag_to_packed_shard`` (``tp_group`` given: ``zigzag_to_packed_shard_tp_replicated``) by
    all-to-all. Same result and gradient, bit for bit."""
    to_packed, _ = _relayout_plans(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _A2ARelayout.apply(hidden_states, to_packed, cp_group, tp_group)


def packed_shard_to_zigzag_a2a(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """``packed_shard_to_zigzag`` by all-to-all. Same result and gradient, bit for bit."""
    _, to_zigzag = _relayout_plans(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _A2ARelayout.apply(hidden_states, to_zigzag, cp_group, None)
