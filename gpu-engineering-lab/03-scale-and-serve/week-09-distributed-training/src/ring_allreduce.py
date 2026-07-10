"""Ring all-reduce from scratch using torch.distributed P2P ops.

YOUR TASK: implement `ring_allreduce` so that it produces bit-for-bit
(well, float-for-float within 1e-5) the same result as `dist.all_reduce`
with SUM, using only point-to-point communication (isend/irecv).

Algorithm (N ranks, tensor split into N chunks):

  Phase 1 — reduce-scatter (N-1 steps):
      At step s, rank r sends chunk (r - s) % N to rank (r + 1) % N
      and receives chunk (r - s - 1) % N from rank (r - 1) % N,
      adding it into its local copy.
      After N-1 steps, rank r owns the FULLY-REDUCED chunk (r + 1) % N.

  Phase 2 — all-gather (N-1 steps):
      Circulate the reduced chunks around the ring (no adds, just copy)
      until every rank has every reduced chunk.

Gotchas you will hit (that's the point):
  * Deadlock: if every rank calls a blocking send first, nobody recvs.
    Use isend + irecv and wait on both, or alternate send/recv order
    by rank parity.
  * You cannot reuse the buffer you are currently sending — keep a
    separate recv buffer per step.
  * Tensors whose numel is NOT divisible by world_size: pad, or use
    uneven chunk sizes via tensor.chunk / manual offsets. Tests cover this.

Run the provided tests on CPU/gloo before touching a GPU:
    make test
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _chunk_bounds(numel: int, world_size: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] of length world_size covering [0, numel).

    Chunks may be uneven (last chunks smaller, possibly empty) — same
    convention as torch.chunk on a flat tensor.

    TODO(you): implement. Hint: base = ceil(numel / world_size) works for
    torch.chunk-style splitting, but any consistent convention passes the
    tests as long as reduce-scatter and all-gather agree with each other.
    """
    raise NotImplementedError("implement _chunk_bounds")


def ring_allreduce(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """In-place SUM all-reduce over `tensor` using a ring of P2P ops.

    Must work for:
      * any dtype torch.distributed supports for the active backend
      * any shape (operate on tensor.view(-1))
      * numel not divisible by world_size (including numel < world_size)
      * world_size == 1 (no-op)

    Returns `tensor` (reduced in place) for convenience.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor

    flat = tensor.view(-1)
    bounds = _chunk_bounds(flat.numel(), world_size)

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    # ------------------------------------------------------------------
    # TODO(you): Phase 1 — reduce-scatter, N-1 steps.
    #
    # for step in range(world_size - 1):
    #     send_idx = (rank - step) % world_size
    #     recv_idx = (rank - step - 1) % world_size
    #     ... isend flat[bounds[send_idx]] to send_to
    #     ... irecv into a scratch buffer sized like bounds[recv_idx]
    #     ... wait both, then flat[recv_chunk] += scratch
    # ------------------------------------------------------------------
    raise NotImplementedError("implement reduce-scatter phase")

    # ------------------------------------------------------------------
    # TODO(you): Phase 2 — all-gather, N-1 steps.
    #
    # After phase 1, rank r owns reduced chunk (r + 1) % world_size.
    # Circulate: at step s, send chunk (rank + 1 - s) % N, receive
    # chunk (rank - s) % N, COPY (no add) into place.
    # ------------------------------------------------------------------
    raise NotImplementedError("implement all-gather phase")

    # return tensor


def ring_allreduce_mean(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """SUM ring all-reduce followed by division by world_size.

    Used by manual_ddp.py for gradient averaging.
    TODO(you): one-liner once ring_allreduce works.
    """
    raise NotImplementedError
