"""Day 1 — fused row-wise softmax in Triton.

Launch scaffolding is PROVIDED and correct; the kernel body is yours.
Flip IMPLEMENTED to True when the kernel is written — tests skip until then.

Reference while you learn (close the tab while typing your version):
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IMPLEMENTED = False  # flip to True once the kernel body is written


@triton.jit
def _softmax_kernel(
    x_ptr, y_ptr,
    x_row_stride, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO(Day 1): one program handles ONE row (grid axis 0 = row index).
    #
    #   1. row = tl.program_id(0); col offsets = tl.arange(0, BLOCK_SIZE);
    #      mask = offsets < n_cols.
    #   2. Load the row in one shot: tl.load(x_ptr + row * x_row_stride +
    #      offsets, mask=mask, other=-float("inf")).
    #      Key idea: masked lanes get -inf so they contribute exp(-inf)=0 —
    #      no branchy edge handling.
    #   3. Numerically stable softmax: subtract tl.max(row_vals, axis=0)
    #      before tl.exp — otherwise fp16-range inputs overflow.
    #   4. Divide by tl.sum, store with the same mask.
    #
    # Why this is fast: eager torch does max/sub/exp/sum/div as separate
    # kernels — 4-5 HBM round-trips. You do ONE read and ONE write; softmax is
    # bandwidth-bound, so bytes moved ~= runtime.
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor. PROVIDED launcher."""
    if not IMPLEMENTED:
        raise NotImplementedError("Day 1: write _softmax_kernel, then set IMPLEMENTED = True")
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
