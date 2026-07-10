"""Day 1 — fused RMSNorm in Triton.

y = x / sqrt(mean(x^2) + eps) * weight    (you wrote the math in week 05 —
now make it ONE kernel: single read of x, single write of y).

Launcher PROVIDED; kernel body TODO. Flip IMPLEMENTED when done.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IMPLEMENTED = False  # flip to True once the kernel body is written


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    x_row_stride, y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO(Day 1): one program per row, same structure as softmax.
    #
    #   1. Load the row (mask lanes >= n_cols with other=0.0 — zeros don't
    #      perturb a sum of squares, unlike the -inf we used for softmax).
    #   2. Upcast to fp32 with .to(tl.float32) BEFORE squaring: fp16 squares
    #      overflow at |x| > ~256 and lose precision summing thousands of terms.
    #   3. rms_inv = 1 / tl.sqrt(tl.sum(x * x) / n_cols + eps)
    #      NOTE: divide by n_cols, not BLOCK_SIZE — the padded lanes are fake.
    #   4. Load weight (same offsets/mask), y = x * rms_inv * w, cast back to
    #      the OUTPUT dtype (y_ptr.dtype.element_ty), store.
    pass


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim of a 2D tensor. PROVIDED launcher."""
    if not IMPLEMENTED:
        raise NotImplementedError("Day 1: write _rmsnorm_kernel, then set IMPLEMENTED = True")
    assert x.ndim == 2 and x.is_cuda and weight.shape == (x.shape[1],)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _rmsnorm_kernel[(n_rows,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 2048 else 4,
    )
    return y


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """COMPLETE fp32-accumulating torch reference (the test oracle)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
