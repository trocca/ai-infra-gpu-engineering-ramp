"""Day 4 — fused dequant-matmul: fp16 activations x int8 weights (W8A16).

y = x @ W^T, where W lives in HBM as int8 + per-row fp32 scales and is
dequantized IN REGISTERS tile-by-tile. The dense fp16 W never exists in memory
— that is the entire trick, and why decode gets faster: batch-1 decode is
weight-bandwidth-bound, and int8 halves the weight bytes streamed per token.

Launcher + grid PROVIDED (standard tiled-matmul scaffolding, cf. Triton
tutorial 03); kernel body TODO. Flip IMPLEMENTED when done.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IMPLEMENTED = False  # flip to True once the kernel body is written


@triton.jit
def _matmul_w8a16_kernel(
    x_ptr, w_ptr, scales_ptr, y_ptr,
    M, N, K,                      # x:(M,K) w_int8:(N,K) y:(M,N)
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # TODO(Day 4): standard tiled matmul with one twist.
    #
    #   1. pid_m/pid_n from a 2D program id; offs_m/offs_n/offs_k tiles.
    #   2. acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    #   3. K-loop: load x tile (fp16) and w tile (INT8, note w is (N, K) so
    #      you load it transposed for tl.dot), advance pointers by BLOCK_K.
    #      THE TWIST: convert the int8 tile w/ .to(tl.float16) and tl.dot it
    #      against x. Apply the per-row scale AFTER the K-loop:
    #      acc *= scales[offs_n] broadcast over columns — scales factor out of
    #      the K-sum because they are constant per output channel. (One
    #      multiply per output element instead of one per weight.)
    #   4. Store acc.to(y dtype) with M/N bounds masks.
    pass


def matmul_w8a16(x: torch.Tensor, w_int8: torch.Tensor,
                 scales: torch.Tensor) -> torch.Tensor:
    """PROVIDED launcher. x: (M, K) fp16; w_int8: (N, K) int8; scales: (N,) fp32.
    Returns y = x @ dequant(W)^T, shape (M, N), fp16."""
    if not IMPLEMENTED:
        raise NotImplementedError("Day 4: write _matmul_w8a16_kernel, then set IMPLEMENTED = True")
    assert x.ndim == 2 and w_int8.ndim == 2 and x.shape[1] == w_int8.shape[1]
    assert x.is_cuda and w_int8.dtype == torch.int8
    M, K = x.shape
    N = w_int8.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # 1D grid re-linearized inside the kernel is the tutorial-03 pattern;
    # a plain 2D grid is fine too — keep whichever your kernel body assumes.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_w8a16_kernel[grid](
        x, w_int8, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return y


class W8A16Linear(torch.nn.Module):
    """Drop-in replacement for a frozen nn.Linear using the fused kernel.

    COMPLETE except forward — used on Day 4 to quantize the week-05 model
    (or Qwen2.5-1.5B) layer by layer for the perplexity measurement.
    """

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        from .pack import quantize_int8
        w_int8, scales = quantize_int8(linear.weight.data)
        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scales", scales)
        self.bias = linear.bias
        self.out_features = linear.out_features
        self.in_features = linear.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Day 4): flatten leading dims to (M, K), call matmul_w8a16,
        # add bias if present, restore the original leading shape.
        raise NotImplementedError("Day 4: implement W8A16Linear.forward")
