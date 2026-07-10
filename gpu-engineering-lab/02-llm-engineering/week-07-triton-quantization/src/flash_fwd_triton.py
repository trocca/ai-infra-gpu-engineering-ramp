"""Days 2-3 — FlashAttention FORWARD in Triton (single-head-batched, causal).

The whole point: never materialize the (N x N) score matrix in HBM. Tile over
KV blocks, keep a running max / running denominator / un-normalized output
accumulator per query block, and rescale when the max moves (online softmax).

Grid + launcher PROVIDED; kernel body TODO. Flip IMPLEMENTED when done.

Papers: https://arxiv.org/abs/2205.14135 (v1), https://arxiv.org/abs/2307.08691 (v2)
Online softmax: https://arxiv.org/abs/1805.02867
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IMPLEMENTED = False  # flip to True once the kernel body is written


@triton.jit
def _flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qbh, stride_qm, stride_qd,   # (B*H, N, D) layout strides
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_om, stride_od,
    seq_len, sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,   # query rows per program
    BLOCK_N: tl.constexpr,   # kv rows per inner iteration
    BLOCK_D: tl.constexpr,   # head dim (padded to pow2)
):
    # TODO(Days 2-3). Program layout: axis 0 = query-block index, axis 1 = (b*h).
    #
    # Setup:
    #   start_m = tl.program_id(0) * BLOCK_M;  bh = tl.program_id(1)
    #   offs_m = start_m + tl.arange(0, BLOCK_M)      # query rows
    #   offs_d = tl.arange(0, BLOCK_D)
    #   Load the Q tile ONCE (it stays in registers the whole kernel).
    #
    # State per query row (fp32!):
    #   m_i  = -inf   (running max of scores)
    #   l_i  = 0      (running softmax denominator)
    #   acc  = zeros(BLOCK_M, BLOCK_D)   (UN-normalized output)
    #
    # Loop over KV blocks (hi = seq_len, or for causal: (start_m+BLOCK_M)
    # because later KV blocks are entirely masked — skipping them is where the
    # causal ~2x speedup comes from):
    #   1. Load K tile, compute scores = tl.dot(q, tl.trans(k)) * sm_scale.
    #   2. Masks: causal (offs_m[:, None] >= offs_n[None, :]) and seq-len
    #      bounds -> score = -inf where masked.
    #   3. Online softmax update:
    #        m_new = maximum(m_i, rowmax(scores))
    #        alpha = exp(m_i - m_new)          # rescale factor for old state
    #        p     = exp(scores - m_new)
    #        l_i   = l_i * alpha + rowsum(p)
    #        acc   = acc * alpha + tl.dot(p.to(v.dtype), v_tile)
    #        m_i   = m_new
    #      Key idea: exactness — every previously accumulated term carried a
    #      factor exp(-m_old); multiplying by exp(m_old - m_new) rebases it to
    #      exp(-m_new). Algebra, not approximation.
    #   4. After the loop: acc = acc / l_i[:, None]; store O tile (mask rows
    #      >= seq_len).
    pass


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = True,
    block_m: int = 64, block_n: int = 64,
) -> torch.Tensor:
    """PROVIDED launcher. q, k, v: (B, H, N, D) fp16/bf16 CUDA tensors."""
    if not IMPLEMENTED:
        raise NotImplementedError("Days 2-3: write _flash_fwd_kernel, then set IMPLEMENTED = True")
    B, H, N, D = q.shape
    assert k.shape == v.shape == (B, H, N, D) and q.is_cuda
    sm_scale = D ** -0.5

    qf = q.reshape(B * H, N, D)
    kf = k.reshape(B * H, N, D)
    vf = v.reshape(B * H, N, D)
    o = torch.empty_like(qf)

    grid = (triton.cdiv(N, block_m), B * H)
    _flash_fwd_kernel[grid](
        qf, kf, vf, o,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        vf.stride(0), vf.stride(1), vf.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N, sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4, num_stages=2,
    )
    return o.reshape(B, H, N, D)
