"""Causal self-attention, written by hand.

Contract (tests/test_attention.py depends on these exact signatures):

    causal_attention(q, k, v) -> Tensor
        q, k, v: (B, n_head, T, d_head) float tensors, same shape.
        Returns:  (B, n_head, T, d_head) — numerically equal to
        F.scaled_dot_product_attention(q, k, v, is_causal=True) within 1e-5 in fp32.

    CausalSelfAttention(n_embd, n_head, max_seq_len)
        forward(x, rope_cache=None, kv_cache=None, layer_idx=0) -> (B, T, n_embd)

No nn.MultiheadAttention, no F.scaled_dot_product_attention in THIS file —
SDPA is the oracle in the tests, not the implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention with a causal mask, from the raw formula.

    TODO(Day 1): implement in four steps:
      1. scores = q @ k^T, scaled by 1/sqrt(d_head).
         Key idea: q·k dot products have variance ~d_head at init; without the
         scale, softmax saturates and gradients vanish.
      2. Build a causal mask (position i may attend to j <= i). Fill the upper
         triangle of scores with -inf BEFORE softmax (torch.triu / masked_fill).
      3. Softmax over the last (key) dimension. Do it in float32 for stability
         if inputs are half precision.
      4. Return weights @ v.
    """
    raise NotImplementedError("Day 1: implement causal_attention")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection.

    Weight layout (tests and the KV-cache decode path rely on it):
      - self.qkv:  nn.Linear(n_embd, 3 * n_embd, bias=False) — q, k, v stacked.
      - self.proj: nn.Linear(n_embd, n_embd, bias=False)
    """

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int = 1024):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.d_head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache=None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """TODO(Day 1 + Day 4): implement in this order.

        Day 1 (training path, kv_cache is None):
          1. Project x -> qkv, split into q, k, v, reshape each to
             (B, n_head, T, d_head)  [hint: .view(B, T, n_head, d_head).transpose(1, 2)].
          2. If rope_cache is given, apply RoPE to q and k (src/rope.py) — NOT to v.
          3. y = causal_attention(q, k, v).
          4. Merge heads back to (B, T, n_embd) and apply self.proj.

        Day 4 (decode path, kv_cache is not None):
          - x is the NEW token(s) only, T == 1 in steady state.
          - Apply RoPE at the correct ABSOLUTE positions (offset = cache length).
          - Append this step's k, v into kv_cache for layer_idx, then attend the
            new q against ALL cached keys/values. No causal mask needed for a
            single query — it can see the whole past by construction.
        """
        raise NotImplementedError("Day 1: implement CausalSelfAttention.forward")
