"""Rotary Position Embedding (RoPE) — Su et al., https://arxiv.org/abs/2104.09864

Contract (tests/test_rope.py depends on these exact signatures):

    precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=None,
                          dtype=torch.float32) -> (cos, sin)
        cos, sin: each (max_seq_len, head_dim // 2).
        cos[t, i] = cos(t * theta_i), theta_i = base ** (-2i / head_dim).

    apply_rope(x, cos, sin, pos_offset=0) -> Tensor
        x: (B, n_head, T, head_dim). Rotates each consecutive pair
        (x[..., 2i], x[..., 2i+1]) by angle (pos_offset + t) * theta_i.
        pos_offset is used by KV-cache decode, where the "first" token of x
        actually sits at absolute position `pos_offset`.

Pairing convention: INTERLEAVED pairs (2i, 2i+1) — the original RoFormer layout.
(Llama/HF use the split-halves layout instead; both are valid, but the tests
pin the interleaved one. Consistency between q and k is all that matters.)
"""

from __future__ import annotations

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO(Day 2): build the cos/sin tables.

    Key idea: each pair of channels i gets its own rotation frequency
    theta_i = base ** (-2i / head_dim) — low i spins fast (fine positions),
    high i spins slowly (coarse positions), like clock hands.

    Steps:
      1. inv_freq: (head_dim//2,) tensor of theta_i.
      2. Outer product with positions t = 0..max_seq_len-1 -> angles (T, head_dim//2).
      3. Return angles.cos(), angles.sin() cast to `dtype`.
    """
    raise NotImplementedError("Day 2: implement precompute_rope_cache")


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_offset: int = 0,
) -> torch.Tensor:
    """TODO(Day 2): rotate pairs of channels.

    For each pair (a, b) = (x[..., 2i], x[..., 2i+1]) at sequence index t:
        a' = a * cos - b * sin
        b' = a * sin + b * cos
    with cos/sin taken at row (pos_offset + t).

    Key idea: this is multiplication by a 2x2 rotation matrix, i.e. multiplying
    the complex number (a + ib) by e^{i*theta}. In the attention dot product the
    rotations of q (angle m*theta) and k (angle n*theta) compose to (m-n)*theta —
    absolute rotations, RELATIVE position sensitivity.

    Implementation hints: slice even/odd channels (x[..., 0::2], x[..., 1::2]),
    compute the rotated pair, re-interleave (stack + flatten last two dims).
    Do the math in float32, cast back to x.dtype at the end.
    """
    raise NotImplementedError("Day 2: implement apply_rope")
