"""COMPLETE test harness — do not edit.

Oracle: complex-number rotation (x as interleaved complex pairs, multiplied by
e^{i * pos * theta}). Also checks the properties that make RoPE work: norm
preservation and relative-position invariance.
"""

import pytest
import torch

from src.rope import apply_rope, precompute_rope_cache

torch.manual_seed(0)
HEAD_DIM, MAX_LEN, BASE = 32, 64, 10000.0


def reference_rope(x: torch.Tensor, base: float = BASE, pos_offset: int = 0) -> torch.Tensor:
    """Rotate interleaved channel pairs of x (B, H, T, D) via complex multiply."""
    B, H, T, D = x.shape
    inv_freq = base ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)   # (D/2,)
    pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.float64)
    angles = torch.outer(pos, inv_freq)                                     # (T, D/2)
    xc = torch.view_as_complex(x.double().reshape(B, H, T, D // 2, 2))
    rot = torch.polar(torch.ones_like(angles), angles)                      # e^{i*theta}
    return torch.view_as_real(xc * rot).reshape(B, H, T, D).to(x.dtype)


def get_cache(call):
    return call(precompute_rope_cache, HEAD_DIM, MAX_LEN, BASE)


def test_cache_shapes_and_values(call):
    cos, sin = get_cache(call)
    assert cos.shape == (MAX_LEN, HEAD_DIM // 2)
    assert sin.shape == (MAX_LEN, HEAD_DIM // 2)
    # position 0 => angle 0 everywhere
    torch.testing.assert_close(cos[0], torch.ones(HEAD_DIM // 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(sin[0], torch.zeros(HEAD_DIM // 2), atol=1e-6, rtol=0)
    # spot-check theta_1 at position 1: base ** (-2/D)
    theta_1 = BASE ** (-2.0 / HEAD_DIM)
    torch.testing.assert_close(cos[1, 1].double(), torch.cos(torch.tensor(theta_1).double()),
                               atol=1e-6, rtol=0)


def test_matches_complex_reference(call):
    cos, sin = get_cache(call)
    x = torch.randn(2, 4, 16, HEAD_DIM)
    out = call(apply_rope, x, cos, sin)
    torch.testing.assert_close(out, reference_rope(x), atol=1e-5, rtol=1e-5)


def test_pos_offset_matches_sliced_full_sequence(call):
    """KV-cache decode path: rotating a suffix with pos_offset must equal
    rotating the full sequence and slicing."""
    cos, sin = get_cache(call)
    x = torch.randn(1, 2, 20, HEAD_DIM)
    full = call(apply_rope, x, cos, sin)
    suffix = call(apply_rope, x[:, :, 12:], cos, sin, 12)
    torch.testing.assert_close(suffix, full[:, :, 12:], atol=1e-5, rtol=1e-5)


def test_preserves_norm(call):
    """Rotations are isometries: per-position vector norms must not change."""
    cos, sin = get_cache(call)
    x = torch.randn(2, 2, 16, HEAD_DIM)
    out = call(apply_rope, x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_relative_position_property(call):
    """The RoPE point: <rope(q, m), rope(k, n)> depends only on m - n.
    Shifting both positions by the same amount must not change the score."""
    cos, sin = get_cache(call)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def score(m: int, n: int) -> torch.Tensor:
        qm = call(apply_rope, q, cos, sin, m)
        kn = call(apply_rope, k, cos, sin, n)
        return (qm * kn).sum()

    for shift in (1, 7, 30):
        torch.testing.assert_close(score(5, 2), score(5 + shift, 2 + shift),
                                   atol=1e-4, rtol=1e-4)


def test_position_zero_is_identity(call):
    cos, sin = get_cache(call)
    x = torch.randn(1, 1, 1, HEAD_DIM)
    out = call(apply_rope, x, cos, sin)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
