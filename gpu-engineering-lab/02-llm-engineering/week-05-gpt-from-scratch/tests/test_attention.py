"""COMPLETE test harness — do not edit. Oracle: torch SDPA.

Run: pytest tests/test_attention.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from src.attention import CausalSelfAttention, causal_attention

torch.manual_seed(0)


@pytest.mark.parametrize("B,H,T,D", [(1, 1, 8, 16), (2, 4, 64, 32), (2, 12, 128, 64)])
def test_causal_attention_matches_sdpa(call, B, H, T, D):
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = call(causal_attention, q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_causal_attention_is_actually_causal(call):
    """Output at position t must not change when tokens AFTER t change."""
    B, H, T, D = 1, 2, 16, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out1 = call(causal_attention, q, k, v)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, T // 2:], v2[:, :, T // 2:] = torch.randn(2, B, H, T - T // 2, D)
    out2 = call(causal_attention, q, k2, v2)

    torch.testing.assert_close(out1[:, :, : T // 2], out2[:, :, : T // 2],
                               atol=1e-6, rtol=1e-6)


def test_module_output_shape(call):
    m = CausalSelfAttention(n_embd=64, n_head=4, max_seq_len=32)
    x = torch.randn(2, 16, 64)
    y = call(m.forward, x)
    assert y.shape == (2, 16, 64)


def test_module_is_causal(call):
    """Module-level causality: perturbing future tokens leaves the past alone."""
    torch.manual_seed(1)
    m = CausalSelfAttention(n_embd=32, n_head=2, max_seq_len=32).eval()
    x = torch.randn(1, 12, 32)
    y1 = call(m.forward, x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 6, 32)
    y2 = call(m.forward, x2)
    torch.testing.assert_close(y1[:, :6], y2[:, :6], atol=1e-6, rtol=1e-6)


def test_module_matches_sdpa_reference(call):
    """Push the same projections through SDPA and require agreement <= 1e-5."""
    torch.manual_seed(2)
    n_embd, n_head = 64, 4
    m = CausalSelfAttention(n_embd=n_embd, n_head=n_head, max_seq_len=64).eval()
    x = torch.randn(2, 32, n_embd)
    y = call(m.forward, x)

    B, T, _ = x.shape
    d = n_embd // n_head
    q, k, v = m.qkv(x).split(n_embd, dim=2)
    q = q.view(B, T, n_head, d).transpose(1, 2)
    k = k.view(B, T, n_head, d).transpose(1, 2)
    v = v.view(B, T, n_head, d).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = m.proj(ref.transpose(1, 2).contiguous().view(B, T, n_embd))
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
