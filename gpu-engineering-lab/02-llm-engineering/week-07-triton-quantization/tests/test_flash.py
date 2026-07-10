"""COMPLETE test harness — do not edit. Oracle: torch SDPA."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="Triton required (ships with torch cu128 under WSL2)")

from conftest import requires_gpu
from src.flash_fwd_triton import flash_attention_forward

torch.manual_seed(0)


def sdpa_ref(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                          is_causal=causal).to(q.dtype)


@requires_gpu
@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 128, 64),
    (2, 4, 256, 64),
    (1, 8, 1024, 64),
    (1, 2, 512, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(call, B, H, N, D, causal):
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    out = call(flash_attention_forward, q, k, v, causal)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, causal), atol=1e-2, rtol=0)


@requires_gpu
def test_seq_len_not_multiple_of_block(call):
    """N=300 with 64-blocks exercises the bounds masks on both M and N."""
    q, k, v = (torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16)
               for _ in range(3))
    out = call(flash_attention_forward, q, k, v, True)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, True), atol=1e-2, rtol=0)


@requires_gpu
def test_causality(call):
    """Row t of the output must ignore KV rows > t."""
    B, H, N, D = 1, 1, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    out1 = call(flash_attention_forward, q, k, v, True)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, N // 2:] = torch.randn_like(k2[:, :, N // 2:])
    v2[:, :, N // 2:] = torch.randn_like(v2[:, :, N // 2:])
    out2 = call(flash_attention_forward, q, k2, v2, True)
    torch.testing.assert_close(out1[:, :, : N // 2], out2[:, :, : N // 2],
                               atol=1e-3, rtol=0)


@requires_gpu
def test_online_softmax_is_stable(call):
    """Large-magnitude scores overflow a naive exp; online softmax must not."""
    q = (torch.randn(1, 1, 256, 64, device="cuda") * 8).to(torch.float16)
    k = (torch.randn(1, 1, 256, 64, device="cuda") * 8).to(torch.float16)
    v = torch.randn(1, 1, 256, 64, device="cuda", dtype=torch.float16)
    out = call(flash_attention_forward, q, k, v, True)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, sdpa_ref(q, k, v, True), atol=2e-2, rtol=0)


@requires_gpu
def test_bf16(call):
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    out = call(flash_attention_forward, q, k, v, True)
    torch.testing.assert_close(out, sdpa_ref(q, k, v, True), atol=2e-2, rtol=0)
