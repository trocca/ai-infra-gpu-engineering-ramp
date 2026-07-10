"""COMPLETE test harness — do not edit.

Oracle: full-recompute generation. A KV cache is a pure optimization — greedy
decode WITH the cache must be token-for-token identical to greedy decode
WITHOUT it. Runs on CPU with a tiny random-weight model (no training needed).
"""

import pytest
import torch

from src.generate import KVCache, generate, sample_next_token
from src.model import GPT, GPTConfig

torch.manual_seed(0)

TINY = GPTConfig(vocab_size=256, max_seq_len=64, n_layer=2, n_head=2, n_embd=32)


def make_model(call):
    model = call(GPT, TINY)  # skips if model init itself is unimplemented
    model.eval()
    return model


# ---------------------------------------------------------------- KVCache unit

def test_kv_cache_update_shapes_and_cursor(call):
    cache = call(KVCache, 2, 1, 2, 8, 4, "cpu", torch.float32)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    k_all, v_all = call(cache.update, 0, k, v)
    assert k_all.shape == (1, 2, 3, 4) and v_all.shape == (1, 2, 3, 4)
    assert cache.seq_len == 0, "cursor must only advance after the LAST layer"

    call(cache.update, 1, k, v)
    assert cache.seq_len == 3

    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    k_all, v_all = call(cache.update, 0, k2, v2)
    assert k_all.shape == (1, 2, 4, 4)
    torch.testing.assert_close(k_all[:, :, :3], k, atol=0, rtol=0)
    torch.testing.assert_close(k_all[:, :, 3:], k2, atol=0, rtol=0)


# ------------------------------------------------------------------- sampling

def test_greedy_sampling_is_argmax(call):
    logits = torch.randn(3, 256)
    tok = call(sample_next_token, logits, 0.0)
    assert tok.shape == (3, 1)
    torch.testing.assert_close(tok.squeeze(1), logits.argmax(dim=-1), atol=0, rtol=0)


def test_top_k_restricts_support(call):
    torch.manual_seed(3)
    logits = torch.randn(1, 256)
    allowed = set(torch.topk(logits, k=5, dim=-1).indices.flatten().tolist())
    for _ in range(50):
        tok = call(sample_next_token, logits, 1.0, 5, None)
        assert tok.item() in allowed


def test_top_p_keeps_at_least_one_token(call):
    logits = torch.full((1, 256), -10.0)
    logits[0, 42] = 10.0  # near-degenerate distribution
    for _ in range(10):
        tok = call(sample_next_token, logits, 1.0, None, 0.01)
        assert tok.item() == 42


# ------------------------------------------- cache vs full-recompute (oracle)

@pytest.mark.parametrize("prompt_len,new_tokens", [(4, 8), (10, 20)])
def test_cached_greedy_equals_full_recompute(call, prompt_len, new_tokens):
    model = make_model(call)
    idx = torch.randint(0, TINY.vocab_size, (1, prompt_len))
    ref = call(generate, model, idx.clone(), new_tokens, 0.0, None, None, False)
    out = call(generate, model, idx.clone(), new_tokens, 0.0, None, None, True)
    assert ref.shape == out.shape == (1, prompt_len + new_tokens)
    assert torch.equal(ref, out), (
        f"KV-cache decode diverged from full recompute:\nref={ref.tolist()}\nout={out.tolist()}"
    )


def test_cached_greedy_equals_full_recompute_batched(call):
    model = make_model(call)
    idx = torch.randint(0, TINY.vocab_size, (3, 6))
    ref = call(generate, model, idx.clone(), 10, 0.0, None, None, False)
    out = call(generate, model, idx.clone(), 10, 0.0, None, None, True)
    assert torch.equal(ref, out)


def test_generate_prepends_prompt_unchanged(call):
    model = make_model(call)
    idx = torch.randint(0, TINY.vocab_size, (1, 5))
    out = call(generate, model, idx.clone(), 3, 0.0, None, None, True)
    assert torch.equal(out[:, :5], idx)
