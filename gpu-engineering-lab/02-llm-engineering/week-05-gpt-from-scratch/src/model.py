"""Decoder-only GPT, assembled from hand-written parts.

Architecture (modern-decoder recipe, ~Llama-style at toy scale):
  token embedding -> N x Block(pre-RMSNorm attention, pre-RMSNorm SwiGLU MLP)
  -> final RMSNorm -> LM head (weight-TIED to the token embedding).

Contract (tests + train/generate depend on):
    GPTConfig dataclass fields as below.
    GPT(config).forward(idx, targets=None, kv_cache=None)
        -> (logits, loss)  where loss is None if targets is None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .rope import precompute_rope_cache


@dataclass
class GPTConfig:
    vocab_size: int = 50257      # tiktoken gpt2
    max_seq_len: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384            # d12: ~30M params with vocab 50257
    rope_base: float = 10000.0
    dropout: float = 0.0


# Ready-made configs — pick per VRAM/time budget.
CONFIGS = {
    "d6":  GPTConfig(n_layer=6,  n_head=6,  n_embd=192),   # ~10M, smoke tests
    "d12": GPTConfig(n_layer=12, n_head=12, n_embd=384),   # ~30M, the week's main run
    "d16": GPTConfig(n_layer=16, n_head=16, n_embd=512),   # ~50M, if training is fast
}


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no mean-centering, no bias).

    y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO(Day 2): implement the formula above.

        Key idea vs LayerNorm: skip subtracting the mean — re-scaling invariance
        is what matters for training stability, re-centering is (empirically)
        not. One reduction instead of two. Compute mean(x^2) in float32 when x
        is bf16, then cast back.
        """
        raise NotImplementedError("Day 2: implement RMSNorm.forward")


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) ).

    Hidden size uses the 2/3 trick: hidden = int(8/3 * n_embd) rounded to a
    multiple of 64, so a gated MLP has ~the same params as a 4x GELU MLP.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO(Day 2): one line. Key idea: the gate lets the network modulate
        (multiplicatively) which hidden features pass — a data-dependent filter
        a plain GELU MLP can't express as cheaply."""
        raise NotImplementedError("Day 2: implement SwiGLUMLP.forward")


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.max_seq_len)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config.n_embd)

    def forward(self, x, rope_cache=None, kv_cache=None, layer_idx: int = 0):
        """TODO(Day 2): two residual sub-layers. Pre-norm (norm INSIDE the
        residual branch) keeps the residual stream an identity path — that is
        why deep pre-norm nets train without warmup gymnastics. Pass rope_cache
        / kv_cache / layer_idx through to the attention."""
        raise NotImplementedError("Day 2: implement Block.forward")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # TODO(Day 2): WEIGHT TYING — make lm_head.weight and tok_emb.weight the
        # same Parameter object (assign one to the other). Then apply init:
        #   - normal(0, 0.02) for linears/embeddings,
        #   - scale residual-output projections (attn.proj, mlp.down) by
        #     1/sqrt(2 * n_layer) so the residual stream variance stays O(1)
        #     with depth (GPT-2 trick).
        # Register the RoPE cache as a buffer here (persistent=False) so it
        # moves with .to(device):
        #   cos, sin = precompute_rope_cache(config.n_embd // config.n_head,
        #                                    config.max_seq_len, config.rope_base)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """TODO(Day 2 + Day 4).

        Day 2 (training):
          embed idx -> dropout -> each block (with rope_cache) -> final norm
          -> lm_head -> logits (B, T, vocab).
          If targets is not None: cross_entropy(logits.view(-1, V),
          targets.view(-1), ignore_index=-1). Return (logits, loss).

        Day 4 (decode): if kv_cache is not None, idx is only the new token(s);
        thread kv_cache and layer_idx into each block, and only compute the
        lm_head on the LAST position (that's all sampling needs — saves a
        (T, vocab) matmul).

        Sanity check you MUST do before training: loss at init ~= ln(vocab_size).
        """
        raise NotImplementedError("Day 2: implement GPT.forward")

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; excludes tied embedding double-count by construction."""
        n = sum(p.numel() for p in self.parameters())
        return n
