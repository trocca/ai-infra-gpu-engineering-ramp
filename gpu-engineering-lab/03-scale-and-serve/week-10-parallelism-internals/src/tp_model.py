"""Tensor-parallel GPT: week-05 model with TP MLP + TP attention.

YOUR TASK (Day 2): replace the week-05 GPT's MLP and attention with
tensor-parallel versions so that a 2-rank run produces logits matching
the single-GPU baseline to <= 1e-3 (fp32, same seed).

Attention sharding (Megatron §3.2):
  * Q, K, V projections: one fused ColumnParallelLinear
    (3 * d_model outputs, split across ranks) -> each rank owns
    n_heads / world_size COMPLETE heads. Head dim is never split.
  * Attention math (softmax(QK^T/sqrt(d))V) is fully local per rank —
    heads are independent. No communication here. That's the insight.
  * Output projection: RowParallelLinear (input already split by heads),
    all-reduce (g) combines the partial projections.

So a full transformer layer costs 2 forward all-reduces (attn.g, mlp.g)
and 2 backward all-reduces (attn.f, mlp.f). Verify this count.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from .tp_layers import ColumnParallelLinear, RowParallelLinear, ParallelMLP

# TODO(you): import your week-05 GPT config/blocks instead of copying them.
# from ...02_phase.week_05.src.model import GPTConfig, Block  # noqa: ERA001


class ParallelAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        super().__init__()
        ws = dist.get_world_size()
        assert n_heads % ws == 0, "heads must divide evenly across ranks"
        self.n_local_heads = n_heads // ws
        self.head_dim = d_model // n_heads
        # TODO(you):
        # self.qkv  = ColumnParallelLinear(d_model, 3 * d_model, gather_output=False, seed=...)
        #   CAREFUL: with a fused QKV, the column shard must slice each of
        #   Q, K, V consistently. Easiest correct scheme: three separate
        #   ColumnParallelLinear layers. Fused is a follow-up optimization —
        #   get parity with three first.
        # self.proj = RowParallelLinear(d_model, d_model, input_is_parallel=True, seed=...)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): project -> reshape to [B, n_local_heads, T, head_dim]
        # -> scaled_dot_product_attention(is_causal=True) -> merge heads
        # -> row-parallel output projection.
        raise NotImplementedError


class ParallelBlock(nn.Module):
    """Pre-LN transformer block with TP attention + TP MLP.

    LayerNorm params are small: keep them REPLICATED (identical on all
    ranks — they see identical activations, so their grads match without
    communication). Convince yourself why, write one sentence in the
    README.
    """

    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        super().__init__()
        # TODO(you)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TPGPT(nn.Module):
    """Week-05 GPT with ParallelBlocks. Embeddings + LM head replicated
    (sharding the vocab dim is Megatron's VocabParallelEmbedding — a
    stretch goal, not required for the acceptance criteria)."""

    def __init__(self, config, seed: int = 0):
        super().__init__()
        # TODO(you)
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Verification helper — COMPLETE, use it for the "exactly 2 all-reduces
# per layer" acceptance criterion.
# ---------------------------------------------------------------------------

class AllReduceCounter:
    """Context manager that counts dist.all_reduce calls.

    with AllReduceCounter() as c:
        logits, _ = model(x)
    print(c.count)   # expect 2 * n_layers for TPGPT forward
    """

    def __init__(self):
        self.count = 0
        self._orig = None

    def __enter__(self):
        self._orig = dist.all_reduce

        def counted(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)

        dist.all_reduce = counted
        return self

    def __exit__(self, *exc):
        dist.all_reduce = self._orig
        return False
