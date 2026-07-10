"""Megatron-style tensor-parallel linear layers, from scratch.

YOUR TASK: implement the f/g conjugate ops and the two parallel linear
layers so tests/test_tp_layers.py passes on CPU/gloo with world_size=2.

Key identity to keep in your head (Megatron-LM §3):

  MLP:  Y = B @ gelu(A @ X)
        A column-split  -> each rank holds A_i, computes gelu(A_i X)  (no comm)
        B row-split     -> each rank computes B_i h_i, SUM via all-reduce (g)
        backward through the input needs an all-reduce too (f)

  f = _CopyToParallelRegion:    fwd identity, bwd all-reduce
  g = _ReduceFromParallelRegion: fwd all-reduce, bwd identity
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


# ---------------------------------------------------------------------------
# The conjugate communication ops
# ---------------------------------------------------------------------------

class _CopyToParallelRegion(torch.autograd.Function):
    """'f' in the Megatron paper.

    forward:  identity (input is replicated on all ranks)
    backward: all-reduce(SUM) of the gradient

    WHY: the same replicated activation feeds a *different* weight shard on
    each rank, so each rank computes only a PARTIAL d(loss)/d(input); the
    true gradient is the sum over ranks.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # TODO(you)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO(you) — careful: all-reduce mutates in place; clone first
        # (autograd may hand you a grad buffer it still owns).
        raise NotImplementedError


class _ReduceFromParallelRegion(torch.autograd.Function):
    """'g' in the Megatron paper.

    forward:  all-reduce(SUM) — combine partial outputs of a row-parallel matmul
    backward: identity — the reduced output is replicated, so each rank
              already receives the full gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # TODO(you)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO(you)
        raise NotImplementedError


def copy_to_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _CopyToParallelRegion.apply(x)


def reduce_from_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _ReduceFromParallelRegion.apply(x)


# ---------------------------------------------------------------------------
# The parallel layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear with weight split along the OUTPUT dimension.

    Full layer: out = X @ W.T + b, W is [out_features, in_features].
    Rank i holds rows [i*out/ws : (i+1)*out/ws] of W (and of b).

    forward: y_i = f(X) @ W_i.T + b_i        -> shape [..., out/ws]
    If gather_output: all-gather the y_i into [..., out] (autograd-aware —
    hint: you may write a third autograd Function, or keep gather_output
    False everywhere like Megatron does inside blocks).

    INIT CONTRACT (the tests rely on this): generate the FULL weight with
    torch.Generator().manual_seed(seed), identically on every rank, then
    take your shard. Same for bias. This makes TP bit-comparable to a
    single-process reference layer built with the same seed.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = True, gather_output: bool = False, seed: int = 0):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert out_features % self.world_size == 0, "keep it simple: divisible only"
        # TODO(you): full-init-then-shard (see INIT CONTRACT above).
        # self.weight = nn.Parameter(...)   # [out/ws, in]
        # self.bias   = nn.Parameter(...)   # [out/ws] or None
        self.gather_output = gather_output
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): f, then local matmul (F.linear), then optional gather.
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """Linear with weight split along the INPUT dimension.

    Rank i holds columns [i*in/ws : (i+1)*in/ws] of W. The input is
    expected ALREADY SPLIT (input_is_parallel=True — the normal case when
    it follows a ColumnParallelLinear with gather_output=False).

    forward: partial_i = x_i @ W_i.T ; y = g(partial_i) + b
    Bias trap: add the bias AFTER the all-reduce, on the full result —
    otherwise you add it world_size times. (Alternatively only rank 0
    holds it; pick one, be consistent, document it.)
    """

    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = True, input_is_parallel: bool = True, seed: int = 0):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert in_features % self.world_size == 0
        # TODO(you): full-init-then-shard along dim=1.
        self.input_is_parallel = input_is_parallel
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): (split x if not input_is_parallel), local matmul,
        # g (all-reduce), then bias.
        raise NotImplementedError


class ParallelMLP(nn.Module):
    """The Megatron MLP block: ColumnParallel -> gelu -> RowParallel.

    Exactly ONE all-reduce in forward (inside g) and ONE in backward
    (inside f). The tests count them.
    """

    def __init__(self, d_model: int, d_hidden: int, seed: int = 0):
        super().__init__()
        # TODO(you): compose the two layers (gather_output=False,
        # input_is_parallel=True).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
