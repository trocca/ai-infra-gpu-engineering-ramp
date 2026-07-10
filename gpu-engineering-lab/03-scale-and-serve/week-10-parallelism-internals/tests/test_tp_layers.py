"""COMPLETE test harness — do not modify. Make src/tp_layers.py pass.

CPU/gloo, world_size=2, spawned processes. No GPU needed.
Run:  pytest tests/test_tp_layers.py -v   (or: make test)

The tests exploit the INIT CONTRACT documented in tp_layers.py: layers
initialize the FULL weight from a seeded generator on every rank, then
shard. A single-process reference nn.Linear built from the same seed
must therefore match the parallel result exactly (up to fp32 reduction
noise).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

WORLD_SIZE = 2
D_IN, D_OUT, BATCH = 16, 32, 4


def _full_linear(in_f: int, out_f: int, seed: int, bias: bool = True) -> nn.Linear:
    """Reference layer built exactly like the INIT CONTRACT requires."""
    g = torch.Generator().manual_seed(seed)
    lin = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(out_f, in_f, generator=g) * 0.02)
        if bias:
            lin.bias.copy_(torch.randn(out_f, generator=g) * 0.02)
    return lin


def _spawn(target, name: str, port_salt: int):
    ctx = mp.get_context("spawn")
    errq = ctx.SimpleQueue()

    def wrapped(rank):
        try:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(27000 + port_salt)
            dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
            target(rank)
            errq.put((rank, None))
        except Exception as e:  # noqa: BLE001
            errq.put((rank, f"{type(e).__name__}: {e}"))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    procs = [ctx.Process(target=wrapped, args=(r,)) for r in range(WORLD_SIZE)]
    for p in procs:
        p.start()
    results = [errq.get() for _ in range(WORLD_SIZE)]
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            pytest.fail(f"{name}: worker hung (deadlocked collective?)")
    failures = [f"rank {r}: {err}" for r, err in results if err is not None]
    assert not failures, f"{name}:\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# f / g conjugate ops
# --------------------------------------------------------------------------

def _copy_op_body(rank):
    from src.tp_layers import copy_to_parallel_region

    x = torch.full((5,), float(rank + 1), requires_grad=True)
    y = copy_to_parallel_region(x)
    torch.testing.assert_close(y, x)  # forward identity
    y.sum().backward()
    # backward all-reduces the (all-ones) grad: expect world_size
    torch.testing.assert_close(x.grad, torch.full((5,), float(WORLD_SIZE)))


def test_copy_to_parallel_region():
    _spawn(_copy_op_body, "copy_op", 1)


def _reduce_op_body(rank):
    from src.tp_layers import reduce_from_parallel_region

    x = torch.full((5,), float(rank + 1), requires_grad=True)
    y = reduce_from_parallel_region(x)
    torch.testing.assert_close(y, torch.full((5,), 3.0))  # 1 + 2
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(5))  # backward identity


def test_reduce_from_parallel_region():
    _spawn(_reduce_op_body, "reduce_op", 2)


# --------------------------------------------------------------------------
# ColumnParallelLinear
# --------------------------------------------------------------------------

def _col_fwd_body(rank):
    from src.tp_layers import ColumnParallelLinear

    torch.manual_seed(99)
    x = torch.randn(BATCH, D_IN)
    layer = ColumnParallelLinear(D_IN, D_OUT, gather_output=True, seed=7)
    ref = _full_linear(D_IN, D_OUT, seed=7)
    torch.testing.assert_close(layer(x), ref(x), rtol=1e-5, atol=1e-5)


def test_column_parallel_forward_gathered():
    _spawn(_col_fwd_body, "col_fwd", 3)


def _col_shard_body(rank):
    from src.tp_layers import ColumnParallelLinear

    torch.manual_seed(99)
    x = torch.randn(BATCH, D_IN)
    layer = ColumnParallelLinear(D_IN, D_OUT, gather_output=False, seed=7)
    ref = _full_linear(D_IN, D_OUT, seed=7)
    shard = D_OUT // WORLD_SIZE
    expected = ref(x)[:, rank * shard:(rank + 1) * shard]
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_column_parallel_local_shard():
    _spawn(_col_shard_body, "col_shard", 4)


# --------------------------------------------------------------------------
# RowParallelLinear
# --------------------------------------------------------------------------

def _row_fwd_body(rank):
    from src.tp_layers import RowParallelLinear

    torch.manual_seed(99)
    x_full = torch.randn(BATCH, D_OUT)  # note: row layer input dim = D_OUT here
    shard = D_OUT // WORLD_SIZE
    x_local = x_full[:, rank * shard:(rank + 1) * shard].contiguous()
    layer = RowParallelLinear(D_OUT, D_IN, input_is_parallel=True, seed=11)
    ref = _full_linear(D_OUT, D_IN, seed=11)
    torch.testing.assert_close(layer(x_local), ref(x_full), rtol=1e-5, atol=1e-5)


def test_row_parallel_forward():
    _spawn(_row_fwd_body, "row_fwd", 5)


# --------------------------------------------------------------------------
# Full MLP block: forward + backward parity, and comm count
# --------------------------------------------------------------------------

def _mlp_body(rank):
    import torch.nn.functional as F

    from src.tp_layers import ParallelMLP

    torch.manual_seed(99)
    x = torch.randn(BATCH, D_IN, requires_grad=True)

    mlp = ParallelMLP(D_IN, 4 * D_IN, seed=13)

    # reference: same seeds -> col layer uses seed, row layer uses seed+1
    # (ParallelMLP must follow this convention; it is part of the contract)
    ref_up = _full_linear(D_IN, 4 * D_IN, seed=13)
    ref_down = _full_linear(4 * D_IN, D_IN, seed=14)
    x_ref = x.detach().clone().requires_grad_(True)
    ref_out = ref_down(F.gelu(ref_up(x_ref)))

    # count all-reduces in forward: must be exactly 1 (the g op)
    calls = {"n": 0}
    orig = dist.all_reduce

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    dist.all_reduce = counted
    try:
        out = mlp(x)
    finally:
        dist.all_reduce = orig
    assert calls["n"] == 1, f"expected exactly 1 forward all-reduce, saw {calls['n']}"

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)

    # backward parity on the input gradient (exercises the f op)
    grad_out = torch.randn_like(ref_out)
    dist.broadcast(grad_out, src=0)  # same upstream grad everywhere
    out.backward(grad_out)
    ref_out.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)


def test_parallel_mlp_parity_and_comm_count():
    _spawn(_mlp_body, "mlp", 6)
