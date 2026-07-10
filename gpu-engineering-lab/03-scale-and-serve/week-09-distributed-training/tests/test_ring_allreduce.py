"""COMPLETE test harness — do not modify. Make src/ring_allreduce.py pass.

Runs on CPU with the gloo backend and 2 spawned processes, so it works on
the local Windows/WSL laptop and in CI with zero GPUs.

Run:  pytest tests/test_ring_allreduce.py -v   (or: make test)
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2


def _init(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, port: int, shape, dtype_name: str, seed: int, errq) -> None:
    try:
        from src.ring_allreduce import ring_allreduce

        _init(rank, world_size, port)
        dtype = getattr(torch, dtype_name)

        g = torch.Generator().manual_seed(seed + rank)
        if dtype.is_floating_point:
            t = torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
        else:
            t = torch.randint(-100, 100, shape, generator=g, dtype=dtype)

        expected = t.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        got = ring_allreduce(t)

        if dtype.is_floating_point:
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(got, expected)
        # must be in-place: returned tensor is the input tensor
        assert got.data_ptr() == t.data_ptr(), "ring_allreduce must operate in place"
        errq.put((rank, None))
    except Exception as e:  # noqa: BLE001 — surface any failure to the parent
        errq.put((rank, f"{type(e).__name__}: {e}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_case(shape, dtype_name: str = "float32", seed: int = 0) -> None:
    ctx = mp.get_context("spawn")
    errq = ctx.SimpleQueue()
    port = 29500 + (abs(hash((tuple(shape), dtype_name, seed))) % 1000)
    procs = [
        ctx.Process(target=_worker, args=(r, WORLD_SIZE, port, shape, dtype_name, seed, errq))
        for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    results = [errq.get() for _ in range(WORLD_SIZE)]
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            pytest.fail("worker hung — likely a send/recv deadlock in your ring")
    failures = [f"rank {r}: {err}" for r, err in results if err is not None]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "shape",
    [
        (8,),            # divisible by world_size
        (7,),            # NOT divisible
        (1,),            # numel < world_size — one chunk is empty
        (128, 64),       # 2-D, divisible
        (33, 5),         # 2-D, awkward
        (1_000_003,),    # large-ish, prime numel
    ],
)
def test_matches_all_reduce_float32(shape):
    _run_case(shape, "float32")


def test_matches_all_reduce_float64():
    _run_case((513,), "float64")


def test_matches_all_reduce_int64():
    _run_case((100,), "int64", seed=7)


def test_repeated_calls_same_group():
    """Two back-to-back reduces must not deadlock or cross-contaminate."""

    def worker(rank, world_size, port, errq):
        try:
            from src.ring_allreduce import ring_allreduce

            _init(rank, world_size, port)
            a = torch.full((16,), float(rank + 1))
            b = torch.full((16,), float((rank + 1) * 10))
            ring_allreduce(a)
            ring_allreduce(b)
            torch.testing.assert_close(a, torch.full((16,), 3.0))    # 1 + 2
            torch.testing.assert_close(b, torch.full((16,), 30.0))   # 10 + 20
            errq.put((rank, None))
        except Exception as e:  # noqa: BLE001
            errq.put((rank, f"{type(e).__name__}: {e}"))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    ctx = mp.get_context("spawn")
    errq = ctx.SimpleQueue()
    procs = [ctx.Process(target=worker, args=(r, WORLD_SIZE, 28999, errq)) for r in range(WORLD_SIZE)]
    for p in procs:
        p.start()
    results = [errq.get() for _ in range(WORLD_SIZE)]
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            pytest.fail("worker hung on repeated calls")
    failures = [f"rank {r}: {err}" for r, err in results if err is not None]
    assert not failures, "\n".join(failures)
