"""Distributed training driver: manual DDP vs real DDP vs FSDP2.

Launched via torchrun (see src/launch.sh). Must support:

    --impl {manual,ddp,fsdp,single}   which data-parallel wrapper
    --backend {nccl,gloo}             gloo for local CPU debugging
    --steps N                         training steps (default 200)
    --seed N                          MUST produce identical data order
                                      across impls for loss-parity runs
    --out PATH                        JSON metrics file (per rank 0)

The week-05 GPT is the model under training. Import it; do not copy it.
If your repo layout differs, fix the import below — that's part of the job.

Output JSON schema (bench/scaling.py consumes this — keep it):
{
  "impl": "manual", "world_size": 2, "backend": "nccl",
  "steps": 200, "seed": 1337,
  "loss_curve": [..per-step float..],
  "step_time_ms": {"mean": ..., "p50": ..., "p90": ...},
  "tokens_per_sec": ...
}
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# TODO(you): fix this import to point at your week-05 GPT.
# from ...02_phase.week_05.src.model import GPT, GPTConfig  # noqa: ERA001


def setup_distributed(backend: str) -> tuple[int, int, torch.device]:
    """Init process group from torchrun env vars; return (rank, world, device).

    TODO(you) Day 1: implement using env:// rendezvous. Read RANK,
    WORLD_SIZE, LOCAL_RANK from the environment. For nccl, also call
    torch.cuda.set_device(local_rank) BEFORE init (avoids the classic
    both-ranks-on-GPU-0 bug — you want to see that bug once, though).
    """
    raise NotImplementedError


def build_dataloader(rank: int, world_size: int, seed: int):
    """Sharded data. Requirements for loss parity across impls:

    * With the same seed, the UNION of batches across ranks must be the
      same set regardless of impl, and per-rank order must be
      deterministic. DistributedSampler(shuffle=True, seed=seed) or a
      manual stride-shard both work.
    * Effective global batch must match the single-GPU baseline (i.e.
      per-rank batch = global_batch / world_size).

    TODO(you): implement against the week-05 dataset.
    """
    raise NotImplementedError


def wrap_model(model: torch.nn.Module, impl: str) -> torch.nn.Module:
    """Return the model wrapped per --impl.

    TODO(you) Day 3:
      manual -> ManualDDP(model)                     (your code)
      ddp    -> torch.nn.parallel.DistributedDataParallel(model)
      fsdp   -> torch.distributed.fsdp.fully_shard(model)  (FSDP2)
      single -> model unchanged (world_size must be 1)
    """
    raise NotImplementedError


def train(args: argparse.Namespace) -> None:
    # TODO(you): the loop. Keep it boring:
    #   setup -> seed everything -> model -> wrap -> optimizer(AdamW)
    #   -> for step in range(args.steps): fwd, loss, bwd,
    #      (manual impl: model.finalize_backward()), step, zero_grad
    #   -> time each step with torch.cuda.synchronize() around timers
    #   -> rank 0 writes the JSON metrics file (schema in module docstring)
    #
    # Loss-parity trap: DDP averages grads across ranks; make sure your
    # manual impl AND your loss reduction (mean over batch) compose to
    # the same effective gradient as the single-GPU global-batch run.
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", choices=["manual", "ddp", "fsdp", "single"], required=True)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
