#!/usr/bin/env python3
"""
train.py — the ENTIRE user-facing surface of the Kubernetes-AI stack (layer 15).

This script is deliberately boring. That's the point of the demo:
everything that makes it run across N nodes — rendezvous, ranks, NCCL
transport selection, GPU injection, scheduling — happens BELOW this file
and is injected by the platform.

What it does:
  1. Prints the distributed environment it was handed (proof of injection).
  2. Initializes torch.distributed via the standard env:// contract.
  3. Trains a small Transformer on synthetic data with DDP.
  4. Logs per-step throughput and an all-reduced global loss.
  5. Checkpoints from rank 0 only.

Run locally (single node, 2 GPUs):
  torchrun --nproc-per-node=2 train.py --steps 50

On Kubernetes the TrainJob operator + torchrun set every env var below.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


# --------------------------------------------------------------------------
# 1. The env-var contract. WE READ, THE PLATFORM WRITES.
#    - Kubeflow Trainer creates the headless Service + sets the master addr
#    - torchrun assigns RANK / LOCAL_RANK / WORLD_SIZE per process
# --------------------------------------------------------------------------
ENV_CONTRACT = [
    "MASTER_ADDR",   # FQDN of the rank-0 pod (headless-Service DNS)
    "MASTER_PORT",   # port of the c10d TCPStore on rank 0
    "RANK",          # this process's global rank        (0..WORLD_SIZE-1)
    "LOCAL_RANK",    # this process's rank on this node  (picks the GPU)
    "WORLD_SIZE",    # total processes across all nodes
    "GROUP_RANK",    # this node's rank (torchrun)
    "LOCAL_WORLD_SIZE",
]


def print_env_contract() -> None:
    print("=" * 62)
    print("DISTRIBUTED ENV (injected by the operator + torchrun, not by me)")
    for key in ENV_CONTRACT:
        print(f"  {key:<18} = {os.environ.get(key, '<unset>')}")
    print("=" * 62, flush=True)


# --------------------------------------------------------------------------
# 2. A small-but-real model: a tiny GPT-style Transformer.
#    Big enough that DDP's bucketed all-reduce actually shows up in
#    NCCL_DEBUG=INFO logs; small enough to run on any cloud GPU.
# --------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab=8192, d_model=512, n_heads=8, n_layers=6, seq=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq = seq

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)
        h = self.blocks(h)
        return self.head(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt", type=str, default="/tmp/ckpt.pt")
    args = parser.parse_args()

    print_env_contract()

    # ----------------------------------------------------------------------
    # 3. init_process_group with the default env:// method reads
    #    MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE from the environment.
    #    Backend "nccl" hands collectives to NCCL, which auto-detects the
    #    topology (NVLink intra-node, IB/RoCE + GPUDirect RDMA inter-node).
    # ----------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])  # LOCAL_RANK picks the GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[rank 0] world_size={world}, device={torch.cuda.get_device_name(device)}",
              flush=True)

    model = TinyGPT(seq=args.seq).to(device)
    # DDP: replicate the model per GPU, all-reduce gradient buckets in backward.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic next-token-prediction data: no dataset download, fully offline.
    def batch():
        x = torch.randint(0, 8192, (args.batch_size, args.seq), device=device)
        return x[:, :-1], x[:, 1:]

    model.train()
    tokens_per_step = args.batch_size * (args.seq - 1)
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        inp, tgt = batch()
        optim.zero_grad(set_to_none=True)
        out = model(inp)
        loss = loss_fn(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
        loss.backward()          # <- DDP all-reduces gradient buckets here
        optim.step()

        if step % 10 == 0:
            # All-reduce the loss so every rank logs the same global number —
            # also a nice explicit NCCL collective to point at in the demo.
            with torch.no_grad():
                g = loss.detach().clone()
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
            dt = time.perf_counter() - t0
            tps = 10 * tokens_per_step * world / dt
            if rank == 0:
                print(f"step {step:>4}/{args.steps}  global_loss={g.item():.4f}  "
                      f"tokens/sec (all ranks)={tps:,.0f}", flush=True)
            t0 = time.perf_counter()

    # Rank-0-only checkpoint: the standard pattern for resumable training.
    if rank == 0:
        torch.save({"model": model.module.state_dict(),
                    "steps": args.steps}, args.ckpt)
        print(f"[rank 0] checkpoint written to {args.ckpt}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
