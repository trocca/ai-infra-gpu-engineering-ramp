"""Training loop: bf16 autocast, grad accumulation, cosine schedule, CSV logging.

Usage:
    python -m src.train --config d12 --data data --out checkpoints/d12 \
        --batch-size 32 --seq-len 512 --grad-accum 32 --max-steps 5000

Effective tokens/step = batch_size * seq_len * grad_accum  (aim ~0.5M).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch

from .data import get_batch
from .model import CONFIGS, GPT


def cosine_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup: int) -> float:
    """TODO(Day 3): linear warmup then cosine decay.

      step < warmup:            max_lr * (step + 1) / warmup
      warmup <= step < max:     min_lr + 0.5*(max_lr-min_lr)*(1 + cos(pi * progress))
      step >= max_steps:        min_lr

    Key idea: warmup protects Adam's second-moment estimates early; cosine
    gives a smooth anneal so the final weights sit in a flatter minimum.
    """
    raise NotImplementedError("Day 3: implement cosine_lr")


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str, batch_size: int, seq_len: int,
                  device: str, iters: int = 50) -> dict[str, float]:
    """TODO(Day 3): mean loss over `iters` random batches for train.bin and
    val.bin. model.eval() before / model.train() after. Run under the same
    bf16 autocast context as training so the numbers are comparable."""
    raise NotImplementedError("Day 3: implement estimate_loss")


def train(args: argparse.Namespace) -> None:
    """TODO(Day 3): the loop. Target structure:

      1. Setup: device cuda, torch.manual_seed, model = GPT(CONFIGS[args.config])
         .to(device), print param count. Optional: torch.compile (stretch).
         Enable TF32 matmuls for any residual fp32 ops:
         torch.set_float32_matmul_precision("high").
      2. Optimizer: AdamW(lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1),
         and DECAY ONLY 2D+ params (weights), not norms/embeddings-bias — build
         two param groups.
      3. autocast context: torch.autocast("cuda", dtype=torch.bfloat16).
         NOTE: bf16 has fp32's exponent range -> NO GradScaler needed
         (GradScaler is an fp16 artifact). Blackwell runs bf16 natively.
      4. Each step: accumulate args.grad_accum micro-batches
         (loss / grad_accum, backward each), then clip grad norm to 1.0,
         set lr from cosine_lr, optimizer.step(), zero_grad(set_to_none=True).
      5. Logging: every step append to {out}/log.csv
         (step, loss, lr, tokens_per_sec, elapsed). tokens/s needs
         torch.cuda.synchronize() before reading the clock. wandb optional —
         CSV is the contract, plots are made from it.
      6. Every args.eval_every steps: estimate_loss, print, save checkpoint
         {model, optimizer, step, config} if val improved.
    """
    raise NotImplementedError("Day 3: implement train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT on TinyStories")
    p.add_argument("--config", default="d12", choices=sorted(CONFIGS))
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="checkpoints/d12")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true", help="stretch: torch.compile")
    p.add_argument("--profile", action="store_true",
                   help="Day 5: wrap ~5 steps in torch.profiler, export chrome trace")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    train(a)
