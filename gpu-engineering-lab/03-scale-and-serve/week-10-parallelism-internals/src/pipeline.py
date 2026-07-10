"""GPipe-style pipeline parallelism: 2 stages, fill-drain schedule.

YOUR TASK (Day 3): split the week-05 GPT across 2 ranks and train with
microbatching. No torch.distributed.pipelining — you ARE the scheduler.

Mechanics for p=2 stages, m microbatches (fill-drain / "GPipe" schedule):

  rank 0 (stage 0): fwd(mb_0) fwd(mb_1) ... fwd(mb_{m-1})   then
                    bwd(mb_0) bwd(mb_1) ... bwd(mb_{m-1})
  rank 1 (stage 1): recv act, fwd+loss, bwd, send grad — per microbatch.

  Forward:  stage0 sends activations  ->  stage1 receives
  Backward: stage1 sends d(loss)/d(activations)  ->  stage0 receives,
            then calls activation.backward(grad=received)

Gotchas:
  * Stage 0 must KEEP each microbatch's output activation (with graph)
    until its backward arrives — that's the GPipe memory cost, and why
    1F1B (stretch) exists.
  * requires_grad: the received activation on stage 1 must have
    requires_grad_(True) BEFORE the stage-1 forward, and you backward
    from stage-1 loss to that leaf, then ship leaf.grad to stage 0.
  * Gradient scale: sum of per-microbatch mean-losses != mean over the
    global batch. Divide each microbatch loss by m (or average grads)
    so parity with the single-GPU baseline holds.
  * Shapes must be known to the receiver: send a header or fix them by config.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist
from torch import nn


def split_model(model: nn.Module, stage: int) -> nn.Module:
    """Return the sub-module for this stage.

    stage 0: embeddings + blocks[: n//2]
    stage 1: blocks[n//2 :] + final ln + lm_head (+ loss)

    TODO(you): implement against the week-05 GPT structure. An
    nn.Sequential of the relevant pieces is fine; balance can be tuned
    later (uneven split shows up in the bubble measurement — interesting!).
    """
    raise NotImplementedError


class PipelineStage:
    """One rank's half of the fill-drain schedule."""

    def __init__(self, module: nn.Module, stage: int, num_stages: int = 2):
        self.module = module
        self.stage = stage
        self.num_stages = num_stages
        self.peer = 1 - stage  # only valid for p=2; generalize if you like

    def run_batch(self, microbatches, targets, optimizer) -> float:
        """One optimizer step over m microbatches. Returns mean loss (stage 1)
        or 0.0 (stage 0).

        TODO(you):
          Phase 1 (fill): all m forwards.
            stage 0: out = self.module(mb); dist.send(out.detach(), peer);
                     stash out for backward
            stage 1: act = recv; act.requires_grad_(True);
                     loss = criterion(self.module(act), tgt) / m;
                     stash (loss, act)
          Phase 2 (drain): all m backwards, SAME order (GPipe) — then
            stage 1: loss.backward(); dist.send(act.grad, peer)
            stage 0: g = recv; stashed_out.backward(gradient=g)
          Then: optimizer.step(); optimizer.zero_grad()

        Instrument: record wall time of compute vs waiting-on-recv per
        phase — you need it for the bubble measurement below.
        """
        raise NotImplementedError


def measure_bubble(stage_runner: PipelineStage, microbatch_counts=(1, 2, 4, 8, 16)) -> dict:
    """Measure bubble fraction vs m and compare to theory.

    bubble_theory = (p - 1) / (m + p - 1)

    Measured bubble: 1 - (busy_time / total_time) per rank, where busy
    time is time spent in fwd/bwd compute (you instrumented run_batch).
    Use fixed-size microbatches so per-mb compute is constant; keep the
    GLOBAL batch fixed while varying m.

    TODO(you): loop over microbatch_counts, run a few timed steps each,
    return {"m": [...], "measured": [...], "theory": [...]} and dump to
    bench/results/bubble.json (create the dir). run_experiments.py plots it.
    """
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--measure-bubble", action="store_true")
    args = p.parse_args()
    # TODO(you): init process group (nccl on cloud, gloo locally),
    # build + split model, train, optionally measure_bubble, write JSON.
    raise NotImplementedError


if __name__ == "__main__":
    main()
