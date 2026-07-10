"""Manual DDP: gradient-hook data parallelism built on YOUR ring all-reduce.

YOUR TASK: wrap an arbitrary nn.Module so that, after backward(), every
rank holds identical averaged gradients — using ring_allreduce_mean from
src/ring_allreduce.py, not dist.all_reduce.

What real DDP does that you are re-implementing (see the DDP design note
in the README background reading):
  1. Broadcast parameters from rank 0 at construction so all replicas
     start identical.
  2. Register a hook per parameter that fires when its grad is ready.
  3. Average gradients across ranks before optimizer.step().

What real DDP does that you are NOT implementing today (stretch goals):
  bucketing (~25 MB flat buffers) and comm/compute overlap.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from .ring_allreduce import ring_allreduce_mean


class ManualDDP(nn.Module):
    """Minimal correct data-parallel wrapper.

    Usage (identical on every rank):
        model = ManualDDP(build_model().to(device))
        ...
        loss.backward()
        model.finalize_backward()   # waits for / applies all grad averaging
        optimizer.step()
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # ------------------------------------------------------------
        # TODO(you) 1: broadcast all parameters (and buffers! think
        # BatchNorm running stats, GPT positional buffers) from rank 0
        # so every replica starts from the same weights.
        # Hint: dist.broadcast(p.data, src=0) — or be fancy and flatten.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO(you) 2: register a post-accumulate-grad hook on every
        # parameter that requires grad:
        #
        #   p.register_post_accumulate_grad_hook(self._grad_hook)
        #
        # Design decision you must make (document it in the README):
        #   (a) all-reduce inside the hook, synchronously (simplest,
        #       correct, zero overlap), or
        #   (b) collect ready params and reduce them in
        #       finalize_backward() (enables bucketing later).
        # Start with (a). Loss-parity first, speed second.
        # ------------------------------------------------------------
        raise NotImplementedError("implement ManualDDP.__init__")

    def _grad_hook(self, param: torch.Tensor) -> None:
        # TODO(you): average param.grad across ranks with
        # ring_allreduce_mean. Careful: hooks fire during backward —
        # keep it simple and synchronous for the baseline version.
        raise NotImplementedError

    def finalize_backward(self) -> None:
        """Call after loss.backward(), before optimizer.step().

        For the synchronous version this can be a no-op; for the
        bucketed/overlapped stretch version this is where you wait on
        outstanding comm handles and copy reduced buckets back.
        """
        # TODO(you)
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def sanity_check_sync(model: ManualDDP) -> None:
    """Assert all ranks hold identical parameters. COMPLETE — use freely.

    Call this after construction and periodically during training; it
    catches 90% of manual-DDP bugs (divergent replicas) early.
    """
    for name, p in model.module.named_parameters():
        ref = p.data.clone()
        dist.broadcast(ref, src=0)
        if not torch.allclose(p.data, ref, rtol=0, atol=0):
            raise RuntimeError(
                f"rank {dist.get_rank()}: parameter {name} diverged from rank 0 "
                f"(max abs diff {(p.data - ref).abs().max().item():.3e})"
            )
