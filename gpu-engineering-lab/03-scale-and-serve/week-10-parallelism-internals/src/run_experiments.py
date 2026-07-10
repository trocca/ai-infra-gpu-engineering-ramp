"""Day-4 experiment driver + Day-5 plotting.

YOUR TASK: produce the numbers for the README results table.

    torchrun --standalone --nproc_per_node=2 -m src.run_experiments --exp tp
    torchrun --standalone --nproc_per_node=2 -m src.run_experiments --exp pp
    python -m src.run_experiments --exp oom      # 1 process, must OOM
    python -m src.run_experiments --plot         # local, free

Experiments (all on the deliberately-too-big config):
  oom : single GPU, demonstrate + LOG the OOM (catch torch.cuda.OutOfMemoryError,
        write the message to bench/results/oom.log — that log is a deliverable).
  tp  : TP=2 — train N steps, record peak mem/GPU, tokens/sec, step time.
  pp  : PP=2 (m=8) — same measurements.

JSON schema per experiment (keep it — plotting depends on it):
  {"exp": "tp", "world_size": 2, "config": {...},
   "peak_mem_bytes_per_rank": [..], "tokens_per_sec": ..., "step_time_ms": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"


def too_big_config() -> dict:
    """A GPT config sized to NOT fit on one 24 GB L4 with AdamW fp32.

    Rough budget: params * (4 bytes weights + 4 grads + 8 Adam states)
    = params * 16 bytes, plus activations. ~1.3-1.5 B params already
    breaks 20 GB before activations.

    TODO(you): pick d_model / n_layers / n_heads / seq_len; verify with
    the oom experiment; record your sizing arithmetic in the README.
    """
    raise NotImplementedError


def run_oom() -> None:
    # TODO(you): build model + optimizer on cuda:0, run ONE step inside
    # try/except torch.cuda.OutOfMemoryError, write bench/results/oom.log.
    raise NotImplementedError


def run_tp(steps: int = 50) -> None:
    # TODO(you): TPGPT with too_big_config, N steps, measure.
    # torch.cuda.reset_peak_memory_stats() before; max_memory_allocated() after.
    raise NotImplementedError


def run_pp(steps: int = 50, microbatches: int = 8) -> None:
    # TODO(you): pipeline with too_big_config, same measurements.
    raise NotImplementedError


def plot() -> None:
    """COMPLETE-ish: bubble plot + memory/throughput bar chart from the
    JSONs above. Extend as needed but keep output filenames stable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bubble = RESULTS / "bubble.json"
    if bubble.exists():
        d = json.loads(bubble.read_text())
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(d["m"], d["theory"], "o--", label="theory (p-1)/(m+p-1)")
        ax.plot(d["m"], d["measured"], "s-", label="measured")
        ax.set(xlabel="microbatches m", ylabel="bubble fraction",
               title="GPipe bubble: theory vs measured (p=2)")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(RESULTS / "bubble.png", dpi=150, bbox_inches="tight")
        print("wrote bubble.png")

    exps = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("exp_*.json"))]
    if exps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        names = [e["exp"] for e in exps]
        ax1.bar(names, [max(e["peak_mem_bytes_per_rank"]) / 2**30 for e in exps])
        ax1.axhline(22, ls="--", c="r", label="L4 usable ~22 GB")
        ax1.set(ylabel="peak GiB / GPU", title="Memory: it didn't fit on one")
        ax1.legend()
        ax2.bar(names, [e["tokens_per_sec"] for e in exps])
        ax2.set(ylabel="tokens/sec", title="Throughput")
        fig.savefig(RESULTS / "tp_vs_pp.png", dpi=150, bbox_inches="tight")
        print("wrote tp_vs_pp.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", choices=["oom", "tp", "pp"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot()
    elif args.exp == "oom":
        run_oom()
    elif args.exp == "tp":
        run_tp(args.steps)
    elif args.exp == "pp":
        run_pp(args.steps)
    else:
        p.error("need --exp or --plot")


if __name__ == "__main__":
    main()
