"""COMPLETE benchmark harness — do not modify (extend via new files if needed).

Two modes:

1. Bandwidth sweep (run under torchrun, 2 processes):
     torchrun --standalone --nproc_per_node=2 bench/scaling.py \
         --mode bandwidth --backend nccl --out bench/results/bw_nccl.json
   Compares YOUR ring_allreduce vs dist.all_reduce across message sizes,
   reports algbw and busbw (busbw = algbw * 2*(n-1)/n for all-reduce).

2. Plotting (single process, local, free):
     python bench/scaling.py --mode plot
   Reads every JSON in bench/results/ and emits PNGs:
     - bw_vs_size.png        (yours vs NCCL, log-log)
     - loss_parity.png       (overlaid loss curves from train_*.json)
     - scaling_efficiency.png (tokens/sec: 1 GPU vs 2 GPUs per impl)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

RESULTS = Path(__file__).parent / "results"

SIZES_BYTES = [2**p for p in range(10, 29)]  # 1 KiB .. 256 MiB
WARMUP, ITERS = 5, 20


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_op(fn, tensor_factory, device, iters=ITERS, warmup=WARMUP) -> float:
    """Median seconds per op. Fresh tensor per iter (all-reduce mutates)."""
    for _ in range(warmup):
        fn(tensor_factory())
    _sync(device)
    times = []
    for _ in range(iters):
        t = tensor_factory()
        _sync(device)
        dist.barrier()
        t0 = time.perf_counter()
        fn(t)
        _sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_bandwidth(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend)

    try:
        from src.ring_allreduce import ring_allreduce
        have_ring = True
    except (ImportError, NotImplementedError):
        have_ring = False
        if rank == 0:
            print("NOTE: src.ring_allreduce not importable yet — benchmarking NCCL/gloo only")

    rows = []
    for nbytes in SIZES_BYTES:
        numel = nbytes // 4  # fp32
        factory = lambda: torch.randn(numel, device=device)  # noqa: E731

        t_ref = _time_op(lambda t: dist.all_reduce(t), factory, device)
        row = {"bytes": nbytes, "ref_sec": t_ref}
        if have_ring:
            try:
                row["ring_sec"] = _time_op(lambda t: ring_allreduce(t), factory, device)
            except NotImplementedError:
                have_ring = False
        # algbw = S/t ; busbw = algbw * 2(n-1)/n   (NCCL tests convention)
        bus_factor = 2 * (world - 1) / world
        row["ref_algbw_GBs"] = nbytes / t_ref / 1e9
        row["ref_busbw_GBs"] = row["ref_algbw_GBs"] * bus_factor
        if "ring_sec" in row:
            row["ring_algbw_GBs"] = nbytes / row["ring_sec"] / 1e9
            row["ring_busbw_GBs"] = row["ring_algbw_GBs"] * bus_factor
        rows.append(row)
        if rank == 0:
            ring = f"{row.get('ring_busbw_GBs', float('nan')):8.3f}" if "ring_sec" in row else "     n/a"
            print(f"{nbytes:>12} B  ref busbw {row['ref_busbw_GBs']:8.3f} GB/s  ring busbw {ring} GB/s")

    if rank == 0:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = Path(args.out or RESULTS / f"bw_{args.backend}.json")
        out.write_text(json.dumps({
            "backend": args.backend, "world_size": world,
            "device": str(device), "rows": rows,
        }, indent=2))
        print(f"wrote {out}")
    dist.destroy_process_group()


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- bandwidth ----
    bw_files = sorted(RESULTS.glob("bw_*.json"))
    if bw_files:
        fig, ax = plt.subplots(figsize=(8, 5))
        for f in bw_files:
            d = json.loads(f.read_text())
            xs = [r["bytes"] for r in d["rows"]]
            ax.plot(xs, [r["ref_busbw_GBs"] for r in d["rows"]], "o-",
                    label=f"{d['backend']} all_reduce")
            if all("ring_busbw_GBs" in r for r in d["rows"]):
                ax.plot(xs, [r["ring_busbw_GBs"] for r in d["rows"]], "s--",
                        label=f"my ring ({d['backend']})")
        ax.set(xscale="log", yscale="log", xlabel="message size (bytes)",
               ylabel="busbw (GB/s)", title="All-reduce bus bandwidth, 2 ranks")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.savefig(RESULTS / "bw_vs_size.png", dpi=150, bbox_inches="tight")
        print("wrote bw_vs_size.png")

    # ---- loss parity ----
    train_files = sorted(RESULTS.glob("train_*.json"))
    if train_files:
        fig, ax = plt.subplots(figsize=(8, 5))
        for f in train_files:
            d = json.loads(f.read_text())
            ax.plot(d["loss_curve"], label=f"{d['impl']} (ws={d['world_size']})", alpha=0.8)
        ax.set(xlabel="step", ylabel="loss", title="Loss parity: manual vs DDP vs FSDP2")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(RESULTS / "loss_parity.png", dpi=150, bbox_inches="tight")
        print("wrote loss_parity.png")

        # ---- scaling efficiency ----
        single = next((json.loads(f.read_text()) for f in train_files
                       if json.loads(f.read_text())["impl"] == "single"), None)
        if single:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels, effs = [], []
            for f in train_files:
                d = json.loads(f.read_text())
                if d["impl"] == "single":
                    continue
                eff = d["tokens_per_sec"] / (d["world_size"] * single["tokens_per_sec"])
                labels.append(f"{d['impl']}\nws={d['world_size']}")
                effs.append(eff * 100)
            ax.bar(labels, effs)
            ax.axhline(100, ls="--", c="gray", label="ideal")
            ax.set(ylabel="scaling efficiency (%)",
                   title=f"Scaling vs 1 GPU ({single['tokens_per_sec']:.0f} tok/s)")
            for i, e in enumerate(effs):
                ax.text(i, e + 1, f"{e:.1f}%", ha="center")
            fig.savefig(RESULTS / "scaling_efficiency.png", dpi=150, bbox_inches="tight")
            print("wrote scaling_efficiency.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["bandwidth", "plot"], required=True)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.mode == "bandwidth":
        bench_bandwidth(args)
    else:
        plot()


if __name__ == "__main__":
    main()
