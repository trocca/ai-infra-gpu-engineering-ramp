#!/usr/bin/env python3
"""Benchmark rusty_kernels vs eager PyTorch and torch.compile (COMPLETE).

Sweeps transformer-shaped inputs, times with torch.cuda.Event (10 warmup +
50 timed, median), writes results/bench_ops.json and two charts.

Repo contract: this one command reproduces every Week 04 number.
Usage: python bench/bench_ops.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import rusty_kernels

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
WARMUP = 10
ITERS = 50

# rows = batch * seq, cols = hidden — typical transformer activations
ROWS = [512, 2048, 8192, 16384]
HIDDEN = [768, 1024, 2048, 4096, 8192]
DTYPES = [torch.float32, torch.float16]


def time_fn(fn, *args) -> float:
    """Median kernel time in ms via CUDA events."""
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(ITERS):
        start.record()
        fn(*args)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def bench_softmax(rows: list[dict]) -> None:
    compiled = torch.compile(lambda t: F.softmax(t, dim=-1), dynamic=False)
    for dtype in DTYPES:
        for r in ROWS:
            for h in HIDDEN:
                x = torch.randn(r, h, device="cuda", dtype=dtype)
                bytes_moved = 2 * x.numel() * x.element_size()  # 1 read + 1 write, ideal
                variants = {
                    "eager": lambda t: F.softmax(t, dim=-1),
                    "compile": compiled,
                    "fused": lambda t: rusty_kernels.softmax(t, dim=-1),
                }
                for name, fn in variants.items():
                    ms = time_fn(fn, x)
                    rows.append({
                        "op": "softmax", "impl": name, "rows": r, "hidden": h,
                        "dtype": str(dtype).split(".")[-1], "median_ms": ms,
                        "gbps": bytes_moved / (ms * 1e-3) / 1e9,
                    })
                print(f"softmax {dtype} {r}x{h}: "
                      + "  ".join(f"{v['impl']}={v['median_ms']:.3f}ms"
                                  for v in rows[-3:]))


def bench_layernorm(rows: list[dict]) -> None:
    for dtype in DTYPES:
        for r in ROWS:
            for h in HIDDEN:
                x = torch.randn(r, h, device="cuda", dtype=dtype)
                w = torch.randn(h, device="cuda", dtype=dtype)
                b = torch.randn(h, device="cuda", dtype=dtype)
                compiled = torch.compile(
                    lambda t, w_, b_: F.layer_norm(t, (t.shape[-1],), w_, b_),
                    dynamic=False)
                bytes_moved = 2 * x.numel() * x.element_size()
                variants = {
                    "eager": lambda t, w_, b_: F.layer_norm(t, (t.shape[-1],), w_, b_),
                    "compile": compiled,
                    "fused": rusty_kernels.layer_norm,
                }
                for name, fn in variants.items():
                    ms = time_fn(fn, x, w, b)
                    rows.append({
                        "op": "layer_norm", "impl": name, "rows": r, "hidden": h,
                        "dtype": str(dtype).split(".")[-1], "median_ms": ms,
                        "gbps": bytes_moved / (ms * 1e-3) / 1e9,
                    })
                print(f"layer_norm {dtype} {r}x{h}: "
                      + "  ".join(f"{v['impl']}={v['median_ms']:.3f}ms"
                                  for v in rows[-3:]))


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for op in ("softmax", "layer_norm"):
        sub = [r for r in rows if r["op"] == op and r["dtype"] == "float32"
               and r["rows"] == max(ROWS)]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for impl, marker in (("eager", "o"), ("compile", "s"), ("fused", "^")):
            pts = sorted((r for r in sub if r["impl"] == impl), key=lambda r: r["hidden"])
            ax.plot([p["hidden"] for p in pts], [p["median_ms"] for p in pts],
                    marker=marker, label=impl)
        ax.set_xscale("log", base=2)
        ax.set_xticks(HIDDEN, [str(h) for h in HIDDEN])
        ax.set_xlabel("hidden size (rows = %d, fp32)" % max(ROWS))
        ax.set_ylabel("median ms (50 iters after warmup)")
        ax.set_title(f"{op}: fused vs eager vs torch.compile — RTX 5090 Laptop (sm_120)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = RESULTS_DIR / f"bench_{op}.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


def main() -> None:
    assert torch.cuda.is_available(), "benchmarks need the GPU"
    torch.manual_seed(0)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows: list[dict] = []
    bench_softmax(rows)
    bench_layernorm(rows)

    out = RESULTS_DIR / "bench_ops.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} rows -> {out}")
    plot(rows)


if __name__ == "__main__":
    main()
