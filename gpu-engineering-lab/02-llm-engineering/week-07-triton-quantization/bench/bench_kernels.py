"""COMPLETE benchmark harness — do not edit (extend if you like).

Benchmarks your Triton kernels against torch across sizes. Honest-benchmark
rules: CUDA-event timing, warmup, median of >= 50 runs, implementations
interleaved by size. Skips any kernel still raising NotImplementedError.

Also produces the FlashAttention MEMORY figure (O(N) yours vs O(N^2) math
backend) — the plot that makes the writeup.

Usage:
    python bench/bench_kernels.py                      # everything
    python bench/bench_kernels.py --op softmax flash   # subset
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = 50
WARMUP = 10


def bench_ms(fn, runs: int = RUNS, warmup: int = WARMUP) -> dict:
    """Median/p10/p90 latency in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"median_ms": statistics.median(times),
            "p10_ms": times[max(0, int(0.10 * len(times)) - 1)],
            "p90_ms": times[int(0.90 * len(times)) - 1],
            "runs": runs}


def try_kernel(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except NotImplementedError as e:
        return None, str(e)


def bench_softmax(results: dict) -> None:
    from src.softmax_triton import softmax
    rows = []
    for n_cols in (256, 1024, 4096, 16384):
        x = torch.randn(4096, n_cols, device="cuda", dtype=torch.float16)
        _, err = try_kernel(softmax, x)
        if err:
            print(f"softmax: SKIP ({err})")
            return
        row = {"shape": [4096, n_cols],
               "triton": bench_ms(lambda: softmax(x)),
               "torch_eager": bench_ms(lambda: torch.softmax(x, -1))}
        rows.append(row)
        print(f"softmax {n_cols:6d}: triton {row['triton']['median_ms']:.3f} ms  "
              f"eager {row['torch_eager']['median_ms']:.3f} ms")
    results["softmax"] = rows


def bench_rmsnorm(results: dict) -> None:
    from src.rmsnorm_triton import rmsnorm, rmsnorm_torch
    rows = []
    for n_cols in (512, 2048, 8192):
        x = torch.randn(4096, n_cols, device="cuda", dtype=torch.float16)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float16)
        _, err = try_kernel(rmsnorm, x, w)
        if err:
            print(f"rmsnorm: SKIP ({err})")
            return
        row = {"shape": [4096, n_cols],
               "triton": bench_ms(lambda: rmsnorm(x, w)),
               "torch_eager": bench_ms(lambda: rmsnorm_torch(x, w))}
        rows.append(row)
        print(f"rmsnorm {n_cols:6d}: triton {row['triton']['median_ms']:.3f} ms  "
              f"eager {row['torch_eager']['median_ms']:.3f} ms")
    results["rmsnorm"] = rows


def _sdpa_backend(q, k, v, backend):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def bench_flash(results: dict) -> None:
    from torch.nn.attention import SDPBackend
    from src.flash_fwd_triton import flash_attention_forward
    B, H, D = 1, 16, 64
    latency_rows, memory_rows = [], []
    for N in (256, 512, 1024, 2048, 4096, 8192):
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3))
        _, err = try_kernel(flash_attention_forward, q, k, v, True)
        if err:
            print(f"flash: SKIP ({err})")
            return
        row = {"seq_len": N,
               "triton_flash": bench_ms(lambda: flash_attention_forward(q, k, v, True)),
               "sdpa_flash": bench_ms(lambda: _sdpa_backend(q, k, v, SDPBackend.FLASH_ATTENTION))}
        if N <= 4096:  # math backend materializes N^2 — cap it before OOM
            row["sdpa_math"] = bench_ms(lambda: _sdpa_backend(q, k, v, SDPBackend.MATH))
        latency_rows.append(row)
        print(f"flash N={N:5d}: yours {row['triton_flash']['median_ms']:.3f} ms  "
              f"sdpa-flash {row['sdpa_flash']['median_ms']:.3f} ms"
              + (f"  math {row['sdpa_math']['median_ms']:.3f} ms" if "sdpa_math" in row else ""))

        # peak-memory comparison (the O(N) vs O(N^2) figure)
        mem = {"seq_len": N}
        for name, fn in (("triton_flash", lambda: flash_attention_forward(q, k, v, True)),
                         ("sdpa_math", lambda: _sdpa_backend(q, k, v, SDPBackend.MATH))):
            if name == "sdpa_math" and N > 4096:
                continue
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn()
            torch.cuda.synchronize()
            mem[name + "_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        memory_rows.append(mem)
    results["flash_latency"] = latency_rows
    results["flash_memory"] = memory_rows


def bench_w8a16(results: dict) -> None:
    from src.quant.matmul_w8a16 import matmul_w8a16
    from src.quant.pack import quantize_int8
    rows = []
    for M, N, K in ((1, 4096, 4096), (16, 4096, 4096), (128, 4096, 4096)):
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        try:
            w_int8, scales = quantize_int8(w)
        except NotImplementedError as e:
            print(f"w8a16: SKIP ({e})")
            return
        w_int8, scales = w_int8.cuda(), scales.cuda()
        _, err = try_kernel(matmul_w8a16, x, w_int8, scales)
        if err:
            print(f"w8a16: SKIP ({err})")
            return
        row = {"shape": [M, N, K],
               "triton_w8a16": bench_ms(lambda: matmul_w8a16(x, w_int8, scales)),
               "torch_fp16": bench_ms(lambda: x @ w.T)}
        rows.append(row)
        print(f"w8a16 M={M:4d}: yours {row['triton_w8a16']['median_ms']:.3f} ms  "
              f"cuBLAS fp16 {row['torch_fp16']['median_ms']:.3f} ms")
    results["w8a16"] = rows


def plot(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — JSON only")
        return

    if "flash_latency" in results:
        rows = results["flash_latency"]
        xs = [r["seq_len"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [r["triton_flash"]["median_ms"] for r in rows], "o-", label="yours (Triton)")
        ax.plot(xs, [r["sdpa_flash"]["median_ms"] for r in rows], "o-", label="SDPA flash")
        math_pts = [(r["seq_len"], r["sdpa_math"]["median_ms"]) for r in rows if "sdpa_math" in r]
        if math_pts:
            ax.plot(*zip(*math_pts), "o--", label="SDPA math (naive)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("sequence length"); ax.set_ylabel("median latency (ms)")
        ax.set_title("Causal attention forward"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "flash_latency.png", dpi=150)

    if "flash_memory" in results:
        rows = results["flash_memory"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["seq_len"] for r in rows],
                [r["triton_flash_mb"] for r in rows], "o-", label="yours: O(N)")
        math_pts = [(r["seq_len"], r["sdpa_math_mb"]) for r in rows if "sdpa_math_mb" in r]
        if math_pts:
            ax.plot(*zip(*math_pts), "o--", label="math backend: O(N^2)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("sequence length"); ax.set_ylabel("peak extra memory (MB)")
        ax.set_title("Attention activation memory"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "flash_memory.png", dpi=150)

    for op in ("softmax", "rmsnorm"):
        if op not in results:
            continue
        rows = results[op]
        xs = [r["shape"][1] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [r["triton"]["median_ms"] for r in rows], "o-", label="yours (Triton)")
        ax.plot(xs, [r["torch_eager"]["median_ms"] for r in rows], "o-", label="torch eager")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("columns (rows=4096, fp16)"); ax.set_ylabel("median latency (ms)")
        ax.set_title(f"Fused {op}"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{op}_latency.png", dpi=150)
    print(f"plots written to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", nargs="+",
                   default=["softmax", "rmsnorm", "flash", "w8a16"],
                   choices=["softmax", "rmsnorm", "flash", "w8a16"])
    p.add_argument("--out", default="bench/results")
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU required (run under WSL2)"

    results = {"device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "runs": RUNS, "warmup": WARMUP}
    if "softmax" in args.op:
        bench_softmax(results)
    if "rmsnorm" in args.op:
        bench_rmsnorm(results)
    if "flash" in args.op:
        bench_flash(results)
    if "w8a16" in args.op:
        bench_w8a16(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernels_bench.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'kernels_bench.json'}")
    plot(results, out_dir)


if __name__ == "__main__":
    main()
