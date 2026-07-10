#!/usr/bin/env python3
"""Plot Week 02 benchmark results (COMPLETE — do not edit, extend if you like).

Reads every results/*.json produced by the Rust binaries and emits:
  results/bandwidth_chart.png   — strided + offset sweeps (the coalescing cliff)
  results/reduction_chart.png   — reduction ladder vs the cuBLAS Sasum baseline

Usage: python bench/plot_results.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")


def load_all(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.extend(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"warning: skipping {path}: {exc}")
    return rows


def plot_bandwidth(rows: list[dict]) -> None:
    strided = sorted((r for r in rows if r["name"] == "strided_read"), key=lambda r: r["stride"])
    offset = sorted((r for r in rows if r["name"] == "offset_read"), key=lambda r: r["offset"])
    if not strided and not offset:
        print("no bandwidth_sweep results found — skipping bandwidth chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    if strided:
        ax = axes[0]
        ax.plot([r["stride"] for r in strided], [r["gbps"] for r in strided],
                marker="o", linewidth=1.5)
        ax.set_xlabel("read stride (floats)")
        ax.set_ylabel("effective bandwidth (GB/s)")
        ax.set_title("Strided global reads — the coalescing cliff")
        ax.axvline(32, linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3)

    if offset:
        ax = axes[1]
        ax.plot([r["offset"] for r in offset], [r["gbps"] for r in offset],
                marker="s", linewidth=1.5, color="tab:orange")
        ax.set_xlabel("read offset (floats)")
        ax.set_ylabel("effective bandwidth (GB/s)")
        ax.set_title("Misaligned global reads")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RTX 5090 Laptop GPU (Blackwell, sm_120) — median of 50 runs after warmup",
                 fontsize=9)
    out = RESULTS / "bandwidth_chart.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_reductions(rows: list[dict]) -> None:
    order = ["reduce_naive", "reduce_shared", "reduce_warp", "cublas_sasum"]
    have = {r["name"]: r for r in rows if r["name"] in order}
    if not have:
        print("no reduction results found — skipping reduction chart")
        return

    names = [n for n in order if n in have]
    gbps = [have[n]["gbps"] for n in names]
    labels = {"reduce_naive": "naive\n(interleaved)",
              "reduce_shared": "shared\n(sequential)",
              "reduce_warp": "warp shuffle\n(NVRTC)",
              "cublas_sasum": "cuBLAS Sasum\nbaseline"}

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = ax.bar([labels[n] for n in names], gbps,
                  color=["#c44" if n != "cublas_sasum" else "#888" for n in names])
    for bar, v in zip(bars, gbps):
        ax.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Reduction ladder, 2^26 floats — median of 50 runs")
    ax.grid(True, axis="y", alpha=0.3)

    if "reduce_warp" in have and "cublas_sasum" in have:
        pct = 100.0 * have["cublas_sasum"]["median_ms"] / have["reduce_warp"]["median_ms"]
        ax.annotate(f"warp shuffle = {pct:.0f}% of cuBLAS Sasum (target ≥ 80%)",
                    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9)

    out = RESULTS / "reduction_chart.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    rows = load_all(RESULTS)
    if not rows:
        sys.exit(f"no JSON results in {RESULTS}/ — run `make bench` first")
    plot_bandwidth(rows)
    plot_reductions(rows)


if __name__ == "__main__":
    main()
