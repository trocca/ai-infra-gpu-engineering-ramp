#!/usr/bin/env python3
"""Plot the SGEMM ladder (COMPLETE — do not edit, extend if you like).

Reads results/sgemm_ladder.json (emitted by `cargo run --release --bin
sgemm_ladder`) and writes:
  results/ladder_chart.png  — grouped GFLOPS bars per rung at each size,
                              plus a %-of-cuBLAS line for the largest size
  results/ladder_scaling.png — GFLOPS vs size, one line per rung

Usage: python bench/plot_ladder.py [results/sgemm_ladder.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sgemm_ladder.json")

RUNG_ORDER = ["1_naive", "2_coalesced", "3_smem_tiled", "4_1d_blocktile",
              "5_2d_blocktile", "6_vectorized", "7_wmma", "cublas"]
RUNG_LABEL = {
    "1_naive": "1 naive", "2_coalesced": "2 coalesced", "3_smem_tiled": "3 smem tile",
    "4_1d_blocktile": "4 1D regtile", "5_2d_blocktile": "5 2D regtile",
    "6_vectorized": "6 float4", "7_wmma": "7 wmma", "cublas": "cuBLAS",
}


def main() -> None:
    if not PATH.exists():
        sys.exit(f"{PATH} not found — run `make bench` first")
    rows = json.loads(PATH.read_text())

    # gflops[name][size] = value
    gflops: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        gflops[r["name"]][r["size"]] = r["gflops"]

    names = [n for n in RUNG_ORDER if n in gflops]
    sizes = sorted({s for d in gflops.values() for s in d})
    if not names or not sizes:
        sys.exit("no usable rows in JSON")
    big = max(sizes)

    # ---- chart 1: bars at the largest size + % of cuBLAS ------------------
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    vals = [gflops[n].get(big, 0.0) for n in names]
    colors = ["#888" if n == "cublas" else "#2b8" for n in names]
    bars = ax.bar([RUNG_LABEL[n] for n in names], vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:,.0f}", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"SGEMM ladder at {big}x{big} — RTX 5090 Laptop (sm_120), median of 50 runs")
    ax.grid(True, axis="y", alpha=0.3)

    if "cublas" in gflops and gflops["cublas"].get(big):
        ax2 = ax.twinx()
        ref = gflops["cublas"][big]
        pct = [100.0 * gflops[n].get(big, 0.0) / ref for n in names]
        ax2.plot([RUNG_LABEL[n] for n in names], pct, marker="o",
                 color="tab:red", linewidth=1.2, label="% of cuBLAS")
        ax2.axhline(50, linestyle=":", color="tab:red", alpha=0.5)
        ax2.set_ylabel("% of cuBLAS", color="tab:red")
        ax2.set_ylim(0, 110)
        ax2.legend(loc="upper left", fontsize=8)

    out = PATH.parent / "ladder_chart.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # ---- chart 2: scaling across sizes ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for n in names:
        xs = [s for s in sizes if s in gflops[n]]
        ys = [gflops[n][s] for s in xs]
        style = {"linewidth": 2.2, "color": "#444"} if n == "cublas" else {"linewidth": 1.3}
        ax.plot(xs, ys, marker="o", label=RUNG_LABEL[n], **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("matrix size N (square, fp32)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("SGEMM ladder scaling — median of 50 runs after warmup")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out = PATH.parent / "ladder_scaling.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
