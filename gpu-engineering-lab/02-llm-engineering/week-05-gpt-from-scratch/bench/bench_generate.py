"""COMPLETE benchmark harness — do not edit (extend if you like).

Measures generation tok/s with vs without KV cache across sequence lengths.
Honest-benchmark rules: warmup runs, median of >= 50 timed runs,
torch.cuda.synchronize() around every timer. Outputs JSON + a PNG plot.

Usage (random-weight model is fine for speed measurement; pass --ckpt to use
your trained one):
    python bench/bench_generate.py --runs 50 --out bench/results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import generate            # noqa: E402
from src.model import CONFIGS, GPT           # noqa: E402


def time_generate(model, prompt, new_tokens, use_cache, runs, warmup, device):
    times = []
    for i in range(warmup + runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate(model, prompt.clone(), new_tokens,
                 temperature=0.0, use_kv_cache=use_cache)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    med = statistics.median(times)
    return {
        "median_s": med,
        "p10_s": sorted(times)[max(0, int(0.10 * len(times)) - 1)],
        "p90_s": sorted(times)[int(0.90 * len(times))],
        "tok_per_s": new_tokens / med,
        "runs": len(times),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="d12", choices=sorted(CONFIGS))
    p.add_argument("--ckpt", default=None, help="optional trained checkpoint")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--gen-lens", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--out", default="bench/results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CONFIGS[args.config]
    model = GPT(cfg).to(device).eval()
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)

    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)

    results = {
        "config": args.config,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "runs": args.runs,
        "warmup": args.warmup,
        "prompt_len": args.prompt_len,
        "points": [],
    }
    for n in args.gen_lens:
        total = args.prompt_len + n
        if total > cfg.max_seq_len:
            print(f"skip gen_len={n}: {total} > max_seq_len={cfg.max_seq_len}")
            continue
        naive = time_generate(model, prompt, n, False, args.runs, args.warmup, device)
        cached = time_generate(model, prompt, n, True, args.runs, args.warmup, device)
        point = {"gen_len": n, "naive": naive, "kv_cache": cached,
                 "speedup": naive["median_s"] / cached["median_s"]}
        results["points"].append(point)
        print(f"gen_len={n:4d}  naive={naive['tok_per_s']:8.1f} tok/s  "
              f"kv={cached['tok_per_s']:8.1f} tok/s  speedup={point['speedup']:.2f}x")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "generate_bench.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [pt["gen_len"] for pt in results["points"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(xs, [pt["naive"]["tok_per_s"] for pt in results["points"]],
                 "o-", label="naive (full recompute)")
        ax1.plot(xs, [pt["kv_cache"]["tok_per_s"] for pt in results["points"]],
                 "o-", label="KV cache")
        ax1.set_xlabel("generated tokens"); ax1.set_ylabel("tok/s (median)")
        ax1.set_title(f"Generation throughput ({results['device']})"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(xs, [pt["speedup"] for pt in results["points"]], "o-", color="tab:green")
        ax2.axhline(5.0, ls="--", color="gray", label="acceptance: 5x @ 512")
        ax2.set_xlabel("generated tokens"); ax2.set_ylabel("speedup (x)")
        ax2.set_title("KV-cache speedup"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        png = out_dir / "generate_bench.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed — JSON only")


if __name__ == "__main__":
    main()
