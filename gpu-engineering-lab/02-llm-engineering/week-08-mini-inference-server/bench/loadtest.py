"""COMPLETE load-test client — do not edit (extend if you like).

Works against BOTH servers (this is the point — same referee for both):
  - mini-vllm:  events look like  data: {"text": "..."}
  - vLLM (OpenAI API):            data: {"choices": [{"text": "..."}], ...}
    (start with:  vllm serve <model> --port 8001 ; pass --model <model>)

Modes:
  closed-loop: K concurrent users, each fires its next request immediately
               after the previous finishes (measures capacity).
  open-loop:   Poisson arrivals at rate lambda req/s, independent of server
               speed (measures behavior under a fixed offered load — reveals
               queueing collapse that closed-loop hides).

Metrics per request: TTFT (first token), ITL (mean inter-token latency),
E2E latency, output tokens. Aggregates: median + p95, total output tok/s.

Usage:
  python bench/loadtest.py --url http://localhost:8000 --mode closed \
      --concurrency 1 2 4 8 16 --requests-per-point 50 --out bench/results/mine.json
  python bench/loadtest.py --url http://localhost:8001 --model Qwen/Qwen2.5-1.5B-Instruct \
      --mode open --rates 0.5 1 2 4 --duration 60 --out bench/results/vllm_open.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Once upon a time there was a little robot who",
    "The three friends walked into the forest and found",
    "Every morning, the baker on Elm Street would",
    "In a small village by the sea, a young girl discovered",
    "The old lighthouse keeper had a secret:",
    "When the rain finally stopped, the children ran outside to",
    "Deep in the library, behind the tallest shelf,",
    "The spaceship landed quietly in the wheat field and",
]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(p / 100 * len(xs)) - 1))
    return xs[idx]


async def one_request(client: httpx.AsyncClient, url: str, model: str | None,
                      max_tokens: int) -> dict:
    body = {"prompt": random.choice(PROMPTS), "max_tokens": max_tokens,
            "temperature": 0.8, "stream": True}
    if model:
        body["model"] = model
    t_start = time.perf_counter()
    token_times: list[float] = []
    n_tokens = 0
    try:
        async with client.stream("POST", url + "/v1/completions",
                                 json=body, timeout=300) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                text = (chunk.get("text")
                        or (chunk.get("choices") or [{}])[0].get("text", ""))
                if text:
                    token_times.append(time.perf_counter())
                    n_tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return {"ok": False, "error": repr(e)}
    t_end = time.perf_counter()
    if n_tokens == 0:
        return {"ok": False, "error": "no tokens"}
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    return {"ok": True,
            "ttft_s": token_times[0] - t_start,
            "itl_mean_s": statistics.mean(itls) if itls else 0.0,
            "e2e_s": t_end - t_start,
            "tokens": n_tokens}


def aggregate(records: list[dict], wall_s: float) -> dict:
    ok = [r for r in records if r.get("ok")]
    errors = len(records) - len(ok)
    if not ok:
        return {"ok": 0, "errors": errors}
    ttfts = [r["ttft_s"] for r in ok]
    itls = [r["itl_mean_s"] for r in ok]
    e2es = [r["e2e_s"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    return {
        "ok": len(ok), "errors": errors, "wall_s": wall_s,
        "ttft_median_s": statistics.median(ttfts), "ttft_p95_s": percentile(ttfts, 95),
        "itl_median_s": statistics.median(itls), "itl_p95_s": percentile(itls, 95),
        "e2e_median_s": statistics.median(e2es), "e2e_p95_s": percentile(e2es, 95),
        "output_tok_per_s": total_tokens / wall_s,
        "req_per_s": len(ok) / wall_s,
    }


async def closed_loop(url: str, model: str | None, concurrency: int,
                      total_requests: int, max_tokens: int) -> dict:
    records: list[dict] = []
    started = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        # warmup (not recorded)
        await asyncio.gather(*(one_request(client, url, model, max_tokens)
                               for _ in range(min(concurrency, 4))))

        t0 = time.perf_counter()

        async def worker():
            nonlocal started
            while True:
                async with lock:
                    if started >= total_requests:
                        return
                    started += 1
                records.append(await one_request(client, url, model, max_tokens))

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return {"mode": "closed", "concurrency": concurrency,
            **aggregate(records, wall)}


async def open_loop(url: str, model: str | None, rate: float,
                    duration_s: float, max_tokens: int) -> dict:
    records: list[dict] = []
    tasks: list[asyncio.Task] = []
    async with httpx.AsyncClient() as client:
        await one_request(client, url, model, max_tokens)  # warmup
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration_s:
            tasks.append(asyncio.create_task(
                one_request(client, url, model, max_tokens)))
            await asyncio.sleep(random.expovariate(rate))  # Poisson arrivals
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        records.extend(results)
    return {"mode": "open", "rate_req_per_s": rate, **aggregate(records, wall)}


def plot(points: list[dict], out_json: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    closed = [p for p in points if p.get("mode") == "closed" and p.get("ok")]
    if not closed:
        return
    xs = [p["concurrency"] for p in closed]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, [p["ttft_median_s"] * 1e3 for p in closed], "o-", label="median")
    ax1.plot(xs, [p["ttft_p95_s"] * 1e3 for p in closed], "o--", label="p95")
    ax1.set_xlabel("concurrency"); ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to first token"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(xs, [p["output_tok_per_s"] for p in closed], "o-", color="tab:green")
    ax2.set_xlabel("concurrency"); ax2.set_ylabel("output tok/s (aggregate)")
    ax2.set_title("Throughput"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    png = out_json.with_suffix(".png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default=None,
                   help="model name for OpenAI-style servers (vLLM needs it)")
    p.add_argument("--mode", choices=["closed", "open"], default="closed")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests-per-point", type=int, default=50)
    p.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2, 4])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", default="bench/results/loadtest.json")
    args = p.parse_args()
    random.seed(0)

    points: list[dict] = []
    if args.mode == "closed":
        for c in args.concurrency:
            pt = await closed_loop(args.url, args.model, c,
                                   args.requests_per_point, args.max_tokens)
            points.append(pt)
            print(json.dumps(pt, indent=2))
    else:
        for r in args.rates:
            pt = await open_loop(args.url, args.model, r,
                                 args.duration, args.max_tokens)
            points.append(pt)
            print(json.dumps(pt, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"url": args.url, "model": args.model, "max_tokens": args.max_tokens,
         "points": points}, indent=2))
    print(f"wrote {out}")
    plot(points, out)


if __name__ == "__main__":
    asyncio.run(main())
