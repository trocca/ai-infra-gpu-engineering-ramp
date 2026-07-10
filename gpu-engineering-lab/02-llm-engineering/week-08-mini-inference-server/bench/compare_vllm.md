# Methodology — ferrum-serve vs real vLLM (COMPLETE, follow exactly)

The comparison is only worth publishing if it is fair. Fair means: same GPU, same
weights, same prompts, same sampling parameters, same measurement client, same
machine state — and the architectural caveat stated up front. This document pins
all of it.

## 0. The honesty caveat (goes verbatim-ish into RESULTS.md)

ferrum-serve integrates paged KV via **gather-into-contiguous-KV per step**; vLLM
uses **true paged-attention kernels** that index block tables directly, plus CUDA
graphs, chunked prefill, and prefix caching. The throughput gap therefore measures
"a week of Rust + Candle vs years of CUDA engineering", not "Rust vs Python".
Say so, in those words or better.

## 1. What is compared

| | ferrum-serve | vLLM |
|---|---|---|
| Server | `cargo run --release --features cuda -- --port 8000` | `vllm serve <MODEL> --port 8001` |
| Weights | `Qwen/Qwen2.5-1.5B-Instruct` (safetensors via hf-hub) | identical HF repo |
| Client | `bench/loadtest.py` | `bench/loadtest.py` (add `--model <MODEL>`) |

The Python load-test client is kept on purpose: the client's language is irrelevant
(it only speaks HTTP/SSE), and using the IDENTICAL client for both servers is what
makes it a referee.

## 2. Model parity

Both servers load the same HF snapshot of `Qwen/Qwen2.5-1.5B-Instruct` (pin the
same `--revision`), so weight parity is automatic. Precision: run ferrum-serve in
bf16 and pin vLLM to the same (`--dtype bfloat16`). Sampling parity: same
`max_tokens` (128), same temperature (0.8), same prompt pool (the 8 prompts baked
into `loadtest.py`, seeded shuffle).

If you use gated `meta-llama/Llama-3.2-1B-Instruct` instead, both servers need the
HF token; note the model swap in RESULTS.md.

## 3. Machine state (record ALL of this next to the results JSON)

Before EACH server's run, capture:

```bash
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.sm,temperature.gpu \
           --format=csv | tee bench/results/machine_$(date +%s).txt
rustc --version && cargo --version
python -c "import vllm; print(vllm.__version__)"
```

- Plugged in, same Windows power profile, laptop on a hard surface, fans audible.
- No other GPU processes (`nvidia-smi` shows only the server under test).
- Let the GPU cool between the two servers' runs; alternate order across repeats.

## 4. Protocol

1. Build once: `cargo build --release --features cuda` (build time is NOT part of
   any latency metric — but note it once in RESULTS.md for honesty).
2. Start server A. Wait for `/health` (ferrum-serve) or a first successful
   completion (vLLM).
3. Closed-loop: `--concurrency 1 2 4 8 16 --requests-per-point 50 --max-tokens 128`.
4. Open-loop: `--rates 0.5 1 2 4 --duration 60`.
5. Stop server A completely (free VRAM), cool down, start server B, repeat with
   the IDENTICAL client command (plus `--model` for vLLM).
6. vLLM flags: defaults plus `--dtype bfloat16`, and pin `--max-model-len` to
   ferrum-serve's max context so neither side buys headroom the other lacks.
   Record both full command lines.

## 5. Metrics and the headline number

- **TTFT** — request sent → first streamed token (median + p95).
- **ITL** — mean gap between consecutive tokens within a request (median + p95).
- **Throughput** — aggregate output tok/s across all concurrent streams.
- **Headline**: `ferrum-serve throughput / vLLM throughput × 100` at closed-loop
  concurrency 8, max_tokens 128. One number, stated with its conditions and the
  §0 caveat.

## 6. Reporting rules

- Medians AND p95 — serving is a tails business.
- Errors/timeouts per point reported WITH the point, never dropped.
- Expected outcome, written in advance so nobody is tempted to fudge: vLLM wins
  decisively. Single-digit-to-low-double-digit % of vLLM throughput is a normal,
  publishable result for a first gather-based engine. The deliverable is the honest
  number plus the gap ANALYSIS — which vLLM optimization buys what.

## 7. Rust-only metrics (feed week 11's Kubernetes story)

Where a Rust serving binary structurally wins, measure it — these numbers are the
other half of the capstone's argument and go in RESULTS.md as a table:

| Metric | ferrum-serve | vLLM | How to measure |
|---|---|---|---|
| Stripped binary / install size | | | see below |
| Docker image size | | | see below |
| Cold start → first token | | | see below |
| Idle RSS after model load | | | `ps -o rss= -p <pid>` |

**Stripped binary size** — strip a COPY (the release profile keeps `debug = true`
for Nsight, per `setup/rust-cuda-toolchain.md`):

```bash
cargo build --release --features cuda
cp target/release/ferrum-serve /tmp/ferrum-stripped && strip /tmp/ferrum-stripped
ls -lh /tmp/ferrum-stripped
# vLLM's equivalent is an install footprint, not a binary — inside vLLM's venv:
du -sh $(python -c "import site; print(site.getsitepackages()[0])")
```

**Docker image size** — multi-stage build: `FROM rust:1-slim` builder stage →
`FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04` runtime stage with just the binary
(`COPY --from=builder`). Compare `docker images` against the official
`vllm/vllm-openai` tag you tested. Report compressed and uncompressed sizes.
Exclude model weights from BOTH images (mount them) so you compare software, not
safetensors.

**Cold start → first token** — the K8s pod-churn number. For each server, with
weights already in the local HF cache (you are measuring process + framework
startup, not the network):

```bash
start=$(date +%s.%N)
<server start command> &
until curl -sf -X POST localhost:PORT/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"hi","max_tokens":1,"stream":false}' >/dev/null 2>&1; do
  sleep 0.2
done
echo "cold start to first token: $(echo "$(date +%s.%N) - $start" | bc)s"
```

Run each 5 times, report the median. Expect ferrum-serve to win this table by a
lot — that is the structural argument for Rust at the serving layer, and it is
honest to make it precisely BECAUSE §0 concedes the kernel gap.
