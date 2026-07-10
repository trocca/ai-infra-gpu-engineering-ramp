# Lab 2 — Serve FP16 vs Quantized with vLLM and Measure the Difference

Serve a small model with vLLM at FP16, benchmark it, then serve a 4-bit AWQ checkpoint of the same model and benchmark again. See quantization show up as real TTFT/throughput/VRAM numbers.

| | |
|---|---|
| GPU | 1 × L4 / A10G / RTX 4090 (24 GB). 16 GB works for the 1.5B model used here. |
| Est. time | 60–90 min |
| Est. cost | **under $2** at $0.40–1.10/hr |
| Exam domains served | Model Optimization (17%), GPU Acceleration (14%), Deployment (9%), Monitoring (7%) |

**Models (both ungated):**
- FP16 baseline: `Qwen/Qwen2.5-1.5B-Instruct`
- Quantized: `Qwen/Qwen2.5-1.5B-Instruct-AWQ` (official pre-quantized 4-bit AWQ checkpoint — we serve a ready artifact rather than spending lab time calibrating; quantizing it yourself is the optional extension at the end)

*Gated alternative:* Llama-3.2 variants exist in AWQ too but are license-gated; Qwen keeps this friction-free.

> **FP8 note:** L4/4090 are Ada (compute capability 8.9) and support FP8; on Ampere A10G, FP8 checkpoints run via weight-only emulation (Marlin kernels) — legal but less impressive. AWQ INT4 behaves consistently across all three, which is why this lab uses it. FP8 is the optional extension.

## 0. Prerequisites

Same instance profile as Lab 1. Fresh venv recommended:

```bash
python3 -m venv ~/vllm-lab && source ~/vllm-lab/bin/activate
pip install --upgrade pip
pip install "vllm>=0.8" pandas datasets
nvidia-smi   # confirm GPU idle, note total VRAM
```

## 1. Serve the FP16 baseline

Terminal 1:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

**Expected:** log lines showing model load, then something like
`INFO ... # GPU blocks: <N>` (KV-cache blocks — write N down) and
`INFO: Uvicorn running on http://0.0.0.0:8000`.

Terminal 2 — smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "One sentence: what is paged attention?"}],
       "max_tokens": 60}' | python3 -m json.tool
```

**Observe in `nvidia-smi`:** VRAM is ~85% of the card *regardless of model size* — vLLM pre-allocates the KV-cache pool (that's `--gpu-memory-utilization`). The model is ~3 GB; the rest is KV blocks. This is paged attention's arena.

## 2. Benchmark the baseline

vLLM ships a benchmark CLI:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 8 \
  --save-result --result-filename fp16.json
```

> If your vLLM version lacks `vllm bench serve` (older releases), use the equivalent script: `python -m vllm.entrypoints.cli.main bench serve ...`, or the classic `benchmarks/benchmark_serving.py` from the vLLM repo — or the pure-Python fallback below, which needs nothing but `pip install openai`:

```python
# bench_fallback.py -- crude but honest TTFT/throughput probe
import time, statistics, concurrent.futures
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # change for the AWQ run

def one(i):
    t0 = time.perf_counter(); first = None; n = 0
    stream = client.chat.completions.create(model=MODEL, stream=True, max_tokens=128,
        messages=[{"role": "user", "content": f"Explain topic {i} of Kubernetes in detail."}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            n += 1
            if first is None: first = time.perf_counter() - t0
    return first, n, time.perf_counter() - t0

t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(one, range(100)))
wall = time.perf_counter() - t0
ttfts = sorted(r[0] for r in res); toks = sum(r[1] for r in res)
print(f"TTFT  p50={ttfts[50]*1000:.0f}ms  p99={ttfts[98]*1000:.0f}ms")
print(f"Output throughput: {toks/wall:.0f} tok/s   ({toks} tokens / {wall:.1f}s, 16-way concurrency)")
```

**Expected (vllm bench):** a summary block with `Request throughput (req/s)`, `Output token throughput (tok/s)`, `Mean/Median/P99 TTFT (ms)`, `Mean/Median/P99 ITL (ms)`. On an L4 at this size expect very roughly 700–1500 output tok/s aggregate; absolute numbers don't matter — the *delta vs AWQ* does.

Record: TTFT p50/p99, ITL p99, output tok/s, and the `# GPU blocks` from step 1. Then stop the server (Ctrl-C in Terminal 1).

## 3. Serve the AWQ model

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

**Observe in logs:** vLLM auto-detects `quantization: awq` (it may report `awq_marlin` — the fast Marlin kernel path; that's good). Note the new `# GPU blocks` — with ~2.2 GB of weights freed, the KV pool got **bigger**: same card, more concurrency headroom.

Re-run the same benchmark with `--model Qwen/Qwen2.5-1.5B-Instruct-AWQ`, save to `awq.json`.

## 4. Compare — what you should see and why

| Metric | Expected direction | Why |
|---|---|---|
| Weights VRAM | ~4× smaller (3.1 → ~1.1 GB) | 4-bit weights + scales |
| KV blocks | more | freed VRAM becomes KV pool |
| ITL / decode tok/s at low concurrency | better | decode is bandwidth-bound; 4× fewer weight bytes per step |
| TTFT | similar or slightly worse | prefill is compute-bound; AWQ dequantizes to FP16 for the matmul, so FLOPs don't drop |
| Max sustainable req rate | higher | more KV → more concurrent sequences before preemption |
| Quality | tiny degradation | AWQ protects salient channels; spot-check a few generations side by side |

If ITL barely improves: at 1.5B the weights are small, so at *high* concurrency you're compute/overhead-bound — drop `--request-rate` to 2 or run single-stream and the decode gain becomes visible. That observation itself is roofline thinking; write it down.

## 5. Optional extensions (+30–45 min)

- **Serve your Lab 1 merged model:** `vllm serve ./qwen-merged --served-model-name qwen-ft` — connects the two labs.
- **FP8 on Ada (L4/4090):** quantize yourself with LLM Compressor: `pip install llmcompressor`, then follow its FP8 W8A8 one-shot recipe for Qwen2.5-1.5B-Instruct and serve the output dir. Compare TTFT too — FP8 *does* accelerate prefill math, unlike AWQ.
- **Quantize with AutoAWQ yourself** instead of using the prebuilt checkpoint (needs calibration data, ~10–20 min at this size).

## 6. Cleanup

```bash
# Ctrl-C the server, then:
rm -rf ~/.cache/huggingface ~/vllm-lab fp16.json awq.json
```

**Terminate the instance.**

## What you should now be able to explain (exam mapping)

- Weight-only 4-bit helps decode/memory, not prefill FLOPs — and you have your own numbers proving it (Model Optimization 17%)
- gpu-memory-utilization, KV blocks, paged attention as a memory arena (GPU Acceleration 14%)
- TTFT vs ITL vs throughput and reading P99s from a benchmark (Monitoring 7%)
- Quantized serving as a deployment decision: same API, smaller footprint, bigger batch (Deployment 9%)
