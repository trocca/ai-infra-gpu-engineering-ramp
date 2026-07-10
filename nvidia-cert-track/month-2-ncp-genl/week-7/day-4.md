# Week 7 · Day 4 — Serving-engine mechanics: continuous batching, paged attention, TensorRT-LLM

[← Master Plan](../../../MASTER-PLAN.md) · [Week 7 overview](plan.md) · [← previous day](day-3.md) · [next day →](day-5.md)

## Study block (2 h)

Still **GPU Acceleration & Optimization (14%)**, feeding directly into **Model Deployment (9%)** next week. Today: the two ideas that made GPU serving economical — continuous batching and paged attention — plus TensorRT-LLM. This is doubly load-bearing: it's exam material *and* the design spec for ferrum-serve, the Rust inference engine you build next week. Study it as "what will I implement".

### Static batching and why it dies

Naive serving batches requests, runs them together, and returns when **all** finish. Generation lengths vary wildly, so the batch runs at the pace of its slowest member: a sequence that finished at token 20 holds its slot doing padding-work until the 500-token neighbor completes. GPU utilization becomes hostage to **output-length variance**, and new requests queue until the whole batch drains → terrible TTFT under load.

### Continuous (in-flight) batching — iteration-level scheduling

The fix (introduced by **Orca**, OSDI 2022; adopted by vLLM, TensorRT-LLM "in-flight batching", Triton): the scheduler makes decisions **every iteration**, not every batch. Each decode step: finished sequences leave immediately, their slots are refilled by waiting requests *at the next step*, and prefill for newcomers is interleaved with ongoing decodes. Utilization stops depending on length variance. This is THE serving-throughput idea; if an exam question mentions "requests join and leave the running batch each step" or "no waiting for the slowest sequence", this is the answer.

**The two schedulers on a timeline — same four slots, wildly different utilization (█ = decoding, · = slot wasted):**

```
STATIC batching — batch returns when ALL finish; newcomers wait outside
slot1 █████████████████████████████████████  A (500 tok, sets the pace)
slot2 ████·································  B finished @ 20 → dead slot
slot3 ██████████····························  C
slot4 ████████████████······················  D
queue: E, F, G ──── blocked until the whole batch drains ────►

CONTINUOUS (in-flight) batching — scheduler decides EVERY iteration
slot1 █████████████████████████████████████  A
slot2 ████ E ████████ F ███████████ G █████   finished → refilled next step
slot3 ██████████ H ██████████████ I ███████
slot4 ████████████████ J █████████████████
```

### Paged attention — virtual memory for the KV cache

The KV cache is the second problem. Naive engines pre-allocate one **contiguous** region per sequence sized for max-seq-len. Two disasters: **internal fragmentation** (a 200-token chat reserved 4096 tokens' worth) and **external fragmentation** (variable-sized contiguous chunks shred the free pool). Real measurement from the vLLM paper: only ~20–40% of "used" KV memory held actual KV data.

**PagedAttention** copies the OS playbook: carve KV memory into **fixed-size blocks** (e.g. 16 tokens each), give each sequence a **block table** (logical block → physical block indirection), allocate on demand. Consequences:

**The block table in action — logical order lives in the table, physical blocks land anywhere, shared prefixes are ref-counted:**

```
physical KV pool (fixed 16-token blocks, allocated on demand):
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ …
  └────┴────┴────┴────┴────┴────┴────┴────┘

seq A block table: [P2, P5, P0]     ← logical order ≠ physical order
seq B block table: [P2, P5, P7]     ← same system prompt: shares P2, P5
                    └───┴── ref-count = 2 (prefix caching);
                            copy-on-write when A or B diverges
```

- Internal waste ≤ one partial block per sequence; external fragmentation gone (all blocks identical) → far more concurrent sequences per GB → the original 2–4× throughput wins.
- **Sharing**: two sequences can point to the *same* physical blocks with **ref-counts** — that's **prefix caching** (a common system prompt's KV computed once, reused by every request) and cheap beam/parallel sampling, with **copy-on-write** on divergence.
- Cost: the attention kernel must gather K/V through the indirection table (a custom kernel in vLLM); `--gpu-memory-utilization` pre-allocates the whole pool up front — you'll *see* that in the lab as ~85% VRAM regardless of model size.

### Chunked prefill and the scheduler's knobs

A 20k-token prompt's prefill is a multi-second compute-bound monolith; if it runs alone, every concurrent stream's ITL stalls. **Chunked prefill** splits long prompts into chunks and interleaves them with decode steps of running sequences → protects concurrent TTFT/ITL at slight cost to that one prompt's completion. Knobs to know conceptually: **max-num-seqs** (batch ceiling), **KV block size**, **gpu-memory-utilization** (pool size), plus preemption/eviction when the pool runs dry (recompute vs swap). Speculative decoding (Day 2) also plugs into serving here — draft + verify inside the batch loop. KV-cache offload to CPU/NVMe extends capacity for long-idle sessions.

### TensorRT-LLM — the compiled path

**TensorRT-LLM** is NVIDIA's peak-performance engine: an **ahead-of-time compiler** that takes a model + precision config and emits a serialized **engine** — kernels fused and selected for the exact GPU architecture, precision baked in (FP8, INT4 AWQ, NVFP4), TP/PP partitioning compiled in. Runtime adds in-flight batching, paged KV, chunked prefill, spec decode. The catches: **engines are GPU-arch-specific** (build for the target — an H100 engine won't run on an A100), rebuilds accompany config changes, and model-day-one support lags.

**vLLM vs TensorRT-LLM** — the exam-and-real-life trade-off, one line: *TRT-LLM = maximum performance on NVIDIA GPUs at the cost of a build step and flexibility; vLLM = flexibility, day-one model support, easy ops, very good (not peak) performance.* **NIM** (next week) packages either behind one OpenAI-compatible API with pre-optimized per-GPU profiles — that's the layer map you'll draw Monday.

Cross-ref your demo repo: this whole day is the layer *under* your vLLM/Dynamo/NIM demo. Booth answer = exam answer.

### Read next

- Skim now, run Monday: [lab-quantize-serve.md](../labs/lab-quantize-serve.md) — note the `# GPU blocks` log line you'll be asked to record; you now know exactly what it counts.
- PagedAttention/vLLM paper (Kwon et al., SOSP 2023), sections 1–4: <https://arxiv.org/abs/2309.06180> — required reading twice over: it's also next week's build spec.
- Orca (Yu et al., OSDI 2022) — skim for iteration-level scheduling: <https://www.usenix.org/conference/osdi22/presentation/yu>
- TensorRT-LLM overview docs — architecture + supported precisions: <https://nvidia.github.io/TensorRT-LLM/>

### Quick check

1. A serving engine shows low GPU utilization despite a full batch; traces show most sequences finished long ago while two long ones continue. Name the problem and the fix.
<details><summary>Answer</summary>Static batching — the batch drains at the slowest sequence's pace. Fix: continuous (in-flight) batching — iteration-level scheduling that releases finished sequences and admits new ones every decode step.</details>

2. What two kinds of fragmentation does paged attention eliminate, and what data structure does it introduce?
<details><summary>Answer</summary>Internal (over-reserved contiguous max-seq-len regions) and external (variable-size chunks shredding the free pool). It introduces fixed-size KV blocks plus a per-sequence block table (logical→physical indirection), like OS virtual-memory pages.</details>

3. How does paged attention make prefix caching possible?
<details><summary>Answer</summary>Blocks are shared by reference: multiple sequences' block tables point at the same physical blocks holding the common prefix's KV (e.g. a system prompt), with ref-counts and copy-on-write on divergence — compute the prefix once, reuse it for every request.</details>

4. When would you recommend TensorRT-LLM over vLLM, and what operational cost comes with it?
<details><summary>Answer</summary>When peak performance on fixed, known NVIDIA hardware matters (latency-critical production, FP8/NVFP4 on Hopper/Blackwell) and the model is stable. Cost: an ahead-of-time engine build per GPU architecture and per config change, plus slower support for brand-new models.</details>

## Build block (4 h)

**Today: weight-only INT8 (W8A16) end to end** — Monday's quantization theory becomes a kernel. [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md)

- `src/quant/pack.py`: per-output-channel symmetric INT8 — `scale = max|W_row| / 127`, quantize, plus a dequant reference. Quality metric: relative Frobenius error.
- `src/quant/matmul_w8a16.py`: Triton matmul loading INT8 weights + fp16 scales, dequantizing **in registers**, multiplying against fp16 activations — the W8A16 recipe (weights 8-bit, activations 16-bit).
- Evaluate: swap the linear layers of the week-05 model (or `Qwen/Qwen2.5-1.5B-Instruct`) to INT8; measure perplexity fp16 vs int8 on a held-out TinyStories/WikiText slice, plus weight-memory savings.
- **Definition of done:** perplexity within ~1% relative of fp16 at ~2× weight-memory saving; kernel tests green.
- Hint: keep the dequant multiply in the accumulator loop as `(w_int8.to(tl.float16) * scale)` per K-tile — dequantizing to a temporary fp16 buffer in HBM first would forfeit the entire bandwidth win you're demonstrating.

## Close the day (15 min)

- Anki: static vs continuous batching one-liner, paged attention (blocks + block table + ref-counts), chunked prefill purpose, TRT-LLM vs vLLM trade-off, "engines are arch-specific".
- One line in [notes.md](notes.md): your fp16-vs-int8 perplexity delta and memory saving.
- Log blockers. Tomorrow is lab day (2-GPU cloud instance) — confirm tonight that your RunPod/AWS account is ready so you don't burn study time on sign-up friction.
