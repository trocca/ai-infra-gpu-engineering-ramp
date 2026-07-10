# Week 7 · Day 3 — GPU acceleration fundamentals: Tensor Cores, roofline, FlashAttention

[← Master Plan](../../../MASTER-PLAN.md) · [Week 7 overview](plan.md) · [← previous day](day-2.md) · [next day →](day-4.md)

## Study block (2 h)

Domain switch: **GPU Acceleration & Optimization (14%)** — the exam's second-heaviest domain, and the one your build repo makes visceral. Today's single most important idea: **arithmetic intensity decides everything**. Once you can classify a workload as compute-bound or memory-bound, every optimization this week (fusion, FlashAttention, quantization, CUDA graphs, batching) stops being a list and becomes one story.

### Tensor Cores and mixed precision

**Tensor Cores** are dedicated matrix-multiply-accumulate (MMA) units — they consume small tiles of A and B and produce C = A·B + C in one hardware op, at throughput far beyond the general-purpose CUDA cores. Peak math throughput roughly **doubles each time precision halves**: FP16 → FP8 (Hopper/Ada) → FP4 (Blackwell). That's *why* quantization to FP8/NVFP4 speeds up compute-bound work, closing yesterday's loop.

Mixed-precision training: compute in **BF16**, accumulate in **FP32**. Why BF16 beats FP16 for training: BF16 keeps FP32's **8 exponent bits** (same dynamic range, fewer mantissa bits), so gradients don't underflow and you don't need FP16's **loss-scaling** machinery. Exam tell: "no loss scaling required" → BF16.

### Arithmetic intensity and the roofline

**Arithmetic intensity** = FLOPs performed per byte moved from memory. Every GPU has a **ridge point** = peak FLOPs ÷ memory bandwidth; below it you're **memory-bound** (runtime = bytes/bandwidth), above it **compute-bound** (runtime = FLOPs/peak). Example numbers for intuition (H100 SXM): ~990 TFLOPS BF16 dense ÷ ~3.35 TB/s HBM3 → ridge ≈ **300 FLOPs/byte**. A GEMM with a large batch dimension reuses each loaded weight across many rows → high intensity → compute-bound. A **GEMV** (matrix–vector, i.e. batch 1) does ~2 FLOPs per weight loaded → intensity ≈ 1 → hopelessly memory-bound; the GPU idles 99% of its math units waiting on HBM.

**The roofline — attainable performance is min(bandwidth × AI, peak math); which roof you hit is the whole diagnosis:**

```
attainable TFLOPS (log–log axes, H100 SXM numbers)

990 ┤· · · · · · · · · · ·╭───────────────────────  compute roof: peak BF16 math
    │                    ╱:
    │                   ╱ :          ● prefill / big-batch GEMM
    │    memory roof → ╱  :            AI ≫ 300 → compute-bound
    │  perf = AI ×    ╱   :            runtime = FLOPs / peak
    │  3.35 TB/s     ╱    :
    │               ╱     :
    │        ●     ╱      :
    │  decode GEMV        :
    │  AI ≈ 1 → memory-bound, ~99% of math units idle
    │  runtime = bytes / bandwidth
    └─────────┬───────────┴────────────────────────  arithmetic intensity (FLOPs/byte)
              1        ridge ≈ 300
    ◄── memory-bound ──┤├── compute-bound ──►
```

### Prefill vs decode — the asymmetry that runs the whole inference industry

- **Prefill**: all prompt tokens processed in parallel → big GEMMs → **compute-bound**, Tensor-Core-heavy. Determines **TTFT**.
- **Decode**: one token per step, sequentially; every step re-reads **all weights** plus the growing **KV cache** to produce one token → GEMV-shaped → **memory-bandwidth-bound**. Determines **ITL**.

Consequences you should now be able to derive rather than memorize: weight-only quantization ≈ decode speedup (fewer bytes/step); batching decode requests raises intensity (weights amortized across sequences) which is why servers batch aggressively; FP8/W8A8 helps prefill because it raises the *math* rate.

### Kernel fusion — the point is bytes, not FLOPs

Eager PyTorch runs softmax as several kernels: each one reads its input from HBM and writes its output back. For bandwidth-bound elementwise/reduction ops, **bytes moved IS the runtime**, so **fusing** them into one kernel — read `x` once, do everything in registers/SRAM, write `y` once — wins even though the FLOPs are identical. You proved this Monday with your own softmax/RMSNorm benchmarks. Fusion reduces memory traffic and kernel-launch count; it does not reduce arithmetic.

### FlashAttention — fusion's masterpiece

Standard attention materializes the **N×N score matrix** `S = QKᵀ` in HBM, reads it back for softmax, writes P, reads it again for `P·V` — **O(N²) memory traffic and storage**. FlashAttention computes *exact* attention **tiled in on-chip SRAM**:

1. Load a block of Q; stream blocks of K, V through SRAM.
2. For each KV block, compute the partial scores and use **online softmax** — maintain running max `m`, running denominator `l`, un-normalized accumulator `acc`; when a new block raises the max, rescale by `exp(m_old − m_new)` (you implemented exactly this yesterday).
3. Write only the final O block. The N×N matrix **never exists in HBM** → activation memory **O(N)**, traffic O(N²/SRAM) instead of O(N²) full round-trips.

**Where the bytes go — standard attention round-trips an N×N matrix through HBM three times; FlashAttention never lets it exist:**

```
STANDARD ATTENTION                      FLASHATTENTION
HBM traffic per head:                   HBM traffic per head:
  read  Q, K        write S (N×N)         read Q, K, V (tiled)   write O
  read  S           write P (N×N)         — S and P never touch HBM —
  read  P, V        write O
                                        on-chip SRAM, per Q block:
  N×N intermediates live in HBM         ┌──────────────────────────────┐
  → O(N²) memory + traffic              │ Q block (resident)           │
                                        │  ← stream K,V blocks through │
                                        │ online softmax: m, l, acc    │
                                        │ rescale by exp(m_old − m_new)│
                                        └──────────────────────────────┘
                                        → O(N) extra memory, exact result
```

Two exam-grade subtleties: it's **exact, not an approximation** (unlike sparse/linear attention); and it's usually *faster despite doing MORE FLOPs* (softmax rescaling, sometimes recomputation in backward) — because at attention's low arithmetic intensity, trading extra compute for avoided HBM traffic is a bargain. That inversion — "more FLOPs, less time" — is the roofline argument in one sentence.

### CUDA graphs

Decode loops launch hundreds of small kernels per token; at batch 1 each kernel is microseconds and **CPU launch overhead** becomes a real fraction of step time. **CUDA graphs** capture the whole kernel sequence once and replay it with a single launch, removing per-kernel CPU overhead. Serving engines (vLLM, TensorRT-LLM) capture decode steps as graphs; it helps the small-batch, launch-bound regime — not big prefill GEMMs.

### Hands-on (30 min, part of the study block)

Roofline intuition check on any GPU (your 5090 in WSL2, or Colab): time `torch.matmul(x, W)` with `W` 4096×4096 fp16, for `x` of batch 1 vs batch 64. Use proper timing (`torch.cuda.synchronize`, median of ~100 runs after warmup). Expected: near-identical latency — at low batch the op is memory-bound, so 64× more math is ~free until you cross the ridge. Record the two numbers and the implied FLOPs/byte in [notes.md](notes.md).

### Read next

- FlashAttention paper (Dao et al., 2022), sections 1–3 + Figure 1: <https://arxiv.org/abs/2205.14135>
- *Online normalizer calculation for softmax* (Milakov & Gimelshein, 2018) — half a page of math you now own: <https://arxiv.org/abs/1805.02867>
- "Making Deep Learning Go Brrrr From First Principles" (Horace He) — the best roofline-for-DL explainer on the internet: <https://horace.io/brrr_intro.html>
- NVIDIA docs: *GPU Performance Background* — the roofline/arithmetic-intensity chapter: <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>

### Quick check

1. Batch-1 decode of an 8B FP16 model on a GPU with 1 TB/s bandwidth: estimate the ceiling on tokens/s, ignoring KV cache.
<details><summary>Answer</summary>Each token re-reads all weights: 16 GB/step → 1000 GB/s ÷ 16 GB ≈ **~60 tokens/s** ceiling. This back-of-envelope (bandwidth ÷ model bytes) is a classic; it's also why INT4 (~4.5 GB) raises the ceiling ~3–4×.</details>

2. Why does BF16 training not need loss scaling while FP16 training does?
<details><summary>Answer</summary>BF16 keeps FP32's 8-bit exponent → same dynamic range, so small gradients don't underflow to zero. FP16 has a 5-bit exponent and a much smaller representable range, so gradients must be scaled up before backprop and back down after — loss scaling.</details>

3. FlashAttention performs more FLOPs than standard attention yet runs faster. Resolve the paradox in two sentences.
<details><summary>Answer</summary>Attention at long N is memory-bound: runtime is dominated by HBM traffic, not math. FlashAttention eliminates the O(N²) score-matrix round-trips by tiling in SRAM, and the extra rescaling/recompute FLOPs are nearly free because the math units were idle anyway.</details>

4. Which regime do CUDA graphs help, and why don't they speed up prefill much?
<details><summary>Answer</summary>Small-batch decode, where per-kernel CPU launch overhead is comparable to kernel runtime — capturing the step as one graph removes that overhead. Prefill runs a few large, long-running GEMMs where launch overhead is negligible.</details>

## Build block (4 h)

**Today: FlashAttention forward, part 2 — causal masking, arbitrary sizes, autotuned, benchmarked.** [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md)

- Add **causal masking**: skip KV blocks entirely above the diagonal (never even load them — that's half the FLOPs), and apply the intra-block triangular mask with `-inf` before the online-softmax update on the diagonal blocks.
- Generalize to arbitrary N and head dims; add boundary masks; autotune `BLOCK_M`/`BLOCK_N`/`num_warps`.
- **Definition of done:** `pytest tests/test_flash.py` green (oracle: SDPA, ≤ 1e-2 abs fp16, causal and non-causal); benchmark vs SDPA-flash and the math backend across seq 256→8k; produce the **memory plot** — math backend peak memory O(N²) vs yours O(N). That plot is the week's money figure.
- Hint: for causal + boundary correctness, test odd sizes (N=257, N=1000) — power-of-two-only testing hides mask bugs.
- Acceptance framing from the brief: beat the math backend at seq ≥ 2k; staying within ~2–4× of SDPA's flash backend is *expected* — report the factor honestly.

## Close the day (15 min)

- Anki: ridge-point formula, prefill/decode table, "fusion saves bytes not FLOPs", FlashAttention three-liner (tiles + online softmax + exact + O(N)), BF16-vs-FP16, CUDA graphs.
- One line in [notes.md](notes.md): your batch-1-vs-64 matmul numbers and what they imply.
- Log blockers; tomorrow's kernel (W8A16 matmul) reuses today's Triton matmul muscle.
