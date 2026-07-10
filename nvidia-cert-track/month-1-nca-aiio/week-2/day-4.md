# Week 2 · Day 4 — GPU memory and precision formats

[← Master Plan](../../../MASTER-PLAN.md) · [Week 2 overview](plan.md) · [← previous day](day-3.md) · [next day →](day-5.md)

## Study block (2 h)

Flashcards first (15 min). Today's two topics — HBM and precision formats — are really one topic: **bytes are the scarce resource**, and both capacity and the speed of moving them decide what a GPU can do. Direct Domain 1 crossover: precision is the training-vs-inference axis you tabled in week 1.

### HBM: what it is and why it exists

**HBM (High Bandwidth Memory)** is DRAM stacked vertically in 3D and placed *on the GPU package* next to the die, connected through an extremely wide interface (thousands of bits vs a DDR DIMM's 64). Result: multi-TB/s bandwidth at reasonable power. The ladder to know: system **DDR5** (~tens–hundreds of GB/s per socket) < **GDDR** (consumer/graphics cards, hundreds of GB/s) < **HBM** (datacenter: H100 HBM3 ~**3.35 TB/s**, H200 HBM3e ~**4.8 TB/s**, B200 ~8 TB/s). Trade-off: HBM is expensive and capacity-limited per stack — which is exactly why "how much HBM" is the headline differentiator between GPU SKUs (80 → 141 → 192 GB).

**The memory-bandwidth ladder — capacity gets cheaper to the left, bytes/s to the right:**

```
  bandwidth ──────────────────────────────────────────────────────────▶

  DDR5 (CPU socket)        GDDR (consumer GPU)      HBM (datacenter GPU)
  tens–hundreds GB/s       hundreds of GB/s         H100  ~3.35 TB/s
  DIMMs, TBs capacity      on-card                  H200  ~4.8  TB/s
  cheap per GB             mid cost                 B200  ~8    TB/s
                                                    80–192 GB, on-package,
                                                    expensive per GB
```

### Why bandwidth, not just capacity, is the bottleneck

For every parameter an LLM uses to generate **one token**, the weight must be *read from memory*. In the decode phase, batch-1 inference does relatively little math per byte read — **arithmetic intensity is low**, so the GPU finishes the math faster than memory can deliver operands. The workload is **memory-bandwidth-bound**: a rough speed-of-light estimate for single-stream decode is `bandwidth / model bytes` (e.g., ~3.35 TB/s ÷ 140 GB of FP16 weights ≈ ~24 tokens/s ceiling on one H100 — before any cleverness). This is why H200 exists, why quantization pays *speed* dividends (fewer bytes per parameter = proportionally more tokens/s), and why yesterday's coalescing lab matters: unused bytes in a memory transaction are wasted bandwidth.

**Capacity** drives GPU *count*: rule of thumb, FP16/BF16 weights = **2 bytes per parameter** (+ KV cache + activations). 70B params ≈ 140 GB → two H100s, one H200 barely, one B200 comfortably. Training multiplies that by ~8–10× (gradients, optimizer states, activations — week 1 Day 2).

### The precision zoo — build this table in `notes.md`

A float = sign + **exponent bits** (range) + **mantissa bits** (precision). Every format is a different split:

| Format | Bits | Exponent/mantissa | Introduced / used for |
|---|---|---|---|
| FP64 | 64 | 11/52 | HPC/scientific; irrelevant to DL perf |
| FP32 | 32 | 8/23 | Classic single precision; master weights |
| **TF32** | 19 (stored in 32) | 8/**10** | **Ampere+**. FP32's range, FP16's precision; Tensor Cores use it *transparently* for FP32 matmul — free ~×8 speedup, no code change |
| FP16 | 16 | **5**/10 | Half precision; small range → gradients underflow → needs **loss scaling** |
| **BF16** | 16 | **8**/7 | **Ampere+** Tensor Cores. FP32's full range, less precision — **no loss scaling needed** → preferred for training |
| **FP8** | 8 | 4/3 (E4M3) or 5/2 (E5M2) | **Hopper+**, managed by **Transformer Engine** (per-layer scaling, format choice) — LLM training + inference |
| INT8 | 8 | integer | Inference **quantization** (post-training or QAT); needs calibration |
| **FP4** | 4 | — | **Blackwell**, 2nd-gen Transformer Engine — inference frontier |

**Where the bits go — S = sign, E = exponent (range), M = mantissa (precision):**

```
 FP32      │S│EEEEEEEE│MMMMMMMMMMMMMMMMMMMMMMM│  1+8+23
 TF32      │S│EEEEEEEE│MMMMMMMMMM│               1+8+10  FP32 range, FP16 precision
 BF16      │S│EEEEEEEE│MMMMMMM│                  1+8+7   FP32 range → no loss scaling
 FP16      │S│EEEEE│MMMMMMMMMM│                  1+5+10  narrow range → loss scaling
 FP8 E4M3  │S│EEEE│MMM│                          1+4+3   more precision (weights)
 FP8 E5M2  │S│EEEEE│MM│                          1+5+2   more range (gradients)
```

The two memorization anchors: (1) **FP16 vs BF16 differ in where the 16 bits go** — FP16 spends them on precision (10-bit mantissa, 5-bit exponent), BF16 on range (8-bit exponent like FP32, 7-bit mantissa). Range is what training stability needs, hence BF16's win. (2) **Fewer bits = more throughput + less memory + less bandwidth per operand** — each halving roughly doubles Tensor Core throughput and halves memory traffic. Generation tags: **TF32/BF16 → Ampere, FP8 → Hopper, FP4 → Blackwell** (the exam's favorite triple).

### Exam traps and the pre-sales angle

- TF32 is *not* a 32-bit format despite the name — it's a 19-bit Tensor Core math mode applied to FP32 data automatically. "Which format accelerates existing FP32 code with no changes?" → TF32.
- "Which 16-bit format avoids loss scaling and why?" → BF16, because it keeps FP32's exponent range.
- Quantization (INT8/FP8/FP4) is an **inference** story on the exam; training answers should say mixed precision (BF16/FP16 compute + FP32 master weights, or FP8 with Transformer Engine on Hopper+).
- Customer angle: "your inference is bandwidth-bound; FP8 halves bytes-per-parameter vs FP16 — that's ~2× tokens/s *and* half the GPUs for the same fleet, before buying anything new." Precision is the cheapest hardware upgrade there is.

### Read next

- NVIDIA blog: "TensorFloat-32 in the A100 GPU" — the TF32 design rationale.
- NVIDIA blog/docs: FP8 + Transformer Engine post (search "Transformer Engine FP8") — how per-layer scaling keeps FP8 stable.
- H200 product page — read the memory-bandwidth pitch knowing why it's the pitch.

### Quick check

1. Why does LLM decode speed scale with memory bandwidth rather than TFLOPS?
<details><summary>Answer</summary>Generating each token requires streaming essentially all model weights from HBM while doing little math per byte (low arithmetic intensity), so memory delivery — not compute — is the limiting rate. Ceiling ≈ bandwidth ÷ model bytes.</details>

2. FP16 and BF16 are both 16 bits. What's the structural difference and the practical consequence for training?
<details><summary>Answer</summary>FP16: 5 exponent / 10 mantissa bits — more precision, narrow range → gradients underflow → requires loss scaling. BF16: 8 exponent / 7 mantissa — FP32's range, less precision → trains stably without loss scaling. BF16 is preferred for training.</details>

3. Match format to introducing architecture: TF32, FP8, FP4.
<details><summary>Answer</summary>TF32 → Ampere (A100), FP8 → Hopper (H100, with Transformer Engine), FP4 → Blackwell (B200, 2nd-gen Transformer Engine).</details>

4. A 70B model must serve from a single GPU. Name two distinct ways to make it fit, with the arithmetic.
<details><summary>Answer</summary>(a) Bigger memory: FP16 needs ~140 GB → H200 (141 GB) or B200 (192 GB). (b) Quantization: FP8/INT8 → ~70 GB fits an 80 GB H100; FP4 → ~35 GB. (Quantization also roughly doubles/quadruples bandwidth-bound decode speed.)</details>

## Build block (4 h)

**Today: profilers — turn "it's faster" into evidence.** [Project brief](../../../gpu-engineering-lab/01-foundations/week-02-cuda-basics/README.md)

- `nsys profile target/release/reduce_warp` — read the timeline: launch overhead vs kernel time vs memcpys.
- `ncu --set full` on all three reductions: compare Speed-of-Light %, the memory chart, and warp stall reasons.
- Definition of done: per reduction variant, a written answer to "bound by what?" with the metric that proves it (e.g., DRAM throughput % of speed-of-light, stall-reason breakdown).
- WSL2 note: `ERR_NVGPUCTRPERM` means enabling GPU performance counters for non-admin users on the *Windows* side (NVIDIA control panel → Developer).
- Hint: for `reduce_naive`, look at the warp-stall and shared-memory sections before the DRAM section — its problem (divergence + bank conflicts) is in-SM, not at the memory wall.

## Close the day (15 min)

- Anki: the precision table (one card per format: bits, split, generation, use), HBM ladder numbers, the bandwidth-bound decode argument; daily Domain 1 review.
- One line in [notes.md](notes.md): the hardest thing today.
- Log blockers (missing Nsight metrics or counter permissions — Friday's writeup needs them).
