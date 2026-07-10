# Week 7 · Day 1 — Quantization theory: formats, math, and what each actually speeds up

[← Master Plan](../../../MASTER-PLAN.md) · [Week 7 overview](plan.md) · [← previous day](../week-6/day-5.md) · [next day →](day-2.md)

## Study block (2 h)

This week carries the exam's two heaviest domains — **Model Optimization (17%)** and **GPU Acceleration & Optimization (14%)**, 31% combined. Today is the foundation everything else this week stands on: what quantization *is*, mathematically, and the one question the exam asks over and over in disguise — *which phase of inference does this technique speed up, and why?*

### The core math: scale and zero-point

Quantization maps continuous FP values onto a small integer (or low-bit float) grid. The affine scheme:

- **Quantize:** `q = clamp(round(x / s) + z, q_min, q_max)`
- **Dequantize:** `x̂ = s · (q − z)`

where `s` (scale, an FP number) sets the step size of the grid and `z` (zero-point, an integer) shifts it so that real-valued 0 maps exactly onto a grid point. Two flavors:

- **Symmetric** (`z = 0`): grid centered at zero. For INT8, `s = max|x| / 127`. Standard for **weights**, whose distributions are roughly zero-centered. This is exactly what you'll implement in today's build repo later this week (`pack.py`).
- **Asymmetric** (`z ≠ 0`): grid shifted to cover a lopsided range, e.g. post-ReLU activations that are never negative. Costs a little extra arithmetic; buys grid resolution.

**The mapping on a number line — symmetric INT8: a continuous FP16 range snapped onto 255 evenly spaced levels:**

```
FP16 values (symmetric, z = 0):            s = max|x| / 127

 −max|x|                     0                      +max|x|
    │                        │                         │
    ▼                        ▼                         ▼
 ───┼────┼────┼── … ──┼────┼────┼────┼── … ──┼────┼────┼───
  −127  −126        −2   −1    0    1    2          126  127   ← INT8 grid,
                                                                  step = s
quantize:  q = clamp(round(x/s) + z)      dequantize:  x̂ = s·(q − z)

one outlier stretches max|x| → s grows → normal values collapse
onto a handful of levels (why per-channel/per-group scales exist)
```

**Granularity** is the other axis: one `s` per tensor (cheap, fragile — one outlier stretches the grid and crushes everything else into a few levels), per **channel** (one scale per output row — near-free in a fused kernel, the practical default for weights), or per **group/block** (e.g. one scale per 128 weights — what AWQ/GPTQ checkpoints and NVFP4 use). Finer granularity = better accuracy, slightly more scale storage.

### The two decisions that define any quantization scheme

1. **What gets quantized?**
   - **Weight-only** (W8A16, W4A16): weights stored low-bit, dequantized to FP16 in registers before the matmul. Shrinks memory and *weight bandwidth* → speeds up **decode** (which is bandwidth-bound, re-reading every weight per token). Does **not** reduce FLOPs → prefill (compute-bound) barely moves. This is the trap distinction the exam loves.
   - **Weight + activation** (W8A8 INT8 or FP8): both sides of the matmul are low-bit, so the multiply itself runs on low-precision **Tensor Core** paths → 2× math throughput → speeds up **prefill and large-batch serving** too.
   **Decision 1 as a flow — trace WHY each branch speeds up what it speeds up; this is the exam's favorite disguise:**

   ```mermaid
   flowchart TB
       Q{"what is stored low-bit?"} --> WO["weight-only: W8A16 / W4A16"]
       Q --> WA["weights + activations: W8A8 INT8 / FP8"]
       WO --> WO1["fewer weight bytes moved per token"] --> WO2["DECODE speeds up — it is bandwidth-bound"]
       WO --> WO3["dequant to FP16 before matmul — FLOPs unchanged"] --> WO4["prefill / TTFT unchanged"]
       WA --> WA1["matmul itself runs on low-precision Tensor Cores — 2× math rate"] --> WA2["PREFILL and large-batch serving speed up too"]
   ```

2. **When does quantization happen?**
   - **PTQ** (post-training quantization): calibrate scales on a few hundred samples, no gradient updates. Cheap, the default. GPTQ/AWQ/SmoothQuant are all PTQ.
   - **QAT** (quantization-aware training): insert fake-quant ops during (re)training so the model learns around the rounding. Better at very low bits or for small models with little redundancy — at the cost of a training run and data.

### Formats and the hardware map (memorize this table)

| Format | Structure | Hardware math support | Typical use |
|---|---|---|---|
| INT8 | integer grid, needs calibration | Turing+ Tensor Cores | W8A8 serving, weight-only W8A16 |
| FP8 **E4M3** | 4 exp / 3 mantissa bits — precision over range | **Hopper/Ada and later** | weights + activations, inference default |
| FP8 **E5M2** | 5 exp / 2 mantissa — range over precision | Hopper/Ada+ | **gradients** in FP8 training |
| INT4 (AWQ/GPTQ) | weight-only, per-group scales | any (dequant in kernel) | 4-bit checkpoints for decode speed/memory |
| **NVFP4** | 4-bit float, per-block (16-value) FP8 scaling | **Blackwell** Tensor Cores | native 4-bit inference math |

INT8's LLM-specific problem: a handful of **activation channels** carry magnitudes ~100× the rest (emergent outliers in models >~6B). Per-tensor INT8 activations collapse. Fixes: LLM.int8() keeps outlier columns in FP16 (mixed decomposition, slow); **SmoothQuant** migrates the difficulty into the weights — divide activations by a per-channel factor, multiply the corresponding weight columns by it, mathematically identical output, both tensors now quantize cleanly to W8A8. FP8 largely sidesteps the issue because a floating-point grid has dynamic range per element — one reason FP8 is the "near-lossless, no-drama" inference format on Hopper/Ada.

### Memory math drill (do these by hand, they're free exam points)

Parameters × bytes-per-param: an **8B model** = 16 GB at FP16/BF16, **8 GB** at INT8, **~4–4.5 GB** at INT4 (plus per-group scales, a few %). Rule of thumb: batch-1 decode throughput scales roughly with bytes moved per token — every weight byte is re-read each step — so 4-bit weights ≈ **2–3× decode speedup** on bandwidth-bound workloads. Separate lever: **KV-cache quantization** (FP8 KV) halves cache bytes → longer contexts or more concurrent sequences on the same card. Weights and KV cache are independent memory budgets; the exam mixes them to see if you notice.

### NVIDIA tooling names to recognize

- **TensorRT Model Optimizer**: NVIDIA's PTQ/QAT toolkit — FP8, INT8 (SmoothQuant), INT4 AWQ, NVFP4 — exporting checkpoints for TensorRT-LLM (and vLLM).
- **vLLM** loads pre-quantized AWQ/GPTQ/FP8 checkpoints directly (you'll benchmark exactly this in [lab-quantize-serve](../labs/lab-quantize-serve.md), which runs week 8 day 1).

### Read next

- SmoothQuant paper (Xiao et al., 2022) — intro + Figure 1 tell the whole outlier story: <https://arxiv.org/abs/2211.10438>
- *FP8 Formats for Deep Learning* (Micikevicius et al., 2022) — the E4M3/E5M2 split: <https://arxiv.org/abs/2209.05433>
- TensorRT Model Optimizer docs — skim the supported-formats matrix: <https://nvidia.github.io/TensorRT-Model-Optimizer/>
- vLLM quantization docs — which checkpoint formats load out of the box: <https://docs.vllm.ai/en/latest/features/quantization/>

### Quick check

1. A team switches an 8B model from FP16 to W4A16 (AWQ). Decode gets ~2.5× faster, but TTFT on long prompts is unchanged. Why?
<details><summary>Answer</summary>W4A16 is weight-only: decode is memory-bandwidth-bound and moving 4× fewer weight bytes ≈ proportional speedup; prefill is compute-bound and the matmul still runs in FP16 after in-kernel dequantization, so FLOPs — and TTFT — don't change. To cut TTFT you need low-precision math: W8A8 INT8 or FP8.</details>

2. Which FP8 encoding is used for gradients during training, and why?
<details><summary>Answer</summary>E5M2 — 5 exponent bits give the extra dynamic range gradients need; E4M3 (more mantissa, more precision) is used for weights/activations.</details>

3. Per-tensor INT8 activation quantization collapses on a 13B LLM. What's happening and what's the targeted fix?
<details><summary>Answer</summary>Emergent activation outliers: a few channels are ~100× larger, stretching the single per-tensor scale until normal values round to nothing. SmoothQuant migrates the outlier difficulty into the weights via per-channel rescaling, making W8A8 viable — no retraining needed.</details>

4. NVFP4 checkpoints require which GPU generation for native Tensor Core math, and what makes NVFP4 more accurate than plain INT4?
<details><summary>Answer</summary>Blackwell. NVFP4 is a 4-bit *floating-point* format with fine-grained per-block (16-value) FP8 scale factors, so each small block gets its own dynamic range instead of one shared integer grid.</details>

## Build block (4 h)

**Today: fused softmax + RMSNorm Triton kernels** — your first real GPU kernels, and a hands-on proof of the fusion argument you'll study Wednesday. [Project brief](../../../gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md)

- Verify the toolchain first: `python -c "import triton; print(triton.__version__)"` inside WSL2.
- `src/softmax_triton.py`: row-wise fused softmax — one program per row: load once, max-subtract, exp, normalize, store once. Tutorial-02-guided, but written by you with the tutorial tab **closed**.
- `src/rmsnorm_triton.py`: same one-row-per-program structure; you wrote the math in week 5 — now fuse it into a single read of `x` and single write of `y`.
- **Definition of done:** `pytest tests/test_softmax.py tests/test_rmsnorm.py` green (≤ 1e-2 abs vs torch in fp16); `python bench/bench_kernels.py --op softmax rmsnorm` beats or matches eager torch and roughly matches compiled torch.
- Hint: the classic first bug is masking — rows shorter than `BLOCK_SIZE` need `tl.load(..., mask=offs < n_cols, other=-float('inf'))` for softmax (and `other=0.0` for RMSNorm), or the padding lanes poison your max/sum.

## Close the day (15 min)

- Anki: new cards for the quantize/dequantize equations, symmetric vs asymmetric, per-tensor/channel/group granularity, the format→hardware table, and E4M3 vs E5M2.
- One line in [notes.md](notes.md): the hardest thing today and why.
- Log blockers (Triton install issues, test failures) — flash attention starts tomorrow and needs a working toolchain.
