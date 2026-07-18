# Week 07 Companion - Quantization, Triton Tiling, and FlashAttention Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-2-ncp-genl/week-7/plan.md) · [build project](../gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md)

## Prerequisite Checklist

- You can explain scale and zero point for affine quantization.
- You can compute memory use for FP16, INT8, and INT4 weights.
- You know the softmax max trick from week 4.
- You understand why attention stores a `T x T` score matrix in the naive implementation.
- You can read a Triton kernel with program IDs, block pointers, and masks.

## Mini Lesson

Quantization is controlled loss. The goal is to reduce bytes moved and sometimes unlock lower-precision hardware paths while keeping model quality acceptable.

Simple asymmetric quantization:

```text
q = round(x / scale) + zero_point
x_hat = scale * (q - zero_point)
```

Weight-only quantization helps decode because decode is often memory-bandwidth-bound: the GPU spends much of its time reading weights and KV cache, not doing giant compute-bound matmuls.

## Math Insight

FlashAttention is an IO idea before it is an attention idea. Standard attention materializes the full `T x T` score matrix. FlashAttention tiles Q, K, and V, keeps running softmax statistics, and avoids writing the full attention matrix to HBM.

The core mental model:

```text
same math, fewer HBM reads/writes
```

That is why it can be faster without changing the model output.

## Playbook Bridge

Read the
[Week 7 Ultra-Scale Playbook bridge](../references/hf-ultrascale-playbook.md#week-7---kernels-flashattention-and-mixed-precision).
Treat FlashAttention as IO accounting, not a magic attention variant: the result is the
same, but the score matrix stops round-tripping through HBM. For precision, compare
range and mantissa bits before comparing memory size.

## Programming Primer

- Triton kernels operate on blocks. `program_id` chooses which block this instance owns.
- Masks prevent out-of-bounds loads/stores when dimensions are not exact multiples of block size.
- Always compare custom kernels against PyTorch references before benchmarking.
- Benchmark both speed and numerical error; low-bit work is not complete without error reporting.

## 25-Minute Gate

1. Estimate memory for an 8B model in FP16, INT8, and INT4.
2. Explain PTQ vs QAT in one sentence each.
3. Draw why storing `T x T` attention becomes impossible at long context.
4. Identify the week 7 tests that protect RMSNorm, softmax, quantization, and FlashAttention.
