# Week 03 Companion - Network Fabrics, Roofline Thinking, and SGEMM Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-1-nca-aiio/week-3/plan.md) · [build project](../gpu-engineering-lab/01-foundations/week-03-matmul-optimization/README.md)

## Prerequisite Checklist

- You can explain latency vs bandwidth without mixing them up.
- You know why synchronized training traffic is sensitive to tail latency.
- You can compute matrix multiplication FLOPs for `C = A @ B`.
- You remember the week 2 memory-bandwidth model.
- You can explain shared memory as programmer-managed scratchpad memory.

## Mini Lesson

This week has two halves that look different but share the same question: "what bottleneck sets the speed limit?"

For networking, the bottleneck may be latency, bandwidth, congestion, or one slow participant in a collective. For SGEMM, the bottleneck is either memory bandwidth or compute throughput.

Matmul shape:

```text
A: M x K
B: K x N
C: M x N
FLOPs ~= 2 * M * N * K
```

The factor 2 is one multiply and one add.

## Math Insight

Roofline thinking connects arithmetic to memory:

```text
arithmetic_intensity = FLOPs / bytes_moved
ridge_point = peak_FLOPs / peak_bandwidth
```

If arithmetic intensity is below the ridge point, the kernel is bandwidth-bound. If it is above, the kernel can become compute-bound. SGEMM optimization is the art of increasing reuse so each byte loaded from memory supports more FLOPs.

## Programming Primer

- Tiling: load blocks of A and B into shared memory, reuse them for many multiply-adds, then move to the next tile.
- Rust unsafe: use it as a small boundary around operations the compiler cannot prove, not as a style.
- Benchmarking: compare against cuBLAS because the reference is what "good" looks like.

## 25-Minute Gate

1. Compute FLOPs for multiplying two `1024 x 1024` matrices.
2. In one paragraph, explain why GPUDirect RDMA matters for multi-node training.
3. Draw a tiled matmul with a `TILE x TILE` shared-memory block.
4. Read the week 3 acceptance criteria and identify which benchmark proves progress.
