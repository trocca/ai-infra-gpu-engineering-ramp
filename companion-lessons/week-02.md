# Week 02 Companion - GPU Hardware Math, Memory Bandwidth, and CUDA/Rust Basics

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-1-nca-aiio/week-2/plan.md) · [build project](../gpu-engineering-lab/01-foundations/week-02-cuda-basics/README.md)

## Prerequisite Checklist

- You can explain why a GPU wins on data-parallel throughput but may lose on serial latency.
- You can compute bytes moved by a simple vector kernel.
- You know the difference between FP32, TF32, BF16, FP16, FP8, and INT8 at the "range vs precision" level.
- You can read a simple Rust function that returns `Result`.
- You know what a CUDA grid, block, thread, warp, global memory, and shared memory are.

## Mini Lesson

Week 2 is about building the hardware mental model. A GPU is not magic parallel Python; it is many simple threads moving data through a memory hierarchy. Most beginner kernels are limited by memory movement, not arithmetic.

For a vector add:

```text
c[i] = a[i] + b[i]
```

each element reads two FP32 values and writes one FP32 value:

```text
bytes per element = 3 * 4 = 12 bytes
total bytes = 12 * N
```

If the kernel processes `N=100,000,000` elements, it moves about 1.2 GB. Runtime is bounded by memory bandwidth.

## Math Insight

The simplest performance model is:

```text
time_lower_bound = bytes_moved / peak_bandwidth
```

If your measured time is 10x slower than that lower bound, the problem is not "the GPU is slow." Look for uncoalesced loads, small problem size, launch overhead, synchronization, or bad occupancy.

## Programming Primer

- CUDA: contiguous threads should read contiguous addresses; that is memory coalescing.
- Rust: RAII is your friend for GPU buffers. Own the device allocation in a struct and free it in `Drop`.
- Profiling: `torch.cuda.synchronize()` or CUDA events are required around timing; otherwise you time launch enqueue, not GPU work.

## 25-Minute Gate

1. Estimate bytes moved by SAXPY: `y[i] = a*x[i] + y[i]`.
2. Explain why a reduction is harder than vector add.
3. Sketch a grid/block mapping for one thread per array element.
4. Read the week 2 build README and identify the first command that proves the toolchain works.
