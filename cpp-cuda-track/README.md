# C++ ↔ CUDA Dual Track

Two courses run in parallel and are graded against each other. Every module takes
**one operation or concept** and implements it twice — once on the CPU stack
(modern C++, threads, SIMD, OpenMP) and once on the GPU stack (CUDA) — then
*confronts* the two: same problem, different execution model, different memory
model, different failure modes. The goal is not "learn two languages" but to make
the transfer explicit: every CPU concept has a GPU counterpart, and the mapping
is the actual skill.

## The map (learn this table first)

| CPU stack (C++) | GPU stack (CUDA) | Confronted in |
|---|---|---|
| process / `std::thread` | grid / block / thread | 01 |
| core, SMT sibling | SM, resident warps | 01 |
| SIMD lane (AVX) | thread within a warp (SIMT) | 03 |
| L1/L2/L3 cache (implicit) | shared memory (explicit) + L2 | 02, 06 |
| cache line (64 B) | coalesced access segment (32/128 B) | 02 |
| DRAM (~100 GB/s/socket) | HBM (~3–8 TB/s) | 02 |
| `std::atomic`, C++ memory model | `atomicAdd`, scoped atomics `cuda::atomic` | 08 |
| mutex / condition variable | `__syncthreads()`, cooperative groups | 08 |
| thread pool / `std::async` | streams, events, CUDA Graphs | 07 |
| OpenMP `parallel for` | grid-stride kernel launch | 03 |
| cache blocking / tiling | shared-memory tiling | 06 |
| AVX-512 / AMX units | tensor cores (WMMA / WGMMA) | 10 |
| NUMA nodes, MPI ranks | multi-GPU, NVLink, NCCL | 11 |
| `perf`, VTune | Nsight Systems, Nsight Compute | 09 |

## Curriculum tree

```
cpp-cuda-track/
├── common/                       shared timing + validation helpers
├── 01-execution-model/           hello, 10 000 threads — who runs where
├── 02-memory-hierarchy/          caches vs shared memory; coalescing vs cache lines
├── 03-data-parallel-saxpy/       SIMD/OpenMP vs SIMT; the grid-stride idiom
├── 04-reduction/                 the canonical "parallelism is hard" op
├── 05-scan-histogram/            prefix sum & contention; atomics under load
├── 06-matmul-tiling/             cache blocking vs shared-mem tiling → BLAS
├── 07-async-overlap/             thread pools/futures vs streams/graphs
├── 08-sync-atomics-memory-model/ the two memory models, side by side
├── 09-profiling-roofline/        perf/VTune vs Nsight; roofline for both chips
├── 10-advanced-gpu/              warp intrinsics, tensor cores, Hopper/Blackwell
├── 11-multi-device/              NUMA & MPI vs P2P, NVLink, NCCL
└── 12-capstone-pytorch-extension/ one custom PyTorch op, both backends
```

Modules 01–06 are the foundation (do them in order). 07–09 can interleave.
10–12 are the payoff and connect back to the PyTorch/parallelism ramp.

## How to work a module

1. Read the module `README.md` — concept, the confrontation table, and the exercises.
2. Build and run the `cpp/` version. Measure it.
3. Build and run the `cuda/` version. Measure it.
4. Answer the confrontation questions in the README *in writing* (add a `NOTES.md`).
   The questions are chosen so the answer differs between the stacks — that delta
   is the lesson.

## Environment

- **CPU side:** any Linux box; `g++ >= 13` (C++20) or clang; OpenMP (`-fopenmp`).
- **GPU side:** CUDA Toolkit **12.4+** (13.x fine), driver to match, any GPU of
  compute capability 7.0+ (Volta) — modules 10–11 flag the parts that need
  Hopper (9.0) / Blackwell (10.x). No GPU locally? Colab, Lightning Studio, or a
  cheap spot instance all work; every CUDA example is a single `.cu` file.
- Build commands are in each README — plain `g++`/`nvcc` invocations, no build
  system, so nothing gets between you and the compiler flags.

## Rules of engagement

- Always compile with `-O3`. Benchmarking `-O0` code teaches lies.
- Never time a first CUDA kernel launch (JIT + context init); warm up, then time.
- `nvidia-smi` is not a profiler; module 09 installs the real tools early — use
  them from module 03 onward.
- Every claimed speedup gets a denominator: speedup over the *best* CPU version,
  not the naive one.
