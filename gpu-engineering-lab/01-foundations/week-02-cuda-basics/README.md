# Week 02 — CUDA Basics, in Rust

> **Phase 1, Week 2** · First GPU kernels with a Rust host (cudarc) and Rust kernels
> (Rust-CUDA → PTX), honest event-based timing, memory-coalescing experiments, first contact
> with Nsight — plus `gpu-bench`, the reusable benchmark crate the rest of the repo uses.

Prerequisite support: [Week 02 companion lesson](../../../companion-lessons/week-02.md).

## Goal

Write your first kernels (vector add, SAXPY, three generations of reduction), measure them
properly with CUDA events, empirically map the memory system of the RTX 5090 Laptop GPU
(Blackwell, `sm_120`) with a bandwidth sweep, and learn to read Nsight Systems / Nsight
Compute output well enough to *name the bottleneck with evidence*.

**The two hierarchies every kernel this week exercises — where threads live, and where bytes live:**

```
Execution hierarchy                        Memory hierarchy
===================                        ================
Grid — one kernel launch                    fastest / smallest
 ├─ Block 0  ─┐                             ┌──────────────┐
 ├─ Block 1   ├─ each block runs            │  Registers   │  per thread
 ├─ Block 2   │  entirely on one SM         ├──────────────┤
 └─ ...      ─┘                             │  Shared mem  │  per block (on-SM)
      └─ Warp 0 = threads  0..31            ├──────────────┤
         Warp 1 = threads 32..63            │   L2 cache   │  all SMs
         ...    (SIMT lockstep — a          ├──────────────┤
         divergent branch serializes        │  HBM / DRAM  │  device-wide
         the warp)                          └──────────────┘
                                            slowest / largest
```

Systems language is **Rust** (repo thesis: Python where the ecosystem is, Rust where
performance matters — see `../../setup/rust-cuda-toolchain.md`):

- **Host**: [cudarc](https://github.com/coreylowman/cudarc) — driver API, streams, memcpy,
  events, NVRTC, cuBLAS. All host plumbing in this week is PROVIDED COMPLETE.
- **Kernels**: written in Rust in the `kernels/` crate, compiled to PTX by
  **Rust-CUDA** (`rustc_codegen_nvvm` via `cuda_builder` in `exercises/build.rs`).
- **Escape hatch** (policy from the setup doc): if Rust-CUDA fights you >30 min on a kernel,
  write that kernel as CUDA-C compiled at runtime via NVRTC through cudarc, keep the host in
  Rust, and log the friction in this README's "What didn't work" section. The warp-shuffle
  reduction ships *pre-wired* for the escape hatch — warp intrinsics are the classic
  Rust-CUDA pain point, so its kernel is a CUDA-C string in `reduce_warp.rs` (body still
  TODO; only the plumbing is chosen for you).

The deliverable that outlives this week: `common/` = the **`gpu-bench` crate** — CUDA-event
timing (warmup + median) and the repo's JSON result schema. Weeks 3 and 4 depend on it by
path. Read it line by line; you must be able to defend the methodology in an interview.

## Environment: WSL2 + Rust caveats (read before Day 1)

- **CUDA**: compute capability 12.0 needs **CUDA ≥ 12.8** in WSL2 Ubuntu; never install a
  Linux display driver in WSL (the Windows driver is used via `/usr/lib/wsl/lib/libcuda.so`).
  Guide: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- **Pinned nightly**: Rust-CUDA requires a specific nightly toolchain. `rust-toolchain.toml`
  in this folder pins it — **Day 1 task**: check
  <https://github.com/Rust-GPU/Rust-CUDA> for the currently required pin and update the file
  if it moved. `rustup` auto-installs the pin on first `cargo build` in this directory.
- **cudarc needs no CUDA at build time** (it dlopens libcuda at runtime); `cuda_builder`
  DOES need the Rust-CUDA toolchain. CI and GPU-less machines set `SKIP_PTX=1`, which makes
  `build.rs` emit a stub PTX so `cargo clippy`/`cargo check` still work.
- **RAII device buffers**: `CudaSlice<f32>` owns device memory and frees it on `Drop` —
  there is no `cudaFree` to forget. Corollary: keep buffers alive across the benchmark
  closure (bind them to variables; don't let temporaries drop mid-benchmark).
- **`unsafe` policy (repo-wide)**: kernel launches are `unsafe` because the compiler cannot
  check your arg list against the PTX signature. Keep every `unsafe` block minimal, and
  state the invariant you're vouching for in a comment directly above it.
- **Profiling on WSL2**: Nsight Systems works; Nsight Compute needs GPU performance counters
  enabled for non-admin users on the *Windows* side (NVIDIA control panel → Developer →
  Manage GPU Performance Counters). `ERR_NVGPUCTRPERM` means this:
  <https://developer.nvidia.com/ERR_NVGPUCTRPERM>. Profilers see PTX/SASS — they don't care
  that rustc emitted it. Build `--release` (this workspace keeps `debug = true` in release
  for readable symbol names).
- **Laptop power**: the 5090 Laptop GPU is power-limited; clocks move constantly. Plug in,
  fix the Windows power profile, log `nvidia-smi --query-gpu=clocks.sm,power.draw
  --format=csv -l 1` during benchmarks, report the median of ≥50 runs — never the best run.

## Background reading

| Resource | Why |
|---|---|
| *An Even Easier Introduction to CUDA* — <https://developer.nvidia.com/blog/even-easier-introduction-cuda/> | Grid/block mental model. C++-flavored, transfers 1:1 — the GPU only ever sees PTX. |
| CUDA C++ Programming Guide, ch. 1–5 — <https://docs.nvidia.com/cuda/cuda-c-programming-guide/> | THE reference for the machine model, whatever the source language. |
| Rust-CUDA guide — <https://github.com/Rust-GPU/Rust-CUDA> | `#[kernel]`, `cuda_std` thread indexing, shared memory, `cuda_builder`, the nightly pin. |
| cudarc docs — <https://docs.rs/cudarc> | Streams, `memcpy_stod/dtov`, `launch_builder`, events, NVRTC, cuBLAS. |
| Harris, *Optimizing Parallel Reduction in CUDA* — <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf> | The reduction ladder you re-derive on Day 2. |
| *Faster Parallel Reductions on Kepler* — <https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/> | The `__shfl_down_sync` pattern (used in the Day-2 escape-hatch kernel). |
| *How to Access Global Memory Efficiently* — <https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/> | Coalescing — the exact experiment you run Day 3. |
| Nsight Compute docs — <https://docs.nvidia.com/nsight-compute/> | Speed-of-light section, memory workload analysis. |

## Layout

```
Cargo.toml            workspace (common, kernels, exercises)
rust-toolchain.toml   pinned nightly for rustc_codegen_nvvm
common/               gpu-bench crate — COMPLETE (events, warmup+median, JSON)
kernels/              Rust GPU kernels — signatures given, bodies TODO
exercises/            one bin per experiment — host plumbing COMPLETE
  src/bin/            vector_add.rs  saxpy.rs  reduce_naive.rs
                      reduce_shared.rs  reduce_warp.rs (NVRTC escape hatch)
                      bandwidth_sweep.rs
bench/plot_results.py COMPLETE matplotlib (reads the JSON)
```

## Day-by-day plan (4 h/day)

### Day 1 (Mon) — Toolchain + vector add + SAXPY
- Morning: pin the nightly (see caveats), `cargo build --release` until the PTX builds.
  Budget real time for this; log every snag in "What didn't work".
- Read `common/src/lib.rs` top to bottom — it times everything you build this month.
- Implement `kernels/src/lib.rs::vector_add` (one thread per element, bounds check) and
  `saxpy` (as a **grid-stride loop** — see
  <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>).
- Fill the `TODO(Day 1)` grid-size lines in the two bins.
- `cargo run --release --bin vector_add` — the harness checks correctness against the CPU
  and prints achieved bandwidth. Vector add moves 3 floats/element; you should land within
  ~80–90% of Day 3's measured peak.

### Day 2 (Tue) — The reduction ladder
- `reduce_naive`: interleaved-addressing shared-memory tree (deliberately the slow textbook
  version — divergence + bank conflicts are the point; Day 4 profiles them).
- `reduce_shared`: sequential addressing + first-add-during-load + grid-stride input loop.
- `reduce_warp`: the escape-hatch kernel — a CUDA-C string in `reduce_warp.rs`, body TODO:
  `__shfl_down_sync` for the final 32 lanes, one shared slot per warp for the block combine.
- Every variant is validated against a CPU float64 sum by the provided harness.
- **Baseline** (acceptance criterion): the harness times `cublasSasum` on the same buffer
  via cudarc's cuBLAS bindings. Inputs are generated non-negative, so sum == asum and the
  memory traffic is identical to a sum-reduction. CUB isn't reachable from Rust; cuBLAS is
  the honest library baseline, and the results writeup must state that substitution.

### Day 3 (Wed) — Memory coalescing experiments

**What the sweep measures — one 128-byte transaction serving a whole warp vs one per lane:**

```
Coalesced (stride 1): a warp's 32 x 4-byte loads fill one 128-B transaction

 lane:    0    1    2    3   ...   31
 addr:  [  0][  4][  8][ 12] ...  [124]   ->  1 transaction / warp request
         └──────── one 128-B line ────┘       bandwidth ~ peak

Strided (stride 32 floats): every lane touches a DIFFERENT 128-B line

 lane:    0        1        2      ...    31
 addr:  [   0]  [ 128]  [ 256]     ...  [3968]  ->  32 transactions / request,
         line0    line1    line2         line31     31/32 of each line wasted
```

- Implement `strided_read` and `offset_read` in the kernels crate (defeat dead-code
  elimination — the bin comments explain the trick).
- `cargo run --release --bin bandwidth_sweep`, then `python bench/plot_results.py results`.
- Stare at the chart: the cliff as stride crosses the 128-byte transaction boundary.
  Annotate it with the cache-line arithmetic that explains it.

### Day 4 (Thu) — Profilers
- `nsys profile target/release/reduce_warp` — timeline, launch overhead vs kernel time,
  memcpys.
- `ncu --set full` on all three reductions: compare Speed-of-Light %, memory chart, warp
  stall reasons. Per variant, write down: bound by what? evidence?
- This day turns "it's faster" into "DRAM throughput went from 31% to 87% of speed-of-light."

### Day 5 (Fri) — Benchmark, plot, publish
- `make bench` — runs every bin, aggregates `results/*.json`, regenerates both charts.
- Write `RESULTS.md`: table (median over ≥50 runs), the two charts, one Nsight-evidence
  paragraph per reduction rung, clock/power log summary, and the **"What didn't work"**
  section — every Rust-CUDA friction point and every escape hatch taken, honestly. Push.

## Deliverables

- [ ] Working `vector_add`, `saxpy`, three reductions, `bandwidth_sweep` (all via `make bench`)
- [ ] `common/` (gpu-bench) understood line-by-line — it's the repo's shared harness
- [ ] `results/*.json` + two plots
- [ ] `RESULTS.md` incl. Nsight evidence + "What didn't work" toolchain log

## Acceptance criteria

1. All kernels produce correct results vs the CPU reference (harnesses check automatically;
   bins exit non-zero on mismatch).
2. `reduce_warp` achieves **≥80% of `cublasSasum` throughput** on 2²⁶ floats (the harness
   runs the baseline and prints the percentage; exit code 2 = correct but too slow).
   Stated substitution: cuBLAS Sasum stands in for CUB, which has no Rust bindings.
3. The strided-access chart clearly shows the coalescing cliff, and `RESULTS.md` explains it
   in terms of 32-byte/128-byte memory transactions.
4. `make bench` is the one command that reproduces every number you publish.

## Benchmark methodology (repo contract, laptop edition)

- CUDA-event timing around kernel work only (no memcpys in the timed region unless the
  experiment is *about* transfer). Implemented once in `gpu-bench`, used everywhere.
- ≥5 warmup launches, then ≥50 timed launches, report the **median**.
- Log clocks and power during runs; discard runs where clocks sagged >10% from the median.
- JSON schema (enforced by `gpu_bench::write_json`):
  `{"name", "n", "median_ms", "runs", "bytes_moved", "gbps", "gflops", ...meta}`.

## Stretch goals

- A `float4`-style vectorized copy kernel in the sweep (`vek` types in cuda_std, or via the
  escape hatch) — how much of peak do you gain at stride 1?
- Port `reduce_warp` from the CUDA-C escape hatch to pure Rust-CUDA and diff the PTX.
- Grid-size autotune loop for the reduction (occupancy vs oversubscription).

## What didn't work (fill in as you go — it's a feature, not a confession)

> _Every Rust-CUDA friction point, nightly breakage, or escape hatch taken goes here with a
> date and a one-line diagnosis. Reviewers read this section first: it shows you can
> navigate an immature toolchain without stalling._

## Interview talking points this week earns

1. "I can explain memory coalescing with a chart I measured myself — including exactly where
   the cliff is on Blackwell and why (transaction granularity vs request stride)."
2. "I've walked the classic reduction ladder — divergence, bank conflicts, sequential
   addressing, warp shuffle — and profiled each rung in Nsight Compute rather than trusting
   folklore. The profilers don't care that my host and most kernels are Rust: it's all PTX."
3. "I know what CUDA-event timing does and doesn't measure, why warmup matters, and why I
   report medians with clock logs on a power-limited laptop."
4. "I run a hybrid Rust/CUDA toolchain in anger: Rust kernels via rustc_codegen_nvvm where
   it works, NVRTC-compiled CUDA-C where it doesn't, one Rust host either way — and I can
   tell you exactly where the ecosystem's rough edges are."

## Definition of done

- [ ] `cargo build --release` clean on the pinned nightly; `cargo clippy` clean
- [ ] `make bench` clean from fresh clone (given CUDA ≥12.8 + pinned toolchain)
- [ ] cuBLAS-Sasum comparison ≥80% recorded in RESULTS.md
- [ ] Both charts committed, axes labeled, machine + clock info in the caption
- [ ] Nsight screenshots or metric tables for all three reductions
- [ ] "What didn't work" has at least the Day-1 toolchain notes
- [ ] Pushed
