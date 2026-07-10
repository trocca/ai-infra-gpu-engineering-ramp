//! The Week 03 SGEMM ladder — kernels in Rust (Rust-CUDA / rustc_codegen_nvvm).
//!
//! C = alpha * A @ B + beta * C, fp32, ROW-MAJOR, square sizes.
//! Each rung's comment states the ONE concept it adds — the concept, not the
//! code. Launch configs are already wired in runner/src/main.rs from the
//! tile constants below.
//!
//! ESCAPE HATCH, per rung (../../setup/rust-cuda-toolchain.md): if the
//! toolchain fights a rung >30 min, write it as CUDA-C in the
//! NVRTC_OVERRIDES table in runner/src/main.rs — the runner will use it
//! instead of the same-named kernel here. Log every hatch in the README.
//!
//! cuda_std toolbox: thread::thread_idx_x/block_idx_x/block_dim_x,
//! thread::sync_threads(), shared_array![f32; LEN], `vek` vector types for
//! rung 6. Watch for register spills on rungs 4-6 (ncu will tell you).

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes_definitions, clippy::missing_safety_doc, clippy::too_many_arguments)]

use cuda_std::prelude::*;

// ---------------------------------------------------------------------------
// Tiling parameters — shared with the runner's launch configs (which are
// complete). Tune AFTER everything works. Sweep sizes are multiples of
// BM/BN/BK so you may assume divisibility (predication is a stretch goal).
// ---------------------------------------------------------------------------
pub const BM: usize = 128; // threadblock tile rows of C
pub const BN: usize = 128; // threadblock tile cols of C
pub const BK: usize = 8;   // K-slice staged in shared memory per iteration
pub const TM: usize = 8;   // per-thread tile rows   (rungs 4-6)
pub const TN: usize = 8;   // per-thread tile cols   (rungs 5-6)
pub const TILE: usize = 32; // square smem tile for rung 3

// ===========================================================================
// RUNG 1 — naive. One thread per C element; the inner loop walks K reading
// A[row, k] and B[k, col] straight from global memory.
// CONCEPT: establish correctness + a baseline. Nothing clever. Note which
// index you map to thread_idx_x and what that does to B's access pattern —
// rung 2 exists because of this choice.
// Launch (wired): grid ((N+31)/32, (M+31)/32), block (32, 32).
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_naive(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 1)
}

// ===========================================================================
// RUNG 2 — global-memory coalescing.
// CONCEPT: consecutive thread_idx_x must read/write consecutive addresses.
// Same one-thread-per-element algorithm, different thread->element mapping
// so a warp walks along a ROW of C (and thus a row of B). Expect a large
// jump for a tiny diff — measure it.
// Launch (wired): 1-D block of 32*32 threads; derive row/col from the flat
// index inside the kernel.
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_coalesced(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 1)
}

// ===========================================================================
// RUNG 3 — shared-memory tiling.
// CONCEPT: every element of A and B is reused by a whole tile of threads;
// stage TILE x TILE blocks in shared memory once, let all threads read them
// ~TILE times each. Global traffic drops ~TILE x.
// K-loop skeleton: load tile -> sync_threads() -> FMA over the tile ->
// sync_threads() -> next tile. Be able to justify BOTH syncs.
// ACCEPTANCE: >= 10x rung 1 at 4096^2.
// Shared memory: let a_tile = shared_array![f32; TILE * TILE]; etc.
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_smem_tiled(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 2)
}

// ===========================================================================
// RUNG 4 — 1-D register blocking (blocktiling).
// CONCEPT: arithmetic intensity per thread. Each thread OWNS TM vertical
// results of C, accumulated in a register array [f32; TM]. Each value read
// from shared memory is reused TM times => smem bandwidth per FMA drops by
// TM. Threadblock tile becomes BM x BN with a BK-deep K slice.
// Launch (wired): (BM*BN)/TM threads per block.
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_1d_blocktile(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 3)
}

// ===========================================================================
// RUNG 5 — 2-D register tiling.
// CONCEPT: outer products. Each thread owns a TM x TN patch of C
// ([[f32; TN]; TM]). Inner loop per k: load TM values of the A-column and
// TN values of the B-row into registers, then do the TM*TN FMAs — an outer
// product. smem reads per FMA fall to (TM+TN)/(TM*TN). This is the rung
// that should move you from memory-bound toward compute-bound in ncu.
// Launch (wired): (BM*BN)/(TM*TN) threads per block.
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_2d_blocktile(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 3)
}

// ===========================================================================
// RUNG 6 — vectorized loads.
// CONCEPT: issue 128-bit memory instructions. Global->shared loads move 4
// floats at a time (16-byte-aligned — the sweep sizes guarantee it), and A
// is stored TRANSPOSED into shared memory during the load so the compute
// loop reads both tiles with stride-1/broadcast patterns.
// In Rust: cuda_std's `vek` Vec4<f32>, or raw 16-byte reads via pointer
// casts — or take the ESCAPE HATCH; this rung is its most likely customer.
// ===========================================================================
#[kernel]
pub unsafe fn sgemm_vectorized(
    m: usize, n: usize, k: usize, alpha: f32,
    a: *const f32, b: *const f32, beta: f32, c: *mut f32,
) {
    // TODO(Day 4)
}

// RUNG 7 (stretch) — Tensor Cores. No Rust-CUDA surface for wmma: this one
// is escape-hatch-only (nvcuda::wmma in an NVRTC override) or CubeCL.
// Register it in the runner's RUNGS table when you attempt it.
