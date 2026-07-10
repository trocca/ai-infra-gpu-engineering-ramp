//! rusty-kernels GPU crate — fused softmax + LayerNorm, in Rust.
//!
//! Layout contract (enforced Python-side in ops.py and Rust-side in ext/):
//! inputs are contiguous 2-D (rows, cols); the op works over cols (dim=-1).
//! One block per row is the intended starting design; BLOCK = 256.
//!
//! Dtype story: no generics across PTX entry points — each op has explicit
//! _f32 / _f64 (and _f16 where required) variants. f64 exists ONLY so
//! torch.autograd.gradcheck can run; it will be slow and that's fine.
//! For f16, accumulate in f32 (`half::f16` on the wire, f32 math).
//!
//! ESCAPE HATCH (../../setup/rust-cuda-toolchain.md): any kernel here may
//! drop to CUDA-C compiled via NVRTC — ext/src/lib.rs has an override table
//! like week-03's runner. Log every hatch in the README.
//!
//! Your Week-2 warp/block reduction choreography is directly reusable in
//! every kernel below.

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes_definitions, clippy::missing_safety_doc, clippy::too_many_arguments)]

use cuda_std::prelude::*;

pub const BLOCK: usize = 256;

// ===========================================================================
// Day 1 — passthrough (build smoke test): out[i] = input[i].
// TODO(Day 1): grid-stride copy of n elements. The point is proving the
// maturin -> PyO3 -> data_ptr -> PTX -> launch chain, not the kernel.
// ===========================================================================
#[kernel]
pub unsafe fn passthrough_f32(input: *const f32, out: *mut f32, n: usize) {
    // TODO(Day 1)
}

// ===========================================================================
// Day 2 — ONLINE softmax forward, one block per row.
//
// The idea (Milakov & Gimelshein 2018): keep a running maximum m and a
// running sum d of exp(x - m). When a new element raises the max from m to
// m', rescale the old sum by exp(m - m'). ONE read of the row yields BOTH
// the true max and the correctly-scaled denominator; a second sweep writes
// exp(x - m) / d. (Naive stable softmax needs 3 sweeps: max, sum, write.)
//
// TODO(Day 2):
//   * each thread strides over the row accumulating its own (m, d) pair
//   * merge (m, d) pairs across the warp with shuffles — the merge rule for
//     two pairs is the same rescaling trick
//   * merge warps via shared memory (Week-2 block-reduction shape)
//   * broadcast the final (m, d); each thread writes its strided elements
// ===========================================================================
#[kernel]
pub unsafe fn softmax_fwd_f32(x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    // TODO(Day 2)
}

/// f64 twin — same algorithm, for gradcheck paths that route through
/// softmax. TODO(Day 2): implement after _f32 works (copy, retype).
#[kernel]
pub unsafe fn softmax_fwd_f64(x: *const f64, y: *mut f64, rows: usize, cols: usize) {
    // TODO(Day 2)
}

/// f16 in/out, f32 accumulation. Required by the fp16 forward tests —
/// implement natively OR document an upcast fallback in ops.py (and its
/// cost). Wire format is raw u16 bits (IEEE binary16): convert with
/// cuda_std's half support if available, else bit-twiddle / take the
/// escape hatch (__half in CUDA-C). TODO(Day 2 / stretch).
#[kernel]
pub unsafe fn softmax_fwd_f16(x: *const u16, y: *mut u16, rows: usize, cols: usize) {
    // TODO(Day 2 / stretch)
}

// ===========================================================================
// Day 3 — LayerNorm forward, one block per row.
//
//   y[r, c] = (x[r, c] - mean_r) * rstd_r * weight[c] + bias[c]
//   rstd_r  = 1 / sqrt(var_r + eps)
//
// mean/rstd are WRITTEN OUT (saved for backward — cheaper than recomputing;
// a real memory-vs-math design decision, note it in RESULTS.md).
//
// Variance: PICK ONE and defend it in RESULTS.md:
//   (a) Welford / Chan parallel merge: one read of the row; per-thread
//       (count, mean, M2) triples merged across the block with the parallel
//       combination formula. Numerically robust for huge rows.
//   (b) Two-pass: pass 1 mean, pass 2 E[(x-mean)^2]. One extra (L2-warm)
//       read, dead simple, very stable.
// ===========================================================================
#[kernel]
pub unsafe fn layernorm_fwd_f32(
    x: *const f32, weight: *const f32, bias: *const f32,
    y: *mut f32, mean: *mut f32, rstd: *mut f32,
    rows: usize, cols: usize, eps: f32,
) {
    // TODO(Day 3)
}

/// f64 twin — THIS is the one gradcheck exercises. TODO(Day 3).
#[kernel]
pub unsafe fn layernorm_fwd_f64(
    x: *const f64, weight: *const f64, bias: *const f64,
    y: *mut f64, mean: *mut f64, rstd: *mut f64,
    rows: usize, cols: usize, eps: f64,
) {
    // TODO(Day 3)
}

// ===========================================================================
// Day 3 — LayerNorm backward, two kernels.
//
// dx (one block per row). Derive on paper first. With
// xhat = (x - mean) * rstd and g = dy * weight:
//   dx = rstd * ( g - mean_c(g) - xhat * mean_c(g * xhat) )
// where mean_c is the mean over the row. Two row reductions (sum g, sum
// g*xhat), then one elementwise pass — all in one kernel.
// ===========================================================================
#[kernel]
pub unsafe fn layernorm_bwd_dx_f32(
    dy: *const f32, x: *const f32, weight: *const f32,
    mean: *const f32, rstd: *const f32, dx: *mut f32,
    rows: usize, cols: usize,
) {
    // TODO(Day 3)
}

#[kernel]
pub unsafe fn layernorm_bwd_dx_f64(
    dy: *const f64, x: *const f64, weight: *const f64,
    mean: *const f64, rstd: *const f64, dx: *mut f64,
    rows: usize, cols: usize,
) {
    // TODO(Day 3)
}

// ===========================================================================
// Day 3 — dweight/dbias: reductions over ROWS (the other axis!).
//   dweight[c] = sum_r dy[r,c] * (x[r,c] - mean_r) * rstd_r
//   dbias[c]   = sum_r dy[r,c]
// Launched with one thread per column stripe (grid = ceil(cols / BLOCK)):
// each thread walks down its column accumulating both sums — coalesced,
// atomic-free. (Alternative: atomicAdd from the row kernel — measure both
// if time allows.)
// ===========================================================================
#[kernel]
pub unsafe fn layernorm_bwd_dwdb_f32(
    dy: *const f32, x: *const f32,
    mean: *const f32, rstd: *const f32,
    dweight: *mut f32, dbias: *mut f32,
    rows: usize, cols: usize,
) {
    // TODO(Day 3)
}

#[kernel]
pub unsafe fn layernorm_bwd_dwdb_f64(
    dy: *const f64, x: *const f64,
    mean: *const f64, rstd: *const f64,
    dweight: *mut f64, dbias: *mut f64,
    rows: usize, cols: usize,
) {
    // TODO(Day 3)
}
