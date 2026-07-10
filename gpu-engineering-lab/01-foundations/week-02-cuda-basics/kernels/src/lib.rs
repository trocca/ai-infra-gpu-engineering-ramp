//! Week 02 GPU kernels, written in Rust (Rust-CUDA / rustc_codegen_nvvm).
//!
//! Each `#[kernel]` fn below compiles to one PTX entry point with the same
//! (unmangled) name; the host bins in ../exercises load them by that name.
//! Signatures are fixed (the provided hosts pass args in this exact order) —
//! your work is the bodies.
//!
//! Useful cuda_std items (see the Rust-CUDA guide):
//!   thread::index_1d()            global linear thread index
//!   thread::thread_idx_x(), thread::block_idx_x(), thread::block_dim_x(),
//!   thread::grid_dim_x()          the raw coordinates
//!   thread::sync_threads()        __syncthreads()
//!   shared_array![f32; N]         static shared memory
//!
//! Safety note (repo policy): these fns are `unsafe` because they trust the
//! host's grid/block config and pointer sizes. Every raw-pointer write must
//! be guarded by a bounds check against `n`.

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes_definitions, clippy::missing_safety_doc)]

use cuda_std::prelude::*;

pub const BLOCK: usize = 256;

/// c[i] = a[i] + b[i].
///
/// TODO(Day 1): one thread per element.
///   * global index from thread::index_1d()
///   * n is not necessarily a multiple of the block size — bounds check
///   * read via a.add(i).read(), write via c.add(i).write(v) (raw pointers)
#[kernel]
pub unsafe fn vector_add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    // TODO(Day 1)
}

/// y[i] = alpha * x[i] + y[i], as a GRID-STRIDE loop.
///
/// TODO(Day 1): each thread handles i, i + total_threads, i + 2*total, ...
/// so ANY grid size is correct and the host can launch "enough blocks to
/// fill the GPU" instead of one thread per element. See:
/// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#[kernel]
pub unsafe fn saxpy(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    // TODO(Day 1)
}

/// Rung 1: block-level sum with INTERLEAVED addressing (the deliberately
/// slow textbook version). out[blockIdx.x] = sum of this block's elements.
///
/// TODO(Day 2):
///   * shared tile: let s = shared_array![f32; BLOCK];
///   * each thread loads one element (0.0 if out of range), sync
///   * tree step s = 1, 2, 4, ...: threads with tid % (2*s) == 0 add
///     tile[tid + s]; sync between steps — be able to say why exactly there
///   * thread 0 writes the block sum to out[block_idx]
/// The host driver re-launches this kernel on the partial sums until one
/// value remains.
#[kernel]
pub unsafe fn reduce_naive(input: *const f32, out: *mut f32, n: usize) {
    // TODO(Day 2)
}

/// Rung 2: SEQUENTIAL addressing + first-add-during-load + grid-stride.
///
/// TODO(Day 2):
///   * register accumulator over a grid-stride loop across the input
///   * one shared store per thread, then the tree with s = BLOCK/2, BLOCK/4,
///     ...: "if tid < s { tile[tid] += tile[tid + s] }" — active threads
///     stay contiguous (no divergence in active warps), conflict-free banks
///   * thread 0 writes out[block_idx]
/// The host launches a FIXED grid (blocks ~ 4x SM count) plus one final
/// single-block pass.
#[kernel]
pub unsafe fn reduce_shared(input: *const f32, out: *mut f32, n: usize) {
    // TODO(Day 2)
}

// NOTE: rung 3 (warp-shuffle reduction) intentionally does NOT live here.
// Warp intrinsics are the classic Rust-CUDA pain point, so that kernel takes
// the documented escape hatch: CUDA-C source compiled at runtime with NVRTC.
// See exercises/src/bin/reduce_warp.rs. (Stretch: port it back here and
// diff the PTX.)

/// Strided-read probe: thread i reads input[i * stride], writes ONE float.
///
/// TODO(Day 3):
///   * the compiler dead-code-eliminates unused loads — defeat it by writing
///     the loaded value to out[i] (one write per thread keeps write traffic
///     constant across strides, isolating READ behavior)
///   * bounds check against n_threads
#[kernel]
pub unsafe fn strided_read(input: *const f32, out: *mut f32, n_threads: usize, stride: usize) {
    // TODO(Day 3)
}

/// Offset-read probe: thread i reads input[i + offset].
///
/// TODO(Day 3): same dead-code-elimination defense as strided_read.
#[kernel]
pub unsafe fn offset_read(input: *const f32, out: *mut f32, n_threads: usize, offset: usize) {
    // TODO(Day 3)
}
