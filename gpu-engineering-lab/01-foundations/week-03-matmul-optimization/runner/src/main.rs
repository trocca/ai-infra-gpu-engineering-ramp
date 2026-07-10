//! sgemm_ladder runner — COMPLETE. Do not edit except to (a) paste CUDA-C
//! escape-hatch kernels into NVRTC_OVERRIDES, (b) register a stretch rung 7.
//!
//! For every size in the sweep and every rung:
//!   1. computes the cuBLAS reference (also the 100% line in the chart)
//!   2. runs your kernel once and gates on max-abs error <= 2e-2
//!   3. if correct, benchmarks it (gpu-bench: 5 warmup + 50 timed, median)
//!   4. writes results/sgemm_ladder.json for bench/plot_ladder.py
//!
//! Run: cargo run --release --bin sgemm_ladder [-- size ...]
//! Default sweep: 128 256 512 1024 2048 4096

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError,
                     LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, write_json, BenchResult};
use sgemm_kernels::{BK, BM, BN, TILE, TM, TN};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sgemm_kernels.ptx"));

// ---------------------------------------------------------------------------
// ESCAPE HATCH TABLE. To take the hatch for a rung, add an entry mapping the
// kernel name to CUDA-C source; the runner compiles it with NVRTC and uses
// it INSTEAD of the Rust kernel of the same name.
//
// The CUDA-C signature MUST match the Rust one bit-for-bit, i.e.:
//   extern "C" __global__ void sgemm_xxx(
//       size_t M, size_t N, size_t K, float alpha,
//       const float* A, const float* B, float beta, float* C)
// (size_t, not int — the Rust host passes usize.)
// ---------------------------------------------------------------------------
const NVRTC_OVERRIDES: &[(&str, &str)] = &[
    // ("sgemm_vectorized", include_str!("nvrtc/sgemm_vectorized.cu")),
];

struct Rung {
    name: &'static str,   // JSON/chart name
    kernel: &'static str, // PTX entry point
    id: u32,              // selects the launch config below
}

const RUNGS: &[Rung] = &[
    Rung { name: "1_naive",        kernel: "sgemm_naive",        id: 1 },
    Rung { name: "2_coalesced",    kernel: "sgemm_coalesced",    id: 2 },
    Rung { name: "3_smem_tiled",   kernel: "sgemm_smem_tiled",   id: 3 },
    Rung { name: "4_1d_blocktile", kernel: "sgemm_1d_blocktile", id: 4 },
    Rung { name: "5_2d_blocktile", kernel: "sgemm_2d_blocktile", id: 5 },
    Rung { name: "6_vectorized",   kernel: "sgemm_vectorized",   id: 6 },
];

/// Launch configuration per rung, derived from the tile constants.
fn launch_config(id: u32, m: usize, n: usize) -> LaunchConfig {
    let (grid, block): ((u32, u32, u32), (u32, u32, u32)) = match id {
        1 => (
            (n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1),
            (32, 32, 1),
        ),
        2 => (
            (n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1),
            (32 * 32, 1, 1),
        ),
        3 => (
            (n.div_ceil(TILE) as u32, m.div_ceil(TILE) as u32, 1),
            ((TILE * TILE) as u32, 1, 1),
        ),
        4 => (
            (n.div_ceil(BN) as u32, m.div_ceil(BM) as u32, 1),
            (((BM * BN) / TM) as u32, 1, 1),
        ),
        5 | 6 => (
            (n.div_ceil(BN) as u32, m.div_ceil(BM) as u32, 1),
            (((BM * BN) / (TM * TN)) as u32, 1, 1),
        ),
        other => panic!("unknown rung id {other}"),
    };
    LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 }
}

#[allow(clippy::too_many_arguments)]
fn launch_rung(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    id: u32,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &CudaSlice<f32>, b: &CudaSlice<f32>,
    beta: f32,
    c: &mut CudaSlice<f32>,
) -> Result<(), DriverError> {
    let cfg = launch_config(id, m, n);
    // SAFETY: arg order matches every rung's fixed signature
    // (m, n, k, alpha, a, b, beta, c); a/b/c are live and non-aliasing.
    unsafe {
        stream.launch_builder(f)
            .arg(&m).arg(&n).arg(&k).arg(&alpha)
            .arg(a).arg(b).arg(&beta).arg(&mut *c)
            .launch(cfg)?;
    }
    Ok(())
}

/// cuBLAS reference for ROW-major C = alpha*A@B + beta*C: compute the
/// column-major product B' * A' = C', which lands in memory exactly as
/// row-major C. (The classic transpose trick — be able to explain it.)
fn cublas_ref(
    blas: &CudaBlas,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &CudaSlice<f32>, b: &CudaSlice<f32>,
    beta: f32,
    c: &mut CudaSlice<f32>,
) -> Result<(), cudarc::cublas::result::CublasError> {
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha,
        lda: n as i32,
        ldb: k as i32,
        beta,
        ldc: n as i32,
    };
    // SAFETY: dimensions describe the buffers exactly (all n*n contiguous).
    unsafe { blas.gemm(cfg, b, a, c) }
}

/// Deterministic pseudo-random values in [-1, 1).
fn fill_random(len: usize, seed: u64) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let h = (i as u64).wrapping_add(seed).wrapping_mul(0x9E3779B97F4A7C15);
            ((h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes: Vec<usize> = {
        let args: Vec<usize> = std::env::args().skip(1)
            .filter_map(|s| s.parse().ok()).collect();
        if args.is_empty() { vec![128, 256, 512, 1024, 2048, 4096] } else { args }
    };

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    println!("device: {}", ctx.name()?);

    // Main module (Rust kernels) + any NVRTC escape-hatch overrides.
    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let mut overrides: HashMap<&str, CudaFunction> = HashMap::new();
    for (kernel, src) in NVRTC_OVERRIDES {
        let ptx = cudarc::nvrtc::compile_ptx(src)?;
        let m = ctx.load_module(ptx)?;
        overrides.insert(kernel, m.load_function(kernel)?);
        println!("escape hatch active for {kernel} (NVRTC)");
    }
    let resolve = |kernel: &str| -> Result<CudaFunction, DriverError> {
        if let Some(f) = overrides.get(kernel) {
            Ok(f.clone())
        } else {
            module.load_function(kernel)
        }
    };

    // fp32 tolerance: different accumulation orders diverge; with inputs in
    // [-1, 1] and K <= 4096 this bound is comfortable for a correct kernel
    // and unreachable for a wrong one.
    const TOL: f64 = 2e-2;
    const ALPHA: f32 = 1.0;
    const BETA: f32 = 0.0;

    let mut results: Vec<BenchResult> = Vec::new();
    let mut all_ok = true;

    for &sz in &sizes {
        let (m, n, k) = (sz, sz, sz);
        let elems = sz * sz;
        let flop = 2.0 * (m as f64) * (n as f64) * (k as f64);

        let ha = fill_random(elems, 1);
        let hb = fill_random(elems, 2);
        let da = stream.memcpy_stod(&ha)?;
        let db = stream.memcpy_stod(&hb)?;
        let mut dc = stream.alloc_zeros::<f32>(elems)?;
        let mut dref = stream.alloc_zeros::<f32>(elems)?;

        // ---- reference ------------------------------------------------------
        cublas_ref(&blas, m, n, k, ALPHA, &da, &db, BETA, &mut dref)?;
        let href = stream.memcpy_dtov(&dref)?;

        // ---- cuBLAS timing (the 100% line) ----------------------------------
        {
            let mut r = BenchResult {
                name: "cublas".into(),
                n: sz as i64,
                runs: 50,
                meta: format!("\"size\": {sz}"),
                ..Default::default()
            };
            r.median_ms = benchmark(&ctx, &stream, || {
                cublas_ref(&blas, m, n, k, ALPHA, &da, &db, BETA, &mut dref)
                    .expect("cublas gemm failed");
                Ok(())
            }, 5, r.runs as usize)?;
            r.gflops = flop / (r.median_ms * 1e-3) / 1e9;
            println!("[{sz:4}] {:<16} {:9.3} ms  {:8.0} GFLOPS", r.name, r.median_ms, r.gflops);
            results.push(r);
        }

        // ---- the ladder ------------------------------------------------------
        for rung in RUNGS {
            let f = resolve(rung.kernel)?;

            // Correctness gate.
            stream.memset_zeros(&mut dc)?;
            launch_rung(&stream, &f, rung.id, m, n, k, ALPHA, &da, &db, BETA, &mut dc)?;
            let hc = stream.memcpy_dtov(&dc)?;
            let err = max_abs_err(&hc, &href);
            if err > TOL {
                println!("[{sz:4}] {:<16} FAILED  max_abs_err {err:.3e} (tol {TOL:.0e}) — skipping bench",
                         rung.name);
                all_ok = false;
                continue;
            }

            // Benchmark.
            let mut r = BenchResult {
                name: rung.name.into(),
                n: sz as i64,
                runs: 50,
                meta: format!("\"size\": {sz}, \"max_abs_err\": {err:e}"),
                ..Default::default()
            };
            r.median_ms = benchmark(&ctx, &stream, || {
                launch_rung(&stream, &f, rung.id, m, n, k, ALPHA, &da, &db, BETA, &mut dc)
            }, 5, r.runs as usize)?;
            r.gflops = flop / (r.median_ms * 1e-3) / 1e9;
            println!("[{sz:4}] {:<16} {:9.3} ms  {:8.0} GFLOPS  (err {err:.1e})",
                     r.name, r.median_ms, r.gflops);
            results.push(r);
        }
    }

    write_json("results/sgemm_ladder.json", &results)?;
    std::process::exit(if all_ok { 0 } else { 1 });
}
