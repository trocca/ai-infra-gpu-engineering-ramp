//! Day 2, rung 3: warp-shuffle reduction + the cuBLAS Sasum baseline.
//!
//! THIS KERNEL TAKES THE DOCUMENTED ESCAPE HATCH (see
//! ../../setup/rust-cuda-toolchain.md): warp intrinsics are the classic
//! Rust-CUDA pain point, so the kernel below is CUDA-C source compiled at
//! runtime with NVRTC through cudarc. Host stays Rust. The kernel BODY is
//! still yours to write. (Stretch: port it to Rust-CUDA and diff the PTX.)
//!
//! ACCEPTANCE CRITERION lives in this bin: your kernel must reach >= 80% of
//! cuBLAS Sasum throughput on the same 2^26-float buffer. Inputs are
//! generated NON-NEGATIVE, so sum == asum and the memory traffic is
//! identical to a sum-reduction. Stated substitution: CUB has no Rust
//! bindings; cuBLAS is the honest library baseline reachable from Rust.
//!
//! Run: cargo run --release --bin reduce_warp
//! Exit codes: 0 = pass, 1 = incorrect, 2 = correct but < 80% of baseline.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr,
                     DriverError, LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const N: usize = 1 << 26;
const BLOCK: usize = 256;

/// The escape-hatch kernel: CUDA-C, JIT-compiled by NVRTC at startup.
/// Structure comments are the WHAT; the body is yours.
const REDUCE_WARP_CUDA: &str = r#"
// Warp-shuffle sum reduction.
//
// TODO(Day 2), the three levels:
//   warp:   v += __shfl_down_sync(0xffffffff, v, 16);  then 8, 4, 2, 1 —
//           lane 0 ends with the warp total. No shared memory, no syncs.
//   block:  lane 0 of each warp writes its sum to shared[warp_id]
//           (BLOCK/32 slots); __syncthreads(); warp 0 loads those values
//           and shuffles them down.
//   grid:   each thread first accumulates a register value over a
//           GRID-STRIDE loop across the input; the host runs a second
//           single-block pass over the per-block partials (same kernel).
extern "C" __global__ void reduce_warp(const float* in, float* out, int n) {
    // TODO(Day 2)
}
"#;

fn make_input(n: usize) -> Vec<f32> {
    // Non-negative on purpose: sum == asum, so Sasum is a fair baseline.
    (0..n).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0).collect()
}

fn reduce_two_pass(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    d_in: &CudaSlice<f32>,
    d_partial: &mut CudaSlice<f32>,
    d_final: &mut CudaSlice<f32>,
    n: usize,
    grid: u32,
) -> Result<(), DriverError> {
    let cfg1 = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK as u32, 1, 1), shared_mem_bytes: 0 };
    let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (BLOCK as u32, 1, 1), shared_mem_bytes: 0 };
    let n_i32 = n as i32;
    let grid_i32 = grid as i32;
    // SAFETY: matches the CUDA-C signature (const float*, float*, int);
    // buffers do not alias.
    unsafe {
        stream.launch_builder(f).arg(d_in).arg(&mut *d_partial).arg(&n_i32).launch(cfg1)?;
        stream.launch_builder(f).arg(&*d_partial).arg(&mut *d_final).arg(&grid_i32).launch(cfg2)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let grid = (sms as u32) * 4;
    println!("device: {} ({sms} SMs) -> grid {grid}", ctx.name()?);

    // NVRTC: CUDA-C -> PTX at runtime. This is the whole escape hatch.
    let ptx = cudarc::nvrtc::compile_ptx(REDUCE_WARP_CUDA)?;
    let module = ctx.load_module(ptx)?;
    let f = module.load_function("reduce_warp")?;

    let h = make_input(N);
    let cpu_sum: f64 = h.iter().map(|&v| v as f64).sum();

    let d_in = stream.memcpy_stod(&h)?;
    let mut d_partial = stream.alloc_zeros::<f32>(grid as usize)?;
    let mut d_final = stream.alloc_zeros::<f32>(1)?;

    // ---- correctness -------------------------------------------------------
    reduce_two_pass(&stream, &f, &d_in, &mut d_partial, &mut d_final, N, grid)?;
    let gpu_sum = stream.memcpy_dtov(&d_final)?[0] as f64;
    let rel = (gpu_sum - cpu_sum).abs() / cpu_sum.abs();
    println!("gpu {gpu_sum:.6}  cpu {cpu_sum:.6}  rel err {rel:.3e}");
    if rel > 1e-4 {
        eprintln!("FAIL: reduction incorrect");
        std::process::exit(1);
    }
    println!("correctness: OK");

    // ---- mine --------------------------------------------------------------
    let mut mine = BenchResult {
        name: "reduce_warp".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: N as i64 * 4,
        ..Default::default()
    };
    mine.median_ms = benchmark(&ctx, &stream, || {
        reduce_two_pass(&stream, &f, &d_in, &mut d_partial, &mut d_final, N, grid)
    }, 5, mine.runs as usize)?;
    finalize(&mut mine);

    // ---- cuBLAS Sasum baseline (COMPLETE — the oracle, not the answer) -----
    let blas = CudaBlas::new(stream.clone())?;
    let mut host_result = 0f32;
    let mut sasum = || -> Result<(), DriverError> {
        // SAFETY: d_in is a live, contiguous device buffer of N f32s; the
        // cuBLAS handle is bound to `stream`; host_result outlives the call
        // (Sasum blocks until the scalar result is written back).
        unsafe {
            let status = cudarc::cublas::sys::cublasSasum_v2(
                *blas.handle(),
                N as i32,
                *d_in.device_ptr() as *const f32,
                1,
                &mut host_result,
            );
            assert_eq!(status, cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
        Ok(())
    };
    let mut base = BenchResult {
        name: "cublas_sasum".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: N as i64 * 4,
        ..Default::default()
    };
    base.median_ms = benchmark(&ctx, &stream, &mut sasum, 5, base.runs as usize)?;
    finalize(&mut base);
    let rel_base = (host_result as f64 - cpu_sum).abs() / cpu_sum.abs();
    assert!(rel_base < 1e-4, "Sasum disagrees with CPU — inputs not non-negative?");

    let pct = 100.0 * base.median_ms / mine.median_ms;
    println!("mine    {:.4} ms ({:.1} GB/s)", mine.median_ms, mine.gbps);
    println!("cublas  {:.4} ms ({:.1} GB/s)", base.median_ms, base.gbps);
    println!("you are at {pct:.1}% of cuBLAS Sasum  (target >= 80%)");

    write_json("results/reduce_warp.json", &[mine, base])?;
    std::process::exit(if pct >= 80.0 { 0 } else { 2 });
}
