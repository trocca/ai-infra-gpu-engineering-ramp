//! Day 2, rung 2: sequential addressing + grid-stride load.
//! Host COMPLETE — with a grid-stride kernel the grid is FIXED (~4 blocks
//! per SM) and reduction finishes in exactly two passes.
//!
//! Run: cargo run --release --bin reduce_shared

use std::sync::Arc;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError,
                     LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const N: usize = 1 << 26;
const BLOCK: usize = kernels::BLOCK;

fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect()
}

/// Two passes: grid blocks -> partials, one block -> final at partials[0].
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
    let grid_n = grid as usize;
    // SAFETY: arg order matches kernels::reduce_shared(input, out, n);
    // buffers do not alias.
    unsafe {
        stream.launch_builder(f).arg(d_in).arg(&mut *d_partial).arg(&n).launch(cfg1)?;
        stream.launch_builder(f).arg(&*d_partial).arg(&mut *d_final).arg(&grid_n).launch(cfg2)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let grid = (sms as u32) * 4; // ~4 resident blocks per SM; try other values
    println!("device: {} ({sms} SMs) -> grid {grid}", ctx.name()?);

    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let f = module.load_function("reduce_shared")?;

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

    // ---- benchmark ---------------------------------------------------------
    let mut r = BenchResult {
        name: "reduce_shared".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: N as i64 * 4,
        ..Default::default()
    };
    r.median_ms = benchmark(&ctx, &stream, || {
        reduce_two_pass(&stream, &f, &d_in, &mut d_partial, &mut d_final, N, grid)
    }, 5, r.runs as usize)?;
    finalize(&mut r);
    println!("median {:.4} ms  ->  {:.1} GB/s effective", r.median_ms, r.gbps);
    write_json("results/reduce_shared.json", &[r])?;
    Ok(())
}
