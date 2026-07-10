//! Day 2, rung 1: sum 2^26 floats, interleaved-addressing kernel.
//! Host driver (multi-pass re-launch loop) COMPLETE — kernel body is yours.
//!
//! Run: cargo run --release --bin reduce_naive

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError,
                     LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const N: usize = 1 << 26;
const BLOCK: usize = kernels::BLOCK;

/// Deterministic pseudo-random input in [-0.5, 0.5) — same on CPU and GPU.
fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect()
}

/// Reduce until one value remains, re-launching the block-level kernel on
/// the partial sums (ping-pong between the two scratch buffers). COMPLETE —
/// the same driver shape serves rung 2 too. Returns true if the final value
/// sits in scratch_a[0], false if in scratch_b[0].
fn reduce_on_device(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    d_in: &CudaSlice<f32>,
    scratch_a: &mut CudaSlice<f32>,
    scratch_b: &mut CudaSlice<f32>,
    n: usize,
) -> Result<bool, DriverError> {
    let mut remaining = n;
    let mut pass = 0usize; // even pass writes a, odd pass writes b
    while remaining > 1 {
        let blocks = remaining.div_ceil(BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arg order matches kernels::reduce_naive(input, out, n);
        // src and dst never alias (strict a/b ping-pong).
        unsafe {
            let mut b = stream.launch_builder(f);
            match pass {
                0 => b.arg(d_in).arg(&mut *scratch_a).arg(&remaining),
                p if p % 2 == 1 => b.arg(&*scratch_a).arg(&mut *scratch_b).arg(&remaining),
                _ => b.arg(&*scratch_b).arg(&mut *scratch_a).arg(&remaining),
            };
            b.launch(cfg)?;
        }
        remaining = blocks;
        pass += 1;
    }
    Ok(pass % 2 == 1) // last write was to a iff an odd number of passes ran
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let f = module.load_function("reduce_naive")?;

    let h = make_input(N);
    let cpu_sum: f64 = h.iter().map(|&v| v as f64).sum();

    let d_in = stream.memcpy_stod(&h)?;
    let max_blocks = N.div_ceil(BLOCK);
    let mut scratch_a = stream.alloc_zeros::<f32>(max_blocks)?;
    let mut scratch_b = stream.alloc_zeros::<f32>(max_blocks)?;

    // ---- correctness -------------------------------------------------------
    let in_a = reduce_on_device(&stream, &f, &d_in, &mut scratch_a, &mut scratch_b, N)?;
    let result_buf = if in_a { &scratch_a } else { &scratch_b };
    let gpu_sum = stream.memcpy_dtov(result_buf)?[0] as f64;
    let rel = (gpu_sum - cpu_sum).abs() / cpu_sum.abs();
    println!("gpu {gpu_sum:.6}  cpu {cpu_sum:.6}  rel err {rel:.3e}");
    if rel > 1e-4 {
        eprintln!("FAIL: reduction incorrect");
        std::process::exit(1);
    }
    println!("correctness: OK");

    // ---- benchmark ---------------------------------------------------------
    let mut r = BenchResult {
        name: "reduce_naive".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: N as i64 * 4, // dominant traffic: one read of the input
        ..Default::default()
    };
    r.median_ms = benchmark(&ctx, &stream, || {
        reduce_on_device(&stream, &f, &d_in, &mut scratch_a, &mut scratch_b, N).map(|_| ())
    }, 5, r.runs as usize)?;
    finalize(&mut r);
    println!("median {:.4} ms  ->  {:.1} GB/s effective", r.median_ms, r.gbps);
    write_json("results/reduce_naive.json", &[r])?;
    Ok(())
}
