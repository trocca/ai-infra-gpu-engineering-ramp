//! Day 1: y = alpha*x + y with a grid-stride kernel. Host COMPLETE except
//! the grid-size TODO.
//!
//! Run: cargo run --release --bin saxpy

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const N: usize = 1 << 26;
const ALPHA: f32 = 2.5;
const BLOCK: u32 = kernels::BLOCK as u32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    println!("device: {} ({sms} SMs)", ctx.name()?);

    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let f = module.load_function("saxpy")?;

    let hx: Vec<f32> = (0..N).map(|i| 0.001 * i as f32).collect();
    let hy: Vec<f32> = vec![1.0; N];
    let href: Vec<f32> = hx.iter().zip(&hy).map(|(x, y)| ALPHA * x + y).collect();
    let dx = stream.memcpy_stod(&hx)?;
    let mut dy = stream.memcpy_stod(&hy)?;

    // TODO(Day 1): with a grid-stride kernel the grid comes from the DEVICE,
    // not from N — e.g. 4 blocks per SM. Try a few multipliers, note the
    // effect in RESULTS.md.
    let grid: u32 = 0; // TODO(Day 1)
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };

    // SAFETY: arg order matches kernels::saxpy(alpha, x, y, n).
    unsafe {
        stream.launch_builder(&f).arg(&ALPHA).arg(&dx).arg(&mut dy).arg(&N).launch(cfg)?;
    }

    let hout = stream.memcpy_dtov(&dy)?;
    for i in (0..N).step_by(1013) {
        if (hout[i] - href[i]).abs() > 1e-4 {
            eprintln!("MISMATCH at {i}: got {} want {}", hout[i], href[i]);
            std::process::exit(1);
        }
    }
    println!("correctness: OK");

    // NOTE: y is mutated each timed launch — fine for timing (values grow;
    // the memory traffic is identical).
    let mut r = BenchResult {
        name: "saxpy".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: 3 * N as i64 * 4, // read x, read y, write y
        ..Default::default()
    };
    r.median_ms = benchmark(&ctx, &stream, || {
        // SAFETY: same launch as the verified one above.
        unsafe {
            stream.launch_builder(&f).arg(&ALPHA).arg(&dx).arg(&mut dy).arg(&N).launch(cfg)
        }
    }, 5, r.runs as usize)?;
    finalize(&mut r);
    println!("median {:.4} ms  ->  {:.1} GB/s", r.median_ms, r.gbps);
    write_json("results/saxpy.json", &[r])?;
    Ok(())
}
