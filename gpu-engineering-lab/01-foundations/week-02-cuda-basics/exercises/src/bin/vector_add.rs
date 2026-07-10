//! Day 1: c[i] = a[i] + b[i]. Host plumbing COMPLETE except one TODO (grid).
//!
//! Run: cargo run --release --bin vector_add

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const N: usize = 1 << 26; // 64M floats — big enough to be DRAM-bound
const BLOCK: u32 = kernels::BLOCK as u32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("device: {}", ctx.name()?);

    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let f = module.load_function("vector_add")?;

    let ha: Vec<f32> = (0..N).map(|i| 0.5 * i as f32).collect();
    let hb: Vec<f32> = (0..N).map(|i| 1.0 - 0.25 * i as f32).collect();
    let da = stream.memcpy_stod(&ha)?;
    let db = stream.memcpy_stod(&hb)?;
    let mut dc = stream.alloc_zeros::<f32>(N)?;

    // TODO(Day 1): grid size — the ceiling-division idiom over N and BLOCK.
    let grid: u32 = 0; // TODO(Day 1)
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };

    // SAFETY: arg order matches kernels::vector_add(a, b, c, n) exactly.
    unsafe {
        stream.launch_builder(&f).arg(&da).arg(&db).arg(&mut dc).arg(&N).launch(cfg)?;
    }

    // ---- correctness first, speed second ----------------------------------
    let hc = stream.memcpy_dtov(&dc)?;
    for i in (0..N).step_by(997) {
        let want = ha[i] + hb[i];
        if (hc[i] - want).abs() > 1e-5 {
            eprintln!("MISMATCH at {i}: got {} want {want}", hc[i]);
            std::process::exit(1);
        }
    }
    println!("correctness: OK");

    // ---- benchmark ---------------------------------------------------------
    let mut r = BenchResult {
        name: "vector_add".into(),
        n: N as i64,
        runs: 50,
        bytes_moved: 3 * N as i64 * 4, // 2 reads + 1 write
        ..Default::default()
    };
    r.median_ms = benchmark(&ctx, &stream, || {
        // SAFETY: same launch as the verified one above.
        unsafe {
            stream.launch_builder(&f).arg(&da).arg(&db).arg(&mut dc).arg(&N).launch(cfg)
        }
    }, 5, r.runs as usize)?;
    finalize(&mut r);
    println!("median {:.4} ms  ->  {:.1} GB/s", r.median_ms, r.gbps);
    write_json("results/vector_add.json", &[r])?;
    Ok(())
}
