//! Day 3: map the memory system empirically. Host COMPLETE.
//!
//! Two experiments, one JSON file, one chart (bench/plot_results.py):
//!   1. STRIDED READS:  thread i reads in[i * stride]   (stride 1..=32)
//!      -> effective bandwidth collapses as each 128-byte transaction
//!         delivers fewer useful bytes.
//!   2. OFFSET READS:   thread i reads in[i + offset]   (offset 0..=32)
//!      -> misalignment costs extra transactions per warp; measure what
//!         Blackwell's L1 actually does rather than trusting old slides.
//!
//! Run: cargo run --release --bin bandwidth_sweep

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use gpu_bench::{benchmark, finalize, write_json, BenchResult};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const N_THREADS: usize = 1 << 24; // 16M reads per launch
const MAX_STRIDE: usize = 32;
const BLOCK: u32 = kernels::BLOCK as u32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("device: {}", ctx.name()?);

    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX))?;
    let f_strided = module.load_function("strided_read")?;
    let f_offset = module.load_function("offset_read")?;

    let in_elems = N_THREADS * MAX_STRIDE + 64;
    let d_in = stream.alloc_zeros::<f32>(in_elems)?;
    let mut d_out = stream.alloc_zeros::<f32>(N_THREADS)?;

    let grid = (N_THREADS as u32).div_ceil(BLOCK);
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };

    // Useful bytes actually requested per launch: 1 read + 1 write per thread.
    let useful_bytes = 2 * N_THREADS as i64 * 4;

    let mut results: Vec<BenchResult> = Vec::new();

    for stride in 1..=MAX_STRIDE {
        let mut r = BenchResult {
            name: "strided_read".into(),
            n: N_THREADS as i64,
            runs: 50,
            bytes_moved: useful_bytes,
            meta: format!("\"stride\": {stride}, \"offset\": 0"),
            ..Default::default()
        };
        r.median_ms = benchmark(&ctx, &stream, || {
            // SAFETY: matches kernels::strided_read(input, out, n_threads,
            // stride); max index (N_THREADS-1)*MAX_STRIDE < in_elems.
            unsafe {
                stream.launch_builder(&f_strided)
                    .arg(&d_in).arg(&mut d_out).arg(&N_THREADS).arg(&stride)
                    .launch(cfg)
            }
        }, 5, r.runs as usize)?;
        finalize(&mut r);
        println!("stride {stride:2} : {:8.2} GB/s effective", r.gbps);
        results.push(r);
    }

    for offset in 0..=32usize {
        let mut r = BenchResult {
            name: "offset_read".into(),
            n: N_THREADS as i64,
            runs: 50,
            bytes_moved: useful_bytes,
            meta: format!("\"stride\": 1, \"offset\": {offset}"),
            ..Default::default()
        };
        r.median_ms = benchmark(&ctx, &stream, || {
            // SAFETY: matches kernels::offset_read(input, out, n_threads,
            // offset); max index N_THREADS-1+32 < in_elems.
            unsafe {
                stream.launch_builder(&f_offset)
                    .arg(&d_in).arg(&mut d_out).arg(&N_THREADS).arg(&offset)
                    .launch(cfg)
            }
        }, 5, r.runs as usize)?;
        finalize(&mut r);
        println!("offset {offset:2} : {:8.2} GB/s effective", r.gbps);
        results.push(r);
    }

    write_json("results/bandwidth_sweep.json", &results)?;
    Ok(())
}
