//! gpu-bench — shared benchmark harness for the whole repo. COMPLETE.
//!
//! The Rust port of what a C++ repo would keep in a `timing.cuh`:
//!   * [`benchmark`]  — CUDA-event timing with warmup + median
//!   * [`BenchResult`] / [`write_json`] — the repo's standard JSON schema
//!   * [`median`]     — the (unit-tested, CPU-safe) statistic itself
//!
//! Used by weeks 02, 03 and 04. Read every line; you must be able to defend
//! this methodology in an interview.
//!
//! Methodology in one breath: warmup launches absorb JIT/context/clock ramp;
//! CUDA events measure GPU-side time (not host launch latency); the median
//! is robust to the stragglers a power-limited laptop GPU produces.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError};

/// One benchmark record. Serialized field-for-field into the repo's JSON
/// schema — bench/plot_*.py scripts across all weeks consume exactly this.
#[derive(Debug, Clone, Default)]
pub struct BenchResult {
    pub name: String,
    /// Problem size in elements.
    pub n: i64,
    pub median_ms: f64,
    /// Number of timed iterations that produced the median.
    pub runs: u32,
    /// Bytes read+written per iteration (0 if not meaningful).
    pub bytes_moved: i64,
    /// Derived by [`finalize`]: bytes_moved / median time.
    pub gbps: f64,
    /// Caller-computed if meaningful (e.g. 2*M*N*K for GEMM).
    pub gflops: f64,
    /// Extra JSON fields as a raw fragment, e.g. `"stride": 4, "offset": 0`.
    pub meta: String,
}

/// Median of a slice (not in-place). CPU-safe, unit-tested below.
pub fn median(samples: &[f32]) -> f64 {
    assert!(!samples.is_empty(), "median of zero samples");
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("NaN in timing samples"));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m] as f64
    } else {
        0.5 * (s[m - 1] as f64 + s[m] as f64)
    }
}

/// Time `launch()` — a closure that must enqueue ALL GPU work for one
/// iteration onto `stream` — with CUDA events. Returns the median ms.
///
/// `launch` is `FnMut` so callers can capture device buffers mutably.
pub fn benchmark<F: FnMut() -> Result<(), DriverError>>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    mut launch: F,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<f64, DriverError> {
    let start = ctx.new_event(None)?;
    let stop = ctx.new_event(None)?;

    for _ in 0..warmup_iters {
        launch()?;
    }
    stream.synchronize()?;

    let mut times_ms = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        start.record(stream)?;
        launch()?;
        stop.record(stream)?;
        stop.synchronize()?;
        times_ms.push(start.elapsed_ms(&stop)?);
    }
    Ok(median(&times_ms))
}

/// Fill the derived bandwidth field from `bytes_moved` + `median_ms`.
pub fn finalize(r: &mut BenchResult) {
    if r.bytes_moved > 0 && r.median_ms > 0.0 {
        r.gbps = r.bytes_moved as f64 / (r.median_ms * 1e-3) / 1e9;
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Write results as a JSON array, creating parent dirs. Overwrites `path`.
pub fn write_json(path: impl AsRef<Path>, results: &[BenchResult]) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "  {{\"name\": \"{}\", \"n\": {}, \"median_ms\": {}, \"runs\": {}, \
             \"bytes_moved\": {}, \"gbps\": {}, \"gflops\": {}",
            escape(&r.name), r.n, r.median_ms, r.runs, r.bytes_moved, r.gbps, r.gflops
        ));
        if !r.meta.is_empty() {
            out.push_str(", ");
            out.push_str(&r.meta);
        }
        out.push('}');
        if i + 1 < results.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("]\n");
    fs::write(path, &out)?;
    println!("wrote {} results -> {}", results.len(), path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU-safe unit tests (these run in CI, where there is no GPU).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[7.5]), 7.5);
    }

    #[test]
    fn json_schema_roundtrips_key_names() {
        let mut r = BenchResult {
            name: "unit".into(),
            n: 42,
            median_ms: 1.5,
            runs: 50,
            bytes_moved: 1_000_000_000,
            meta: "\"stride\": 4".into(),
            ..Default::default()
        };
        finalize(&mut r);
        assert!((r.gbps - 1_000_000_000.0 / 1.5e-3 / 1e9).abs() < 1e-9);

        let dir = std::env::temp_dir().join("gpu_bench_test");
        let path = dir.join("out.json");
        write_json(&path, &[r]).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        for key in ["\"name\"", "\"n\"", "\"median_ms\"", "\"runs\"",
                    "\"bytes_moved\"", "\"gbps\"", "\"gflops\"", "\"stride\""] {
            assert!(text.contains(key), "missing {key} in {text}");
        }
    }
}
