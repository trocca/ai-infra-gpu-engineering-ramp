//! rusty_kernels._rusty — PyO3 extension. COMPLETE — do not edit until the
//! stretch goals (stream-ordered launches, NVRTC overrides).
//!
//! Interop contract (the data_ptr choice — see README "Architecture"):
//!   * Python passes RAW DEVICE ADDRESSES (`tensor.data_ptr()`) plus shapes
//!     and a dtype tag ("f32" | "f64" | "f16"). No copies, no ownership
//!     transfer: Python guarantees the tensors are CUDA, contiguous, and
//!     alive for the duration of the call (ops.py enforces this).
//!   * v0 stream story: ops.py synchronizes torch's current stream BEFORE
//!     calling in; every function here synchronizes the extension's stream
//!     BEFORE returning. Correct, simple, costs ~µs per call — measured and
//!     documented on Day 4; upgrading to stream-ordered is a stretch goal.
//!
//! Kernel names resolve as "{op}_{dtype}" against the PTX built from
//! ../kernels (e.g. "layernorm_fwd_f64"). A todo!()-free kernel body is the
//! learner's job; this file only validates, resolves, and launches.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig,
                     PushKernelArg};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rusty_kernels.ptx"));
const BLOCK: u32 = 256;

struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();

fn gpu() -> PyResult<&'static Gpu> {
    GPU.get_or_init(|| {
        let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext: {e}"))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(PTX))
            .map_err(|e| format!("load_module (was the crate built with SKIP_PTX?): {e}"))?;
        Ok(Gpu { ctx, stream, module })
    })
    .as_ref()
    .map_err(|e| PyRuntimeError::new_err(e.clone()))
}

fn func(op: &str, dtype: &str) -> PyResult<CudaFunction> {
    match dtype {
        "f32" | "f64" | "f16" => {}
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unsupported dtype tag {other:?} (want f32/f64/f16)"
            )))
        }
    }
    let name = format!("{op}_{dtype}");
    gpu()?
        .module
        .load_function(&name)
        .map_err(|e| PyRuntimeError::new_err(format!("kernel {name} not found: {e}")))
}

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn row_cfg(rows: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
}

/// Day 1 smoke test: out = copy(input), n elements of dtype.
#[pyfunction]
fn passthrough(x_ptr: u64, y_ptr: u64, n: usize, dtype: &str) -> PyResult<()> {
    let g = gpu()?;
    let f = func("passthrough", dtype)?;
    let grid = (n as u32).div_ceil(BLOCK).min(4096);
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: ops.py guarantees x_ptr/y_ptr address live, contiguous CUDA
    // buffers of >= n elements of `dtype`, and torch's stream is synced.
    unsafe {
        g.stream.launch_builder(&f).arg(&x_ptr).arg(&y_ptr).arg(&n).launch(cfg).map_err(err)?;
    }
    g.stream.synchronize().map_err(err)
}

/// Fused softmax forward over the last dim of a contiguous (rows, cols).
#[pyfunction]
fn softmax_forward(x_ptr: u64, y_ptr: u64, rows: usize, cols: usize, dtype: &str) -> PyResult<()> {
    let g = gpu()?;
    let f = func("softmax_fwd", dtype)?;
    // SAFETY: same contract as above; one block per row.
    unsafe {
        g.stream.launch_builder(&f)
            .arg(&x_ptr).arg(&y_ptr).arg(&rows).arg(&cols)
            .launch(row_cfg(rows)).map_err(err)?;
    }
    g.stream.synchronize().map_err(err)
}

/// Fused LayerNorm forward. mean/rstd are (rows,) buffers of the SAME dtype
/// (allocated Python-side), saved for backward.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn layernorm_forward(
    x_ptr: u64, w_ptr: u64, b_ptr: u64,
    y_ptr: u64, mean_ptr: u64, rstd_ptr: u64,
    rows: usize, cols: usize, eps: f64, dtype: &str,
) -> PyResult<()> {
    let g = gpu()?;
    let f = func("layernorm_fwd", dtype)?;
    // SAFETY: contract as above; eps is passed at the kernel's precision.
    unsafe {
        let mut b = g.stream.launch_builder(&f);
        b.arg(&x_ptr).arg(&w_ptr).arg(&b_ptr)
         .arg(&y_ptr).arg(&mean_ptr).arg(&rstd_ptr)
         .arg(&rows).arg(&cols);
        match dtype {
            "f64" => { b.arg(&eps); }
            _ => { let eps32 = eps as f32; b.arg(&eps32); }
        }
        b.launch(row_cfg(rows)).map_err(err)?;
    }
    g.stream.synchronize().map_err(err)
}

/// Fused LayerNorm backward: fills dx (rows, cols) and dweight/dbias (cols,).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn layernorm_backward(
    dy_ptr: u64, x_ptr: u64, w_ptr: u64, mean_ptr: u64, rstd_ptr: u64,
    dx_ptr: u64, dw_ptr: u64, db_ptr: u64,
    rows: usize, cols: usize, dtype: &str,
) -> PyResult<()> {
    let g = gpu()?;
    let f_dx = func("layernorm_bwd_dx", dtype)?;
    let f_dwdb = func("layernorm_bwd_dwdb", dtype)?;
    let col_cfg = LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: contract as above; dx kernel is one block per row, dw/db
    // kernel one thread per column.
    unsafe {
        g.stream.launch_builder(&f_dx)
            .arg(&dy_ptr).arg(&x_ptr).arg(&w_ptr).arg(&mean_ptr).arg(&rstd_ptr)
            .arg(&dx_ptr).arg(&rows).arg(&cols)
            .launch(row_cfg(rows)).map_err(err)?;
        g.stream.launch_builder(&f_dwdb)
            .arg(&dy_ptr).arg(&x_ptr).arg(&mean_ptr).arg(&rstd_ptr)
            .arg(&dw_ptr).arg(&db_ptr).arg(&rows).arg(&cols)
            .launch(col_cfg).map_err(err)?;
    }
    g.stream.synchronize().map_err(err)
}

/// Device name — used by the import smoke test and for debugging.
#[pyfunction]
fn device_name() -> PyResult<String> {
    gpu()?.ctx.name().map_err(err)
}

#[pymodule]
fn _rusty(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layernorm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layernorm_backward, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    Ok(())
}
