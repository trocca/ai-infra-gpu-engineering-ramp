//! rust-hello-gpu — Week 01 Day 5: first GPU launch from Rust.
//!
//! What you build (host side ONLY — the kernel is provided as PTX below):
//!   1. create a CUDA context and print the device name
//!   2. build deterministic inputs:  x[i] = 0.01 * i,  y[i] = 1.0
//!   3. copy x and y to the device
//!   4. launch the provided SAXPY kernel:  y = ALPHA * x + y
//!   5. copy y back and print ONE json line to stdout (schema below)
//!
//! Verified by ../tests/test_rust_hello.py, which runs
//! `cargo run --release` and checks the JSON against a NumPy reference.
//!
//! Output schema (single line on stdout, nothing else on stdout — put any
//! debug prints on stderr with eprintln!):
//!   {"device": "<name>", "n": 1048576, "alpha": 2.5,
//!    "y_head": [<first 8 outputs>], "y_sum": <f64 sum of all outputs>}

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

const N: usize = 1 << 20;
const ALPHA: f32 = 2.5;

/// The SAXPY kernel, hand-written PTX. PROVIDED COMPLETE — read it once:
/// this is the ISA-ish format every kernel in this repo ultimately becomes,
/// whether it started as Rust (rustc_codegen_nvvm), CUDA-C (nvcc/NVRTC), or
/// hand-written like here. `.target sm_90` JIT-compiles forward to sm_120.
const SAXPY_PTX: &str = r#"
.version 8.3
.target sm_90
.address_size 64

.visible .entry saxpy(
    .param .f32 alpha,
    .param .u64 x,
    .param .u64 y,
    .param .u32 n
)
{
    .reg .pred %p1;
    .reg .f32  %f<5>;
    .reg .b32  %r<6>;
    .reg .b64  %rd<7>;

    ld.param.f32 %f1, [alpha];
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.u32 %r2, [n];
    // i = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;
    // if (i >= n) return;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f2, [%rd6];      // x[i]
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f3, [%rd6];      // y[i]
    fma.rn.f32 %f4, %f1, %f2, %f3;  // alpha * x[i] + y[i]
    st.global.f32 [%rd6], %f4;
DONE:
    ret;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ------------------------------------------------------------------
    // TODO(Day 5) step 1: context + device name.
    //   let ctx = CudaContext::new(0)?;        (Arc<CudaContext>)
    //   ctx.name()? gives the device string — expect the RTX 5090 Laptop GPU.
    //   Grab the default stream: ctx.default_stream().
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(Day 5) step 2: load the PTX and get the kernel handle.
    //   ctx.load_module(cudarc::nvrtc::Ptx::from_src(SAXPY_PTX))?
    //   then module.load_function("saxpy")?.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(Day 5) step 3: host inputs (x[i] = 0.01 * i as f32, y[i] = 1.0)
    // and device copies via stream.memcpy_stod(&vec)?.
    // Note what you get back: CudaSlice<f32> — an OWNING handle whose Drop
    // frees the device memory. That's RAII replacing cudaFree; say so in
    // RESULTS.md.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(Day 5) step 4: launch.
    //   Grid/block: 256 threads per block, ceiling-division for the grid
    //   (the same idiom as every kernel week to come).
    //   let cfg = LaunchConfig { grid_dim: (?, 1, 1), block_dim: (256, 1, 1),
    //                            shared_mem_bytes: 0 };
    //   Args must match the PTX .param order exactly: alpha (f32),
    //   x (pointer), y (pointer), n (u32).
    //   stream.launch_builder(&f).arg(&ALPHA).arg(&x_dev).arg(&mut y_dev)
    //         .arg(&n_u32) ... then unsafe { .launch(cfg)? }.
    //   Why unsafe? The type system cannot verify the arg list against the
    //   PTX signature — YOU are the proof. Keep the block minimal.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(Day 5) step 5: copy back (stream.memcpy_dtov(&y_dev)?), then
    // print the single JSON line described at the top of this file.
    // Format floats with enough digits: {:.6} for y_head values, {} for the
    // f64 sum. No trailing prints on stdout afterwards.
    // ------------------------------------------------------------------

    todo!("Day 5: implement steps 1-5, then delete this line");
}
