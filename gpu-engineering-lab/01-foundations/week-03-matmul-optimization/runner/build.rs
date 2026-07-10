//! Builds ../kernels (Rust) into PTX at compile time. COMPLETE.
//! SKIP_PTX=1 writes a stub so `cargo check`/clippy work without the NVVM
//! backend (CI / GPU-less machines).

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../kernels/src");
    println!("cargo:rerun-if-env-changed=SKIP_PTX");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("sgemm_kernels.ptx");

    if std::env::var("SKIP_PTX").is_ok() {
        std::fs::write(&out, "// SKIP_PTX=1 — kernels were not built\n").unwrap();
        println!("cargo:warning=SKIP_PTX set: wrote stub sgemm_kernels.ptx");
        return;
    }

    cuda_builder::CudaBuilder::new("../kernels")
        .copy_to(&out)
        .build()
        .expect("Rust-CUDA kernel build failed — see week-02 README toolchain notes");
}
