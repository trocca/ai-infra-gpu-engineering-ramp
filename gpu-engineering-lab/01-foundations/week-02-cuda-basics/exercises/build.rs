//! Builds ../kernels (Rust) into PTX at compile time. COMPLETE.
//!
//! The bins embed the PTX via include_str!(concat!(env!("OUT_DIR"),
//! "/kernels.ptx")).
//!
//! SKIP_PTX=1 (set by CI and useful on GPU-less machines) writes a stub PTX
//! instead, so `cargo check` / `cargo clippy` work without the NVVM backend
//! — the bins will build but obviously cannot run kernels in that mode.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../kernels/src");
    println!("cargo:rerun-if-env-changed=SKIP_PTX");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");

    if std::env::var("SKIP_PTX").is_ok() {
        std::fs::write(&out, "// SKIP_PTX=1 — kernels were not built\n").unwrap();
        println!("cargo:warning=SKIP_PTX set: wrote stub kernels.ptx (check/clippy only)");
        return;
    }

    cuda_builder::CudaBuilder::new("../kernels")
        .copy_to(&out)
        .build()
        .expect(
            "Rust-CUDA kernel build failed. Checklist: pinned nightly installed \
             (rust-toolchain.toml), Rust-CUDA git rev pinned in Cargo.toml, LLVM \
             prerequisites per the Rust-CUDA guide. Escape hatch policy: \
             ../../setup/rust-cuda-toolchain.md",
        );
}
