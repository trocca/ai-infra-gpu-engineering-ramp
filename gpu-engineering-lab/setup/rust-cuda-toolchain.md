# Rust + CUDA toolchain (WSL2)

The Rust GPU ecosystem in early 2026, and what this repo uses each piece for:

| Crate / project | What it is | Used in |
|-----------------|------------|---------|
| **cudarc** | Safe-ish Rust bindings to the CUDA driver API, NVRTC, cuBLAS, NCCL. Loads libcuda dynamically; supports CUDA 12/13. The host-side workhorse. | weeks 2–4, 8 |
| **Rust-CUDA** (`rustc_codegen_nvvm`, Rust-GPU org) | Write the *kernels themselves* in Rust; compiles Rust → NVVM IR → PTX. Revived and active, but expects a pinned nightly — treat it as a per-project pinned tool, not a global install. | weeks 2–3 |
| **CubeCL** (tracel-ai) | Burn's kernel language: write kernels in a Rust dialect, JIT to CUDA/ROCm/wgpu. The most production-real "kernels in Rust" today. | week 3 stretch |
| **Candle** (HuggingFace) | Minimalist Rust inference framework; loads safetensors/GGUF, has CUDA + flash-attn features, `candle-transformers` ships Llama/Qwen implementations. | week 8, 12 |
| **PyO3 + maturin** | Rust ↔ Python bindings and wheel builds. | week 4 |
| **axum + tokio** | The async HTTP stack for the serving weeks. | week 8, 11 |

Escape-hatch policy (sanity-preserving, stated in every kernel week): if Rust-CUDA's
toolchain fights you for more than ~30 min on a given kernel, write that kernel as
CUDA-C source compiled at runtime via **NVRTC through cudarc**, keep the host code in
Rust, and file the friction in the README's "what didn't work" section. The learning
target is GPU architecture *and* Rust systems engineering — not toolchain archaeology.

## Install

```bash
# inside WSL2 Ubuntu (CUDA toolkit ≥ 12.8 already installed per local-wsl2-cuda.md)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup component add clippy rustfmt

# quick sanity: cudarc sees the GPU
cargo new gpu-hello && cd gpu-hello
cargo add cudarc --features cuda-13000   # match your toolkit: cuda-12080 / cuda-13000 …
cat > src/main.rs <<'EOF'
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    println!("GPU 0: {}", ctx.name()?);
    Ok(())
}
EOF
cargo run   # expect: GPU 0: NVIDIA GeForce RTX 5090 Laptop GPU
```

Pin exact crate versions inside each week's `Cargo.toml` — this ecosystem moves
monthly, and reproducibility is part of the repo contract. Check each project's
current docs before starting a week:

- cudarc: https://github.com/coreylowman/cudarc
- Rust-CUDA: https://github.com/Rust-GPU/Rust-CUDA (see its guide for the pinned nightly)
- CubeCL: https://github.com/tracel-ai/cubecl
- Candle: https://github.com/huggingface/candle
- maturin: https://www.maturin.rs

## Rust learning ramp (week 1, days 4–5, and evenings after)

1. `rustlings` — https://github.com/rust-lang/rustlings (do it all; ~6–8 h total)
2. The Rust Book ch. 1–10 + 15 (smart pointers) + 19.1 (unsafe) — skim-read, return as needed
3. Jon Gjengset's "Considering Rust" talk for the mental model

You don't need to be fluent before week 2 — the kernel weeks teach the borrow checker
by fire. `unsafe` blocks around raw device pointers are expected and normal in this
domain; the discipline is keeping them small and documented.

## Nsight note

Profilers operate on PTX/SASS and don't care what language emitted it: `nsys profile`
and `ncu` work identically on Rust binaries. Build with `--release` and
`debug = true` in `[profile.release]` so kernels keep symbol names.
