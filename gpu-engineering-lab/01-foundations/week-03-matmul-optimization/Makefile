# Week 03 — SGEMM ladder (Rust host + Rust kernels; cuBLAS via cudarc)
# Requires: CUDA >= 12.8, pinned nightly (rust-toolchain.toml), pins resolved.

PY ?= python3

.PHONY: build bench plot check clean

build:
	cargo build --release

# One command reproduces every published number (repo contract).
bench: build
	cargo run --release --bin sgemm_ladder
	$(PY) bench/plot_ladder.py results/sgemm_ladder.json

plot:
	$(PY) bench/plot_ladder.py results/sgemm_ladder.json

# GPU-less machines / CI: type-check + lint without the NVVM backend.
check:
	SKIP_PTX=1 cargo clippy --workspace --all-targets

clean:
	cargo clean
	rm -rf results
