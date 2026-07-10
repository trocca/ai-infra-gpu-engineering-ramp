# Week 02 — CUDA basics in Rust (thin wrapper over cargo; repo contract:
# one command for tests, one for benchmarks).
#
# Requires: CUDA >= 12.8 in WSL2, the pinned nightly (rust-toolchain.toml),
# and the Rust-CUDA git pins in Cargo.toml resolved (Day 1 task).

PY ?= python3

.PHONY: build test bench plot check clean

build:
	cargo build --release

# CPU-safe unit tests (gpu-bench median/JSON) — these also run in CI.
test:
	cargo test --release -p gpu-bench

# One command reproduces every published number (repo contract).
bench: build
	cargo run --release --bin vector_add
	cargo run --release --bin saxpy
	cargo run --release --bin reduce_naive
	cargo run --release --bin reduce_shared
	cargo run --release --bin reduce_warp
	cargo run --release --bin bandwidth_sweep
	$(PY) bench/plot_results.py results

plot:
	$(PY) bench/plot_results.py results

# GPU-less machines / CI: type-check + lint without the NVVM backend.
check:
	SKIP_PTX=1 cargo clippy --workspace --all-targets

clean:
	cargo clean
	rm -rf results
