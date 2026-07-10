# Week 04 — rusty_kernels (PyO3 + maturin)
# One command for tests, one for benchmarks (repo contract).

PY ?= python

.PHONY: install test bench check clean

# Builds the kernel crate to PTX, compiles the PyO3 extension, installs
# rusty_kernels into the active venv. Requires: pinned nightly + maturin.
install:
	maturin develop --release

test:
	$(PY) -m pytest tests/ -v

bench:
	$(PY) bench/bench_ops.py

# GPU-less machines / CI: lint the Rust without the NVVM backend.
check:
	SKIP_PTX=1 cargo clippy --workspace --all-targets

clean:
	cargo clean
	rm -rf results .pytest_cache rusty_kernels/__pycache__ tests/__pycache__ \
	       target dist
