# Week 07 — Triton kernels + quantization. GPU required (run under WSL2).

PY ?= python

.PHONY: help setup test test-softmax test-rmsnorm test-flash test-quant bench clean

help:
	@echo "make setup        - install deps (triton ships with torch cu128)"
	@echo "make test         - all correctness tests vs torch oracles"
	@echo "make test-flash   - just the FlashAttention forward tests"
	@echo "make bench        - full kernel sweep: JSON + latency/memory plots"

setup:
	$(PY) -m pip install torch numpy matplotlib pytest transformers datasets
	$(PY) -c "import triton; print('triton', triton.__version__)"

test:
	$(PY) -m pytest tests/ -v

test-softmax:
	$(PY) -m pytest tests/test_softmax.py -v

test-rmsnorm:
	$(PY) -m pytest tests/test_rmsnorm.py -v

test-flash:
	$(PY) -m pytest tests/test_flash.py -v

test-quant:
	$(PY) -m pytest tests/test_quant.py -v

bench:
	$(PY) bench/bench_kernels.py --out bench/results

clean:
	rm -rf bench/results/*.tmp __pycache__ src/__pycache__ tests/__pycache__
