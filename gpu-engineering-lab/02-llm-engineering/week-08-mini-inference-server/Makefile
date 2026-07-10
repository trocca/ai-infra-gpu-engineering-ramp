# Week 08 — ferrum-serve (month capstone, Rust). Run under WSL2.

PY ?= python
CARGO_DIR = ferrum-serve
PORT ?= 8000
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct    # ungated; Llama-3.2-1B-Instruct is gated

.PHONY: help build test lint run smoke loadtest loadtest-open \
        serve-vllm loadtest-vllm bench-binary-size clean

help:
	@echo "make build          - cargo build --release --features cuda"
	@echo "make test           - cargo test (CPU-only; starts RED, you turn it green)"
	@echo "make lint           - clippy -D warnings + fmt --check"
	@echo "make run            - run ferrum-serve on :$(PORT)"
	@echo "make smoke          - one streaming request + /stats via curl"
	@echo "make loadtest       - closed-loop sweep against ferrum-serve"
	@echo "make loadtest-open  - open-loop (Poisson) sweep against ferrum-serve"
	@echo "make serve-vllm     - run real vLLM on :8001 (comparison baseline)"
	@echo "make loadtest-vllm  - same client, same sweep, against vLLM"
	@echo "make bench-binary-size - stripped-binary size metric (compare_vllm.md §7)"

build:
	cd $(CARGO_DIR) && cargo build --release --features cuda

# Pure-logic tests (blocks + scheduler) — no GPU, no model download.
test:
	cd $(CARGO_DIR) && cargo test

lint:
	cd $(CARGO_DIR) && cargo clippy --all-targets -- -D warnings && cargo fmt --check

run: build
	cd $(CARGO_DIR) && cargo run --release --features cuda -- \
		--model-id $(MODEL) --port $(PORT)

smoke:
	curl -N -X POST localhost:$(PORT)/v1/completions \
		-H 'Content-Type: application/json' \
		-d '{"prompt": "Once upon a time", "max_tokens": 50, "stream": true}'
	@echo ""
	curl -s localhost:$(PORT)/stats
	@echo ""

loadtest:
	$(PY) bench/loadtest.py --url http://localhost:$(PORT) --mode closed \
		--concurrency 1 2 4 8 16 --requests-per-point 50 \
		--out bench/results/ferrum_closed.json

loadtest-open:
	$(PY) bench/loadtest.py --url http://localhost:$(PORT) --mode open \
		--rates 0.5 1 2 4 --duration 60 \
		--out bench/results/ferrum_open.json

serve-vllm:
	vllm serve $(MODEL) --port 8001 --dtype bfloat16 --max-model-len 2048

loadtest-vllm:
	$(PY) bench/loadtest.py --url http://localhost:8001 --model $(MODEL) \
		--mode closed --concurrency 1 2 4 8 16 --requests-per-point 50 \
		--out bench/results/vllm_closed.json

# compare_vllm.md §7 — strip a COPY (release profile keeps debug symbols for Nsight)
bench-binary-size: build
	cp $(CARGO_DIR)/target/release/ferrum-serve /tmp/ferrum-stripped
	strip /tmp/ferrum-stripped
	ls -lh $(CARGO_DIR)/target/release/ferrum-serve /tmp/ferrum-stripped

clean:
	cd $(CARGO_DIR) && cargo clean
