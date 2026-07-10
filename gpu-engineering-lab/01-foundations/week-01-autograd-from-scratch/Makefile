# Week 01 — autograd from scratch + Rust ramp
# One command for tests, one for the training run (repo contract).

PY ?= python

.PHONY: test train rust-hello clean

test:
	$(PY) -m pytest tests/ -v

train:
	$(PY) -m src.train_mnist

rust-hello:
	cd rust-hello-gpu && cargo run --release

clean:
	rm -rf results/ .pytest_cache src/__pycache__ tests/__pycache__ \
	       rust-hello-gpu/target
