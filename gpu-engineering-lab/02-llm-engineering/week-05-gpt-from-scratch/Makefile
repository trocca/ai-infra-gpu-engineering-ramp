# Week 05 — GPT from scratch. One-command everything (run under WSL2).

PY ?= python

.PHONY: help setup data test train sample bench profile clean

help:
	@echo "make setup    - install deps"
	@echo "make data     - tokenize TinyStories -> data/*.bin (one-time)"
	@echo "make test     - run the full test suite (CPU-ok)"
	@echo "make train    - train the d12 (~30M) model"
	@echo "make sample   - generate from the best checkpoint"
	@echo "make bench    - tok/s with vs without KV cache (JSON + plot)"
	@echo "make profile  - torch.profiler trace of a few training steps"

setup:
	$(PY) -m pip install torch numpy tiktoken datasets matplotlib pytest

data:
	$(PY) -m src.data --out data --num-tokens 100000000

test:
	$(PY) -m pytest tests/ -v

train:
	$(PY) -m src.train --config d12 --data data --out checkpoints/d12 \
		--batch-size 32 --seq-len 512 --grad-accum 32 --max-steps 5000

sample:
	$(PY) -m src.generate --ckpt checkpoints/d12/best.pt \
		--prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-p 0.95

bench:
	$(PY) bench/bench_generate.py --config d12 --runs 50 --out bench/results

# Day 5: add a --profile flag to src/train.py that wraps ~5 steps in
# torch.profiler and exports a chrome trace, then inspect in chrome://tracing
# or https://ui.perfetto.dev
profile:
	$(PY) -m src.train --config d12 --data data --out checkpoints/profile \
		--batch-size 32 --seq-len 512 --grad-accum 1 --max-steps 5 --profile || \
		echo "hint: implement the --profile flag in src/train.py (Day 5)"

clean:
	rm -rf checkpoints/profile bench/results/*.tmp
