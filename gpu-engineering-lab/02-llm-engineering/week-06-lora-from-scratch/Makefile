# Week 06 — LoRA from scratch. Run under WSL2.

PY ?= python
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct   # ungated default; Llama-3.2-1B-Instruct is gated

.PHONY: help setup test train-lora train-qlora eval-base eval-lora merge clean

help:
	@echo "make setup       - install deps (peft is the test oracle)"
	@echo "make test        - unit + PEFT-parity tests (CPU, no downloads)"
	@echo "make train-lora  - Day 2: bf16 LoRA fine-tune"
	@echo "make train-qlora - Day 3: 4-bit NF4 base (QLoRA)"
	@echo "make eval-base   - Day 4: lm-eval + generations, base model"
	@echo "make eval-lora   - Day 4: lm-eval + generations, adapter model"
	@echo "make merge       - Day 4: fold adapters into dense weights + verify"

setup:
	$(PY) -m pip install torch transformers datasets peft bitsandbytes \
		lm-eval accelerate matplotlib pytest

test:
	$(PY) -m pytest tests/ -v

train-lora:
	$(PY) -m src.train_lora --model $(MODEL) --num-examples 5000 \
		--r 16 --alpha 32 --targets q_proj k_proj v_proj o_proj \
		--out runs/lora-bf16

train-qlora:
	$(PY) -m src.train_lora --model $(MODEL) --num-examples 5000 \
		--r 16 --alpha 32 --targets q_proj k_proj v_proj o_proj \
		--load-4bit --out runs/qlora-nf4

eval-base:
	$(PY) -m src.eval --model $(MODEL) --out runs/eval-base

eval-lora:
	$(PY) -m src.eval --model $(MODEL) --adapter runs/lora-bf16/adapter.pt \
		--out runs/eval-lora

merge:
	$(PY) -m src.merge --model $(MODEL) --adapter runs/lora-bf16/adapter.pt \
		--out runs/lora-bf16/merged --verify

clean:
	rm -rf runs/eval-base runs/eval-lora
