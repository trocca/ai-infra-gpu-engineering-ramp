# Week 10 — TP & PP internals. Local targets need no GPU.

.PHONY: test tp-parity pp-train oom exp-tp exp-pp bubble plot reproduce clean

# Day 1 gate: all green on CPU/gloo before renting anything.
test:
	pytest tests/ -v

# Day 2 (cloud): TP=2 logits/loss parity run.
tp-parity:
	torchrun --standalone --nproc_per_node=2 -m src.tp_model

# Day 3 (cloud): pipeline training + bubble sweep.
pp-train:
	torchrun --standalone --nproc_per_node=2 -m src.pipeline --microbatches 8 --steps 100
bubble:
	torchrun --standalone --nproc_per_node=2 -m src.pipeline --measure-bubble

# Day 4 (cloud): the punchline.
oom:
	python -m src.run_experiments --exp oom
exp-tp:
	torchrun --standalone --nproc_per_node=2 -m src.run_experiments --exp tp
exp-pp:
	torchrun --standalone --nproc_per_node=2 -m src.run_experiments --exp pp

# Day 5 (local, free).
plot:
	python -m src.run_experiments --plot

# One-command reproduce of the no-GPU-required parts.
reproduce: test plot

clean:
	rm -rf bench/results/*.png
