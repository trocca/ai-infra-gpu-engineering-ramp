# Week 09 — distributed training. Local targets need no GPU.

.PHONY: test train-local bw-local bw-cloud compare nccl-debug plot reproduce clean

# Local, CPU/gloo — run before every cloud session.
test:
	pytest tests/ -v

# Day 1: manual data parallel on 2 CPU processes.
train-local:
	bash src/launch.sh local-cpu --steps 50 --out bench/results/train_manual_gloo.json

# Bandwidth sweep, local CPU (sanity) and cloud 2-GPU (real numbers).
bw-local:
	torchrun --standalone --nproc_per_node=2 bench/scaling.py --mode bandwidth --backend gloo
bw-cloud:
	torchrun --standalone --nproc_per_node=2 bench/scaling.py --mode bandwidth --backend nccl

# Day 3: the three-way comparison + single-GPU baseline (cloud).
compare:
	bash src/launch.sh compare --steps 200

# Day 4 (cloud).
nccl-debug:
	bash src/launch.sh nccl-debug
nccl-tests:
	bash src/launch.sh nccl-tests

# Day 5 (local, free).
plot:
	python bench/scaling.py --mode plot

# One-command reproduce of everything that runs WITHOUT a rented GPU.
reproduce: test train-local bw-local plot

clean:
	rm -rf bench/results/*.png
