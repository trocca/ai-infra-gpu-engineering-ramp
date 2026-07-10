# Cloud setup — multi-GPU weeks (9–11)

Weeks 9–10 need **one node with 2 GPUs** (parallelism internals). Week 11 needs a GPU
node you can run k3s on. Budget across all three weeks: **~$25–50** if you terminate
instances religiously.

## Provider options (pick by price that week)

| Provider | Instance | ~$/h (2026) | Notes |
|----------|----------|-------------|-------|
| Lambda / CoreWeave-style GPU clouds | 2× A10 or 2× A100 fractional | 0.60–1.50 | usually cheapest, plain Ubuntu, fastest to start |
| AWS | `g6.12xlarge` (4× L4) or `g5.12xlarge` (4× A10G) | ~2–4 (spot less) | only if you already have quota; request G-instance quota early |
| GCP | `g2-standard-24` (2× L4) | ~2 | L4 has no NVLink/P2P — good, that's realistic for NCCL transport study |
| Vast.ai / RunPod | 2× 3090/4090 | 0.40–0.80 | cheapest; fine for W9–10, avoid for anything sensitive |

2× L4 or 2× A10G is deliberately modest: the point of W9–10 is collectives, transports,
and scaling *measurement*, not absolute speed. A pair of GPUs **without** NVLink teaches
you more about NCCL transport selection than an NVLink pair does.

## Standard node bring-up (every rental, ~10 min)

```bash
nvidia-smi                                   # 2 GPUs visible, driver ≥ 570
python3 -m venv ~/v && source ~/v/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.device_count())"   # 2

# nccl-tests (used in week 09)
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make -j
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2

# P2P topology — know what you rented
nvidia-smi topo -m
```

## Cost discipline (non-negotiable)

- Set a billing alert **before** the first rental.
- Work from a git branch; push before every break. The instance is disposable, your
  code is not. Never keep state only on the node.
- `terminate`, not `stop`, when done for the day (stopped instances still bill disks).
- Keep a `costs.md` log: date, provider, instance, hours, $. It goes in the final
  capstone writeup — "did all of this for $X" is part of the portfolio story.

## Week 11 note (k3s + GPU Operator)

Rent a single-GPU node (1× L4 is enough, ~$0.70/h) with Ubuntu 22.04/24.04, install k3s,
then the GPU Operator via Helm — full steps are in the week-11 project README. Expect
~2 sessions × 4 h; snapshot nothing, the whole point is that the Helm chart + manifests
in git recreate the world from scratch.
