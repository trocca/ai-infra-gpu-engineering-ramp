# k8s-ai-stack-demo

**What the developer writes vs. what's automatic** — a runnable demo of the full Kubernetes-AI stack: distributed PyTorch training and multi-node LLM serving where the developer's entire surface is `train.py`, a `Dockerfile`, and ~25 lines of YAML.

```
Layer 15  train.py            ← you write this      (train/train.py)
Layer 10  Dockerfile          ← you write this      (train/Dockerfile)
Layer  9  TrainJob / LWS CRD  ← you write this      (train/trainjob.yaml, serve/lws-vllm.yaml)
─────────────────────────────────────────────────── the line ──
Layer  8  operator injects MASTER_ADDR/RANK + headless Svc     automatic
Layer  7  KAI scheduler: gangs, queues, fractional GPUs        automatic
Layer  5  DRA: ResourceClaims, ComputeDomains (GB200)          automatic
Layer  4  kubelet+containerd inject /dev/nvidia* via CDI       automatic
Layer  2  GPU Operator: driver, toolkit, DCGM, MIG             one Helm chart
```

Everything below the line is installed **once** (`scripts/setup.sh`) and never touched again.

## Quick start

```bash
# 0. A Kubernetes cluster with NVIDIA GPU nodes (2 nodes × 1 GPU is enough
#    for the training demo; 8 GPUs total makes the scheduling demo shine).

# 1. Install the platform (GPU Operator, KAI, Kubeflow Trainer, LWS) — once.
./scripts/setup.sh

# 2. Build & push the training image, point trainjob.yaml at it.
docker build -t <YOUR_REGISTRY>/k8s-ai-stack-demo/train:latest train/
docker push <YOUR_REGISTRY>/k8s-ai-stack-demo/train:latest

# 3. Run the narrated demo, scene by scene.
./scripts/demo.sh
```

## What's in here

| Path | Layer | What it shows |
|---|---|---|
| `train/train.py` | 15 | Plain DDP trainer; prints the injected env-var contract, logs NCCL topology via `NCCL_DEBUG=INFO`, rank-0 checkpointing |
| `train/Dockerfile` | 10 | `FROM nvcr.io/nvidia/pytorch:YY.MM-py3` — the monthly ABI-matched NGC stack |
| `train/trainjob.yaml` | 9 | Kubeflow **TrainJob v2**: `numNodes`, image, command — the entire training intent |
| `scheduling/deadlock-naive.yaml` | 7 | Reproduces the **partial-gang deadlock** with the default scheduler |
| `scheduling/deadlock-gang-kai.yaml` | 7 | Same jobs, **KAI gang-scheduled** — all-or-nothing, no deadlock |
| `scheduling/fractional-gpu.yaml` | 7 | Two pods sharing one physical GPU (no memory isolation — say so!) |
| `serve/vllm-single.yaml` | 13 | vLLM OpenAI-compatible endpoint on one GPU |
| `serve/lws-vllm.yaml` | 8–9 | **LeaderWorkerSet**: a model sharded across nodes, deployed as one unit |
| `dra/resourceclaim-gpu.yaml` | 5 | **DRA** (GA in K8s v1.34): attribute-based GPU claims vs the old integer count |
| `dra/computedomain-gb200.yaml` | 5 | Read-along: **ComputeDomains** — per-workload isolated IMEX/NVLink memory domains on GB200 NVL72 |
| `scripts/demo.sh` | — | The narrated 7-scene demo storyline |

## The one thing to take away

Distributed training needs every process to know three things: a coordinator address, its own rank, and the world size. On this stack the **operator** answers all three (headless Service DNS + injected env vars), the **scheduler** guarantees all workers start together (gang scheduling), and **NCCL** picks the fastest wire on its own (NVLink → InfiniBand/RoCE with GPUDirect RDMA). The developer never writes an IP address.

## Honest caveats (also demo talking points)

- KAI's fractional GPU sharing has **no memory isolation by default** — use MIG or HAMi for hard multi-tenancy.
- `deadlock-gang-kai.yaml` and the DRA manifests pin **fast-moving APIs** — re-verify apiVersions/labels against current upstream docs before a live run.
- The ComputeDomain manifest requires real GB200/MNNVL hardware; it's included to be *explained*, not executed, on commodity clusters.

## License

Apache-2.0. Built as a portfolio/demo artifact for developer-advocacy work on the NVIDIA Kubernetes-AI stack.
