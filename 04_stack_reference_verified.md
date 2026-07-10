# Kubernetes-AI Stack — Verified Reference (mid-2026)

Condensed, source-cited facts for the NVIDIA Dev Evangelist interview, mapped to the 0–15 stack diagram. Verified via NVIDIA docs, Kubernetes KEPs/blogs, and framework sources. **Version numbers move fast — reverify before quoting live.**

## Layer 7 — KAI Scheduler
- Open-sourced Run:ai scheduler. Apache 2.0, `github.com/NVIDIA/KAI-Scheduler`. Launched KubeCon EU, **Apr 1 2025**. Run:ai acquired ~$700M, closed **Dec 30 2024** (after EU antitrust clearance).
- Runs as a **secondary scheduler alongside** kube-scheduler (opt-in via `schedulerName`). Shares kube-batch lineage with Volcano; complementary to Kueue (queueing).
- Features: gang scheduling (PodGroups, all-or-nothing) + elastic gangs (min/max); hierarchical queues w/ quota/over-quota; **DRF + time-based fair-share**; actions **Allocate→Consolidate→Reclaim→Preempt**; decoupled priority vs preemptibility; **fractional GPU** (`gpu-fraction`/`gpu-memory` + reservation pods, **no memory isolation by default → pair w/ HAMi**). Native KubeRay integration.
- Gang scheduling solves the **partial-gang deadlock** (Job A grabs 6/8, Job B grabs 2, both block on NCCL → deadlock; atomic PodGroup fixes it).
- Sources: developer.nvidia.com/blog/nvidia-open-sources-runai-scheduler-to-foster-community-collaboration/ · github.com/NVIDIA/KAI-Scheduler

## Layer 6 — Topograph
- Topology-discovery toolkit. Apache 2.0. Publishes node labels `network.topology.nvidia.com/{accelerator,leaf,spine,core}` from cloud APIs (AWS/GCP/OCI/Nebius) or IB fabric (NetQ). Consumed by KAI / Kueue TAS for rail-optimized same-leaf-first placement. Complements intra-node K8s Topology Manager (NUMA).
- Sources: github.com/NVIDIA/topograph · developer.nvidia.com/blog/running-ai-workloads-on-rack-scale-supercomputers-...

## Layer 5 — DRA + nvidia-dra-driver-gpu + ComputeDomains
- **DRA** (`resource.k8s.io`) replaced the device-plugin integer-count model. alpha v1.26 → structured-parameters redesign (KEP-4381) → beta **v1.32** → **core GA v1.34** (`resource.k8s.io/v1`, Aug 2025). (Live docs may render v1.35 [stable] from gate-removal; say "GA v1.34".)
- Four objects: **ResourceClaim** (PVC-for-devices), **ResourceClaimTemplate** (per-pod), **DeviceClass** (admin/driver category), **ResourceSlice** (driver-published inventory). Stock kube-scheduler allocates — no vendor webhook.
- **nvidia-dra-driver-gpu**: `gpu.nvidia.com` (fine-grained GPU + MIG, replaces device plugin) and `compute-domain.nvidia.com`. Requires disabling legacy device plugin. K8s v1.34.2+ for GPU alloc; v1.32+ for ComputeDomains; driver 580+; CDI on; GPU Operator v25.10+.
- **ComputeDomains**: GB200 NVL72 = 72 GPUs / 18 nodes / 9 NVLink switches, ~1.8 TB/s per-GPU, >130 TB/s aggregate (Multi-Node NVLink). **IMEX** (Internode Memory Exchange) = driver service enabling+access-controlling cross-node GPU memory sharing. A ComputeDomain CRD dynamically provisions **one isolated IMEX domain per workload**; co-places pods on same NVLink clique via `nvidia.com/gpu.clique`. Current constraint: 1 ComputeDomain/node, 1 pod/node/CD.
- Sources: kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/ · developer.nvidia.com/blog/enabling-multi-node-nvlink-on-kubernetes-for-gb200-and-beyond/ · docs.nvidia.com/.../dra-cds.html

## Layers 8–9 — Workload APIs
- **Kubeflow Trainer / TrainJob v2**: GA **v2.0.0 Jul 21 2025**; API kind `trainer.kubeflow.org/v1alpha1`. One unified `TrainJob` (was per-framework PyTorchJob/TFJob/MPIJob) + reusable `TrainingRuntime`/`ClusterTrainingRuntime`. Built on **JobSet**. `numNodes`/`image`/`command` in TrainJob; `mlPolicy.torch.numProcPerNode` in runtime. Uses torchrun.
- **LWS (LeaderWorkerSet)**: kubernetes-sigs/lws (~v0.9.0). Leader+N workers as one replicated unit → multi-host inference (multi-node vLLM). Dual leaderTemplate/workerTemplate; StatefulSet + headless Service per group; injects `LWS_LEADER_ADDRESS`/`LWS_GROUP_SIZE`/`LWS_WORKER_INDEX`; all-or-nothing rollout.
- **Grove**: ai-dynamo/grove, Apache 2.0, open-sourced **Nov 10 2025** (alpha). Multi-component disaggregated inference: **PodClique → PodCliqueScalingGroup → PodCliqueSet**, emits **PodGang** → KAI schedules. Multi-level autoscaling; topology-aware co-placement on one NVLink domain.
- **Env-var mechanism**: headless Service → stable pod DNS `<pod>.<svc>.<ns>`; operator sets MASTER_ADDR=rank-0 DNS + per-pod RANK. Makes torchrun rendezvous work with no dev config.
- Sources: blog.kubeflow.org/trainer/intro/ · lws.sigs.k8s.io · developer.nvidia.com/blog/streamline-complex-ai-inference-on-kubernetes-with-nvidia-grove/

## Layers 13–15 — Launchers & parallelism
- **torchrun** (supersedes deprecated torch.distributed.launch): elasticity (`--nnodes=1:4`), fault tolerance (`--max-restarts`), **c10d rendezvous** (TCPStore on rank 0; barrier/exclusivity/consistency/fault-tolerance). Injects RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR. `init_process_group(env://)` reads MASTER_ADDR/PORT+RANK+WORLD_SIZE; LOCAL_RANK picks GPU. mpirun = HPC alt (OMPI_COMM_WORLD_*).
- **Parallelism**: DDP (fits 1 GPU, all-reduce grads) → FSDP (ZeRO-3 shard params+grads+optim, all-gather fwd/bwd + reduce-scatter grads; FSDP1 flat-param deprecated) → **FSDP2** (`fully_shard`, per-param DTensor, composes w/ TP, torch 2.4+) → **Megatron-Core** (TP×PP×CP×EP×DP = 5D; SP always-on w/ TP; MoE w/ DeepEP/HybridEP) → **NeMo 2.0** (wraps Megatron-Core via NeMo Lightning, Python config, PEFT, NeMo-Run).
- Sources: pytorch.org/docs FSDP2 tutorial + DDP notes · docs.nvidia.com/megatron-core parallelism guide

## Layers 11–12 — CUDA libs & NCCL
- **libcudart** (Runtime API, in toolkit/container) vs **libcuda** (Driver API, ships w/ driver). Floors: CUDA 13.x→driver≥580, 12.x≥525, 11.x≥450. `cuda-compat` = forward-compat (new toolkit on old kernel driver).
- **cuBLAS/cuBLASLt** (GEMM, FP8 + epilogue fusion), **cuDNN** (attention/Flash-Attn), **TransformerEngine** (FP8 Hopper / FP4-MXFP8 Blackwell, auto scaling factors).
- **NCCL**: collectives (AllReduce/AllGather/ReduceScatter/AllToAll). Algos Ring (BW-opt) / Tree (latency-opt) / NVLS (NVLink SHARP). Transports auto by distance: NVLink/NVSwitch intra-node → IB/RoCE + **GPUDirect RDMA** inter-node (`nvidia_peermem`). **SHARP** = in-network reduction in switch ASIC (NCCL **2.27 Jul 2025**, ≤6 SMs vs ≥16). Topology auto-detect.
- Sources: docs.nvidia.com/deeplearning/nccl · developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/

## Layer 10 — NGC containers
- `nvcr.io`. Tag `YY.MM-pyN` = calendar build (`26.04-py3` = Apr 2026, NOT PyTorch version). Monthly ABI-matched stack (PyTorch `a0` build + CUDA + cuDNN + NCCL + TransformerEngine + Apex + DALI). Base `nvidia/cuda`: base/runtime/devel variants. CDI injects /dev/nvidia* (`/var/run/cdi/nvidia.yaml`).
- Sources: docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes · docs.nvidia.com/.../container-toolkit/cdi-support.html

## Inference/serving
- **vLLM**: PagedAttention (KV in 16-token blocks, no fragmentation), continuous batching, chunked prefill, prefix caching, TP+PP. Backend inside NIM/Dynamo/Triton.
- **NVIDIA Dynamo**: GTC Mar 2025, **1.0 GA Mar 2026**. Disaggregated prefill (compute-bound, low TP) / decode (BW-bound, high TP). 4 parts: **Smart Router** (KV-aware, Radix trees), **Planner** (GPU rebalance by SLO), **KVBM** (HBM→DRAM→SSD→object), **NIXL** (270 GB/s / 8×H100). Engine-agnostic (vLLM/TRT-LLM/SGLang). Successor to Triton. 30× DeepSeek-R1 / GB200; 2×+ Llama / Hopper.
- **NIM** (AI Enterprise; NIM 2.0 = one-container/vLLM backend, OpenAI-compatible), **Triton** (multi-framework, single-node), **TensorRT-LLM** (compile/optimize lib, NVFP4 Blackwell; often the engine NIM wraps).
- GPU sharing: **MIG** (HW partition, hard isolation), **time-slicing** (SW, no isolation), **MPS** (concurrent contexts, weak isolation).
- Sources: developer.nvidia.com/blog/introducing-nvidia-dynamo-... · .../nvidia-dynamo-1-production-ready/ · vllm.ai/blog/2025-09-05-anatomy-of-vllm

## Role, culture, product landscape
- **Role**: NVIDIA Developer Advocate reqs (jobs.nvidia.com) require "world-class communication + presentation skills" + a **content portfolio** + hands-on depth (PyTorch/JAX/DeepSpeed/NeMo, distributed training, scaling laws). ~20% travel. Comp L3 $152–241.5k / L4 $184–287.5k.
- **Interview (third-party, directional)**: 4–7 rounds — recruiter → technical/perf → architecture/system-design (multiple) → behavioral. Prepare a 20–30 min talk + demo regardless.
- **Culture (Jensen, widely attributed, not an HR doc)**: "the mission is the boss"; flat org / ~60 directs; no 1:1s / default-to-broadcast; "speed of light" first-principles benchmarking; intellectual honesty.
- **Products**: Blackwell / GB200-GB300 NVL72; Vera Rubin (GTC 2026, H2 2026) annual cadence; Feynman next. Spectrum-X / Quantum-X networking. DGX/HGX, DGX Cloud, DGX Spark/Station. **NVIDIA AI Enterprise = NIM + NeMo + Run:ai + CUDA-X** (per-GPU subscription). **"AI factory"** = datacenter as token factory (tokens/sec, revenue/rack, joules/token); NVIDIA sells the whole factory.
- Sources: jobs.nvidia.com · blogs.nvidia.com/blog/gtc-2026-news/ · nvidia.com/.../ai-enterprise/

## The pitch (one line)
Developer writes `train.py` + `Dockerfile` + ~30 lines of YAML (L15/10/9); everything below (L8→0) is automatic, installed once. That line between intent and mechanism is the product. Every year NVIDIA moves it up (this year: multi-node NVLink domains as a K8s primitive via DRA ComputeDomains).
