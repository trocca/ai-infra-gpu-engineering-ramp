# Study Plan — Becoming a Great NVIDIA Dev Evangelist / Pre-Sales Engineer for the Kubernetes-AI Stack

**Owner:** Tiziano · **Window:** ~8 weeks · **Baseline:** solid across the board, sharpening to demo-ready
**Goal:** Be able to *pitch, whiteboard, demo, and defend* the full 0→15 stack to a skeptical enterprise developer — the exact skill NVIDIA's Developer Advocate reqs screen for ("world-class communication + a portfolio of developer content + genuine hands-on depth").

---

## How to use this plan

Three tracks run in parallel every week, because the job is the intersection of all three:

1. **Depth track** — learn one horizontal slice of the stack cold (the *why*, not just the *what*).
2. **Hands track** — do the matching lab in `02_hands_on_labs.md`. You retain almost nothing from reading alone; a working demo is worth ten blog posts of passive reading.
3. **Story track** — after each slice, write the 3-sentence "pitch" for that layer: what problem it solves, what the developer no longer has to do, and why an enterprise pays for it. These accumulate into your "teach us something" talk (`01_pitch_and_demo.md`).

**Daily cadence (≈1.5–2.5 hrs/day):** 45 min depth reading → 45–60 min hands-on → 15 min write the layer's pitch card + one flashcard-style Q&A. Fridays: rehearse the pitch out loud and record it. Weekends: catch-up or the bigger labs.

**The mental model to anchor everything:** the stack is a line. *Above the line* (layers 15, 10, 9) the developer writes intent — `train.py`, a `Dockerfile`, and a few dozen lines of YAML. *Below the line* (8→0) everything is mechanism, installed once by platform/admin and automatic thereafter. Your entire value proposition is that line. Learn every layer well enough to explain what moved below it and why that's the product.

---

## The 16 layers, grouped into 4 "systems" you can hold in your head

| Diagram layers | System | One-line mental model |
|---|---|---|
| 0–2 | **Iron & drivers** | Physical GPUs + NVLink/IB, the NVIDIA driver, GPU Operator |
| 3–6 | **The cluster fabric** | K8s control plane, pod networking/RDMA, kubelet+CDI, DRA, topology |
| 7–9 | **Orchestration** | KAI scheduler, workload operators, the CRDs the dev writes |
| 10–15 | **The workload** | Container image, CUDA libs, NCCL, PyTorch, launcher, your code |

You'll learn it **bottom-up for understanding** (weeks 1–4) then **top-down for the pitch** (weeks 5–8), because customers experience it top-down ("I just wrote train.py") but it's *built* bottom-up.

---

## Phase 1 — Iron, drivers & the cluster floor (Week 1)
**Layers 0–4.** *Goal: explain how a GPU physically becomes usable inside a container on a node.*

**Learn:**
- **Hardware (L0):** H100/H200 (Hopper) vs GB200/GB300 NVL72 (Blackwell). Know that GB200 NVL72 = 72 Blackwell GPUs + 36 Grace CPUs across 18 nodes, 9 NVLink switches, ~1.8 TB/s per-GPU NVLink, wired to behave like *one giant GPU* ("Multi-Node NVLink" / MNNVL). NVLink vs NVSwitch vs PCIe. InfiniBand (Quantum) vs RoCE Ethernet (Spectrum-X) for scale-out.
- **GPU driver (L2):** kernel modules + **Fabric Manager** (needed on NVSwitch systems). Driver branches (e.g. 580.x for CUDA 13, 575.x). `libcuda.so` = the user-mode driver, ships *with the driver*.
- **GPU Operator (L2):** the single Helm-installed operator that DaemonSets the whole node stack: containerized driver, **NVIDIA Container Toolkit**, device plugin, **NFD + GFD** (node labels), **DCGM + DCGM-Exporter** (telemetry), **MIG Manager**, validators.
- **Kubelet + containerd + CDI (L4):** how `/dev/nvidia*` device nodes get injected. **CDI (Container Device Interface)** is the modern, runtime-agnostic spec (`/var/run/cdi/nvidia.yaml`, `nvidia-ctk cdi list`) that replaced the old OCI prestart-hook / `NVIDIA_VISIBLE_DEVICES` approach.

**Do:** Lab 1 (GPU Operator on a cloud GPU node; `kubectl exec` into a pod and run `nvidia-smi`; inspect the CDI spec).

**Pitch card to write:** "Before the GPU Operator, standing up a GPU node meant hand-installing driver + toolkit + plugin and praying the versions matched. Now it's one Helm chart, and CDI makes GPUs a first-class container device. The dev never touches any of it."

---

## Phase 2 — DRA, topology & the scheduler (Weeks 2–3)
**Layers 5–7.** *This is the most differentiated, most current, most interview-likely material. Spend real time here.*

### Week 2 — DRA & topology (L5–L6)
**Learn:**
- **Dynamic Resource Allocation (DRA):** the `resource.k8s.io` API framework that replaced the rigid device-plugin *counting* model (`nvidia.com/gpu: 1`). Memorize the version story: **alpha v1.26 → redesigned around "structured parameters" (KEP-4381) → beta v1.32 → core GA in v1.34 (Aug 2025), `resource.k8s.io/v1`.** (Note the live docs may render a v1.35 [stable] label from gate-removal; say "core DRA went GA in v1.34" and you're safe.)
- The four objects, cold: **ResourceClaim** (a PVC-for-devices), **ResourceClaimTemplate** (stamps a claim per pod), **DeviceClass** (admin/driver-defined category), **ResourceSlice** (driver-published inventory the scheduler reads). Structured parameters mean the **stock kube-scheduler** does allocation — no vendor webhook.
- **nvidia-dra-driver-gpu** provides two drivers: `gpu.nvidia.com` (fine-grained GPU + MIG, replaces the device plugin) and `compute-domain.nvidia.com`.
- **ComputeDomains** — the headline. On GB200 NVL72, MNNVL lets GPUs in *different nodes* do direct memory load/stores over NVLink. **IMEX (Internode Memory Exchange)** is the driver service that enables + access-controls that; an **IMEX domain** = the set of nodes allowed to share memory. A **ComputeDomain** CRD dynamically creates/isolates *one IMEX domain per workload* so a shared rack fabric stays securely multi-tenant. Placement uses the `nvidia.com/gpu.clique` node label. This is the canonical example of DRA expressing something the device plugin *fundamentally could not*.
- **Topograph (L6):** NVIDIA's topology-discovery toolkit. Publishes hierarchical node labels — `network.topology.nvidia.com/{accelerator,leaf,spine,core}` — sourced from cloud APIs (AWS/GCP/OCI/Nebius) or the IB fabric (NetQ). Schedulers (KAI, Kueue TAS) consume them for rail-optimized "same-leaf-first" placement. Complements the *intra-node* K8s Topology Manager (NUMA/PCIe).

**Do:** Lab 5 (DRA ResourceClaim concepts; read a ComputeDomain manifest and explain every field).

### Week 3 — KAI Scheduler (L7)
**Learn:**
- **KAI = the open-sourced Run:ai scheduler.** NVIDIA acquired Run:ai (~$700M, announced Apr 2024, closed Dec 30 2024 after EU clearance) and **open-sourced KAI (Apache 2.0, github.com/NVIDIA/KAI-Scheduler) on Apr 1 2025 at KubeCon EU.** Runs as a **secondary scheduler alongside** kube-scheduler (pods opt in via `schedulerName`). Shares kube-batch lineage with Volcano; complementary to Kueue (queueing).
- Feature set cold: **gang scheduling** via PodGroups (all-or-nothing), **elastic gangs** (min/max replicas), **hierarchical queues** with quota/over-quota weights/limits, **DRF + time-based fair-share**, action order **Allocate → Consolidate → Reclaim → Preempt**, **decoupled priority vs preemptibility**, **fractional GPU sharing** (`gpu-fraction` / `gpu-memory` annotations + reservation pods; *no memory isolation by default* — pair with HAMi). Native KubeRay integration.
- **Gang scheduling = the partial-gang-deadlock story.** A distributed job needs all N workers at once (they block on NCCL collectives). Naive scheduler: Job A grabs 6/8 GPUs and waits, Job B grabs the other 2 and waits → deadlock, GPUs "allocated" but idle. Gang scheduling commits the whole PodGroup or none → guaranteed forward progress. **This is a top-3 thing to be able to draw on a whiteboard.**

**Do:** Lab 4 (submit two competing multi-pod jobs with and without gang scheduling; watch the deadlock, then watch KAI fix it).

**Pitch card:** "The scheduler is where GPU dollars are won or lost. KAI raises utilization with fractional sharing and fair-share reclaim, and prevents deadlock on distributed jobs with gang scheduling. NVIDIA open-sourced it because higher utilization = more workloads per cluster = validation for bigger clusters."

---

## Phase 3 — Workload APIs & the operators (Week 4)
**Layers 8–9.** *The CRDs the developer actually writes, and the operators that make rendezvous "just work."*

**Learn:**
- **The env-var contract.** Every distributed framework needs three things per process: a coordinator address (`MASTER_ADDR`), its own identity (`RANK`/`LOCAL_RANK`), and the total (`WORLD_SIZE`). The magic that makes `torchrun` "just work" on K8s: a **headless Service** (`clusterIP: None`) gives each pod stable DNS (`<pod>.<svc>.<ns>`), and the **operator injects** the env vars / sets `MASTER_ADDR` = rank-0 pod's DNS. The dev writes none of it.
- **Kubeflow Trainer / TrainJob v2 (training):** the v1→v2 leap = one framework-specific CRD per framework (`PyTorchJob`, `TFJob`, `MPIJob`…) collapsed into **one unified `TrainJob`**, with framework specifics in reusable **`TrainingRuntime` / `ClusterTrainingRuntime`** blueprints. Built on **JobSet**. GA'd **v2.0.0 on Jul 21 2025** (API kind still `trainer.kubeflow.org/v1alpha1` — a precise nuance). `numNodes`, `image`, `command` live in the TrainJob; `mlPolicy.torch.numProcPerNode` in the runtime. Trainer uses `torchrun` under the hood.
- **LeaderWorkerSet / LWS (inference):** a K8s SIG API that treats a **leader + N workers as one replicated "super-pod."** Solves **multi-host inference** — one model too big for a node, sharded across nodes (e.g. multi-node vLLM with a Ray head on the leader). Dual `leaderTemplate`/`workerTemplate`; creates a StatefulSet + headless Service per group; injects `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX`; all-or-nothing rollout & restart per group.
- **NVIDIA Grove (disaggregated inference):** open-sourced Nov 10 2025 (`ai-dynamo/grove`, alpha). Where LWS is one homogeneous group, Grove models a **whole multi-component system** — prefill leaders/workers + decode leaders/workers + router + frontend — as one CRD hierarchy: **PodClique → PodCliqueScalingGroup → PodCliqueSet**, emitting a **PodGang** that **KAI** gang-schedules. Multi-level autoscaling; topology-aware co-placement on one NVLink domain (GB200). This is the L9→L7 handoff.

**Do:** Lab 3 (multi-node TrainJob v2) and read a real LWS + Grove manifest field-by-field.

**Pitch card:** "The dev writes ~30 lines: numNodes, image, command, GPUs. The operator injects MASTER_ADDR/RANK and stands up the headless Service so torchrun rendezvous just works. For inference, LWS groups a sharded model; Grove models a whole disaggregated topology."

---

## Phase 4 — The workload: containers, CUDA, NCCL, PyTorch (Weeks 5–6)
**Layers 10–15.** *Now go top-down — this is what the dev experiences.*

### Week 5 — Container, CUDA libs, NCCL (L10–L12)
**Learn:**
- **NGC / `nvcr.io` (L10):** the container registry; tag scheme **`YY.MM-pyN`** (calendar version, e.g. `nvcr.io/nvidia/pytorch:26.04-py3` = April 2026 build, *not* the PyTorch version). Monthly re-validated stack (framework + CUDA + cuDNN + NCCL + TransformerEngine, all ABI-matched). Why it beats `pip install torch`: pre-tested compatibility + NVIDIA-patched PyTorch (`2.x.0a0` builds) + extras (TransformerEngine, Apex, DALI). Base images `nvidia/cuda`: `base`/`runtime`/`devel` variants.
- **libcudart vs libcuda (L11):** **runtime API** (`cudaMalloc`, ships in the toolkit/container, linked into the framework) vs **driver API** (`cuLaunchKernel`, ships with the driver). **Compatibility floors:** CUDA 13.x needs driver ≥580, 12.x ≥525, 11.x ≥450. `cuda-compat` forward-compat package lets a newer toolkit run on an older kernel driver — how a CUDA 13 container runs on a datacenter node pinned to a 12-era driver.
- **CUDA libs (L12):** **cuBLAS/cuBLASLt** (GEMM = the core transformer compute; cuBLASLt does FP8 + epilogue fusion), **cuDNN** (attention/Flash-Attention, normalization), **TransformerEngine** (FP8 on Hopper, FP4/MXFP8 on Blackwell; auto-manages scaling factors).
- **NCCL (L12) — learn this deeply, it's a favorite interview topic:** collective ops (AllReduce, AllGather, ReduceScatter, AllToAll). Algorithms (Ring = bandwidth-optimal; Tree = latency-optimal at scale; NVLS = NVLink SHARP). Transports auto-selected by distance (NVLink/NVSwitch intra-node → IB/RoCE with **GPUDirect RDMA** inter-node, NIC DMA'ing straight into GPU HBM). **SHARP** = in-network reduction offloaded into the switch ASIC (NCCL 2.27, Jul 2025, ≤6 SMs vs ≥16). Topology auto-detection; `nvidia_peermem` module enables GPUDirect RDMA.

**Do:** Lab 2 (pull an NGC PyTorch container, run a GPU workload, set `NCCL_DEBUG=INFO` and read the topology it detects).

### Week 6 — PyTorch, launchers, distributed training (L13–L15)
**Learn:**
- **torchrun (L14):** superset of the deprecated `torch.distributed.launch`; adds elasticity (`--nnodes=1:4`), fault tolerance (`--max-restarts`), and **c10d rendezvous**. Injects `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR/PORT`. **`init_process_group(env://)` reads** MASTER_ADDR/PORT + RANK + WORLD_SIZE; **LOCAL_RANK picks the GPU** (`set_device(local_rank)`). c10d rendezvous = distributed barrier + peer discovery with a TCPStore, guaranteeing barrier/exclusivity/consistency/fault-tolerance. `mpirun` is the HPC/MPI alternative (sets `OMPI_COMM_WORLD_*`).
- **PyTorch parallelism (L13):** **DDP** (full replica per GPU, all-reduce grads — model fits on one GPU). **FSDP** (ZeRO-3: shard params+grads+optimizer, all-gather before fwd/bwd, reduce-scatter grads — model too big; FSDP1 flat-param, now deprecated). **FSDP2** (`fully_shard`, per-parameter **DTensor** sharding, composes with TP — current standard, torch 2.4+). **Megatron-Core** (NVIDIA's parallelism library: **TP × PP × CP × EP × DP**, up to 5D; SP "always on with TP"; MoE with DeepEP/HybridEP dispatchers). **NeMo 2.0** wraps Megatron-Core via NeMo Lightning (Python config, PEFT/LoRA, recipes, NeMo-Run).

**Do:** Lab 6 warm-up (run a 2-GPU DDP job, then an FSDP job; read the rank/world-size logs).

**Pitch card:** "The dev writes `train.py` with `init_process_group()` and picks DDP or FSDP. Everything that makes it scale — NCCL choosing NVLink vs InfiniBand, topology, the launcher setting RANK — is below the line."

---

## Phase 5 — Inference at datacenter scale (Week 7)
*The 2025-2026 story that makes you sound current: disaggregated serving.*

**Learn:**
- **vLLM:** PagedAttention (KV cache in 16-token blocks, kills fragmentation like OS paging), continuous/in-flight batching, chunked prefill, automatic prefix caching, TP+PP. The open community standard and a backend inside NIM/Dynamo.
- **NVIDIA Dynamo — the headline inference story.** Announced GTC Mar 2025, **1.0 GA Mar 2026.** "Operating system for AI factories." **Disaggregated prefill/decode:** prefill is *compute-bound* (wants low TP), decode is *memory-bandwidth-bound* (wants high TP) → separate GPU pools scaled independently. Four components: **Smart Router** (KV-cache-aware routing via Radix trees → route to the GPU already holding the KV, cut TTFT), **Planner** (dynamically shifts GPUs between prefill/decode by SLO), **KVBM** (tier KV across HBM→DRAM→SSD→object, petabyte-scale), **NIXL** (fast KV transfer, 270 GB/s across 8×H100). Engine-agnostic (vLLM, TRT-LLM, SGLang). Headline numbers: 30× on DeepSeek-R1 on GB200 NVL72; 2×+ on Llama on Hopper.
- **NIM** (packaged, OpenAI-compatible model microservices in NVIDIA AI Enterprise; NIM 2.0 = one-container/vLLM-backend), **Triton** (mature multi-framework server; Dynamo is its LLM/distributed successor, not a drop-in replacement), **TensorRT-LLM** (compilation/optimization library, FP4/NVFP4 on Blackwell; often the engine NIM wraps).

**Do:** Lab 6 (single-node vLLM), Lab 7 (multi-node vLLM on LWS, then Dynamo disaggregated if you have the GPUs).

---

## Phase 6 — Pitch, product landscape & mock interviews (Week 8)
*Convert knowledge into the three things they screen for: communication, portfolio, depth.*

**Learn / do:**
- **Product landscape** (be able to speak to at a high level): Blackwell / GB200-GB300 NVL72, Vera Rubin (H2 2026, announced GTC 2026) and the annual cadence, Spectrum-X / Quantum-X networking, DGX/HGX, DGX Cloud, and the software motion — **NVIDIA AI Enterprise = NIM + NeMo + Run:ai + CUDA-X** (the recurring per-GPU-subscription business on top of hardware). The **"AI factory"** frame: a datacenter re-conceived as a factory that manufactures tokens, measured in tokens/sec, revenue/rack, joules/token — and NVIDIA sells the *whole factory*.
- **The abstraction pitch** as your "teach us something" talk — finalize and rehearse `01_pitch_and_demo.md`. This single asset proves all three screened traits at once.
- **Objection handling** — drill the 7 common pre-sales objections (utilization, GPU sharing/multi-tenancy, cost/ROI, job failures/checkpointing, observability, lock-in/open-source, networking) in `03_mock_interview_qa.md`.
- **Behavioral / STAR** — write 6–8 stories mapped to: translating deep tech for non-experts, turning an objection into a win, a demo/OSS that drove adoption (bring numbers), handling ambiguity, intellectual honesty, cross-functional influence, first-principles/"speed of light", learning velocity.
- **Culture** — internalize the Jensen frame (the mission is the boss; flat org / default-to-broadcast; "speed of light" first-principles benchmarking; intellectual honesty). Cite conversationally, not as if quoting an HR doc.

**Do:** Two full mock interviews (one technical-deep-dive, one behavioral+presentation). Record the presentation. Have your GitHub/demo repo public and linkable — the req effectively requires a portfolio.

---

## Milestones (how you know you're ready)

- **End of Wk 1:** You can trace a GPU from silicon to `nvidia-smi` inside a pod, naming every automatic step.
- **End of Wk 3:** You can whiteboard the partial-gang deadlock and how KAI + DRA + Topograph solve placement — from memory.
- **End of Wk 4:** You can write a TrainJob v2 and an LWS manifest by hand and explain what the operator injects.
- **End of Wk 6:** You've run real multi-GPU DDP + FSDP jobs and read the NCCL topology logs.
- **End of Wk 7:** You can explain disaggregated serving and Dynamo's four components without notes.
- **End of Wk 8:** You can deliver a recorded 20–30 min "abstraction" talk + live demo, and field all 7 objections cold.

---

## Curated resources (highest-signal, per system)

**Iron & drivers:** NVIDIA GPU Operator docs (getting-started, gpu-sharing, rdma); Container Toolkit CDI docs; the GB200 NVL72 rack-scale blog.
**Cluster fabric / DRA:** Kubernetes DRA concepts + the v1.34 GA blog (kubernetes.io/blog/2025/09/01); KEP-4381; the "Enabling Multi-Node NVLink on Kubernetes for GB200" blog; NVIDIA DRA driver docs; Topograph GitHub.
**Scheduler:** the "NVIDIA open-sources Run:ai scheduler" blog; github.com/NVIDIA/KAI-Scheduler (README + docs/gpu-sharing); the rack-scale topology-aware scheduling blog.
**Workload APIs:** blog.kubeflow.org/trainer/intro; Kubeflow Trainer runtime + PyTorch guides; lws.sigs.k8s.io; the Grove blog + ai-dynamo/grove.
**Frameworks:** PyTorch DDP notes + FSDP2 tutorial; NCCL user guide (overview + env vars) + the NCCL 2.27 blog; Megatron-Core parallelism guide; NGC PyTorch release notes.
**Inference:** the vLLM "anatomy of vLLM" post; the Dynamo intro + "Dynamo 1.0 production-ready" blogs; NIM/AI Enterprise product pages.
**Role & culture:** the live NVIDIA Developer Advocate reqs (jobs.nvidia.com); GTC 2025 + 2026 keynote recaps (blogs.nvidia.com); Jensen management-principle write-ups.

*(Full URLs are in the master reference saved to your project.)*
