# Mock Interview Q&A — NVIDIA Dev Evangelist / Pre-Sales Engineer

**How NVIDIA interviews for this role (directional — from third-party aggregators, not NVIDIA-published):** typically 4–7 rounds. Recruiter screen → technical/problem-solving (coding, memory/perf flavor) → architecture/system-design (often multiple) → behavioral/culture. The reqs *require* a content portfolio and "world-class presentation skills," so **prepare a 20–30 min talk + demo** (`01_pitch_and_demo.md`) even if a formal presentation round isn't confirmed.

Practice these **out loud**. For technical answers, the pattern that wins: **crisp definition → why it exists / what it replaced → one concrete number or example → business translation.** Model answers below are yours to compress; don't recite.

---

## Section 1 — Technical deep-dive (per layer)

**Q1. Walk me through what happens from `kubectl apply` of a training job to GPUs actually computing.**
The operator (Kubeflow Trainer) renders your TrainJob + runtime into a JobSet, creating N pods and a headless Service. Each pod gets stable DNS; the operator sets `MASTER_ADDR` to rank-0's DNS name and assigns each pod a `RANK`. The scheduler — KAI — gang-schedules the whole PodGroup all-or-nothing onto GPUs, using Topograph's topology labels to co-place workers on the same NVLink leaf. Kubelet + containerd pull the NGC image and, via CDI, inject `/dev/nvidia*` and the driver libraries. Inside each pod `torchrun` reads the injected env, does a c10d rendezvous through a TCPStore on rank 0, and spawns one process per GPU. Your `train.py` calls `init_process_group("nccl")`, which reads `MASTER_ADDR/PORT/RANK/WORLD_SIZE`; NCCL auto-detects topology and runs all-reduce over NVLink intra-node and GPUDirect-RDMA/InfiniBand inter-node. *The developer wrote three of those layers; the rest was automatic.*

**Q2. Why Dynamic Resource Allocation? What was wrong with the device plugin?**
The device plugin advertises GPUs as an opaque integer extended resource — `nvidia.com/gpu: 1`. You can't express "an H100 with ≥40GB and MIG profile 3g.20gb," can't cleanly model sharing semantics, and can't describe cross-node topology; every new capability needs a new resource name. DRA (`resource.k8s.io`) replaces counting with attribute-based **ResourceClaims** the *stock kube-scheduler* allocates from driver-published **ResourceSlices**, using structured parameters (KEP-4381) — so no vendor scheduler webhook. Version story: alpha v1.26 → structured-parameters redesign → beta v1.32 → **core GA in v1.34** (`resource.k8s.io/v1`). The four objects: ResourceClaim, ResourceClaimTemplate, DeviceClass, ResourceSlice.

**Q3. What are ComputeDomains and why do they matter for GB200?**
A GB200 NVL72 is 72 Blackwell GPUs across 18 nodes wired by NVLink switches into one Multi-Node NVLink fabric — the rack acts like one giant GPU, GPUs in different nodes doing direct memory load/stores. But you can't let every tenant touch every GPU's memory. **IMEX (Internode Memory Exchange)** is the driver service that enables and access-controls that cross-node sharing; an IMEX domain is the set of nodes allowed to share. A **ComputeDomain** is a Kubernetes CRD (via nvidia-dra-driver-gpu) that dynamically provisions *one IMEX domain per workload* and tears it down at completion, co-placing pods on the same NVLink clique via the `nvidia.com/gpu.clique` label. It's the canonical example of DRA expressing something the device plugin fundamentally could not.

**Q4. Explain gang scheduling and the problem it solves.** *(Draw it.)*
A distributed training job needs all N workers simultaneously because they block on NCCL collectives. A naive scheduler can hand Job A 6 of 8 needed GPUs and Job B the other 2 — both wait forever, neither releases: deadlock, GPUs allocated but idle. Gang scheduling makes the PodGroup atomic — schedule all N or none — so a job that can't fully fit stays Pending while one that fits runs to completion and frees resources. Guaranteed forward progress. KAI implements it via PodGroups, with elastic min/max gangs for autoscaling.

**Q5. KAI Scheduler — what is it, where did it come from, how does it relate to the default scheduler?**
KAI is the open-sourced Run:ai scheduler — NVIDIA acquired Run:ai (~$700M, closed Dec 2024) and open-sourced KAI (Apache 2.0) at KubeCon EU in April 2025. It runs as a **secondary scheduler alongside** kube-scheduler (pods opt in via `schedulerName`), not a replacement. Core features: gang scheduling, hierarchical queues with quota/over-quota, DRF + time-based fair-share, action order Allocate→Consolidate→Reclaim→Preempt, decoupled priority vs preemptibility, and fractional GPU sharing. It shares kube-batch lineage with Volcano and is complementary to Kueue (which is a queueing layer). Open-sourcing was both strategic (make NVIDIA's primitives the default) and antitrust-motivated (avoid orchestration lock-in).

**Q6. DDP vs FSDP vs FSDP2 — when do you use each?**
DDP replicates the full model on every GPU and all-reduces gradients each step — use it when the model + optimizer states fit on one GPU and you just want data-parallel throughput. FSDP is ZeRO-3: it shards params, gradients, and optimizer states across workers, all-gathering params just-in-time for forward/backward and reduce-scattering gradients — use it when the model is too big to fit. FSDP1 used a flat-parameter representation and is now deprecated; **FSDP2** (`fully_shard`) shards per-parameter using DTensor, which composes cleanly with tensor parallelism (2D meshes) — the current standard on torch 2.4+. For trillion-scale you go beyond data parallelism to Megatron-Core's TP × PP × CP × EP × DP.

**Q7. What does NCCL actually do, and how does it use the hardware?**
NCCL implements the collective communication primitives distributed training runs on — AllReduce, AllGather, ReduceScatter, AllToAll — each as a single fused kernel. It auto-detects topology at init and selects transports by distance: NVLink/NVSwitch within a node, InfiniBand or RoCE with GPUDirect RDMA (the NIC DMA'ing straight into GPU HBM) across nodes. Algorithms: Ring (bandwidth-optimal), Tree (latency-optimal at scale), NVLS (NVLink SHARP). **SHARP** offloads the reduction arithmetic into the switch ASIC — in NCCL 2.27 it cut SM usage to ≤6 from ≥16, which matters at 1000+ GPU scale. `nvidia_peermem` is the module that enables GPUDirect RDMA.

**Q8. Why the NGC container instead of `pip install torch`?**
The `nvcr.io/nvidia/pytorch:YY.MM-pyN` image is a monthly, re-validated, ABI-matched stack — PyTorch (an NVIDIA-patched `a0` build), CUDA, cuDNN, NCCL, TransformerEngine, Apex, DALI all built and integration-tested together. With pip you're resolving compatibility yourself and you miss NVIDIA extras like TransformerEngine (FP8/FP4). The tag is a *calendar* version, not the PyTorch version — `26.04` is the April 2026 build.

**Q9. libcuda vs libcudart, and the compatibility story.**
`libcuda.so` is the user-mode driver — the low-level Driver API (`cuLaunchKernel`), ships *with the GPU driver*, admin-owned, bottom of the stack. `libcudart.so` is the CUDA Runtime (`cudaMalloc`), ships *in the toolkit/container*, linked into the framework, dev-owned, top of the stack. Compatibility floors: CUDA 13.x needs driver ≥580, 12.x ≥525. The `cuda-compat` forward-compatibility package ships a newer user-mode driver so a newer CUDA container runs on an older *kernel* driver — which is how a CUDA 13 image runs on a datacenter node pinned to a 12-era driver. The container image is the clean contract between the two halves of the stack.

**Q10. Explain disaggregated serving and NVIDIA Dynamo.**
Prefill and decode have opposite bottlenecks: prefill processes the whole prompt and is compute-bound (wants low tensor-parallelism to cut comm), decode generates one token at a time and is memory-bandwidth-bound (wants high tensor-parallelism). Serving both in one pool forces a bad compromise. Dynamo splits them into independently-scaled GPU pools. Its four components: **Smart Router** (KV-cache-aware routing via Radix trees — send each request to the GPU already holding its KV cache, cutting recompute and TTFT), **Planner** (dynamically shifts GPUs between prefill and decode by SLO), **KVBM** (tiers the KV cache across HBM→DRAM→SSD→object storage), **NIXL** (fast KV transfer, ~270 GB/s across 8×H100). It's engine-agnostic (vLLM, TRT-LLM, SGLang), open-source, the successor to Triton for distributed LLM serving, and NVIDIA cites 30× throughput on DeepSeek-R1 on GB200 NVL72.

**Q11. LWS vs Grove — when would you reach for each?**
LeaderWorkerSet models one homogeneous leader+workers group as a single replicated unit — perfect for a single model sharded across nodes (multi-node vLLM). Grove models a *whole multi-component system* — prefill leaders/workers, decode leaders/workers, a router, a frontend — as one CRD hierarchy (PodClique → PodCliqueScalingGroup → PodCliqueSet) with per-component scaling and startup ordering, emitting a PodGang that KAI gang-schedules. You use LWS for straightforward multi-node serving; Grove when you're deploying a disaggregated Dynamo topology.

**Q12. How do GPUs get shared, and what are the trade-offs?**
Three mechanisms. **MIG** partitions the GPU in *hardware* into isolated instances with dedicated compute + memory + fault isolation (A100/H100/Blackwell, profiles like 3g.20gb) — strongest isolation, best for multi-tenant QoS. **Time-slicing** round-robins the SM in software with *no* memory or fault isolation — oversubscription for bursty dev/inference. **MPS** runs multiple process contexts concurrently — better spatial sharing than time-slicing, weaker isolation than MIG. On top, KAI adds scheduler-level fractional sharing and quotas. The honest caveat: KAI's default fractional sharing has no memory isolation — for hard multi-tenancy pair it with MIG or HAMi.

---

## Section 2 — System design (whiteboard rounds)

**Q13. Design a platform to train and serve LLMs for 200 ML engineers on a shared 512-GPU cluster.**
Structure the answer: (1) *Requirements* — multi-tenant fairness, high utilization, both training and inference, failure resilience, observability, cost accountability. (2) *Node layer* — GPU Operator for driver/toolkit/DCGM/MIG; Network Operator for RDMA; DRA driver for GPU allocation. (3) *Scheduling* — KAI with hierarchical queues per team (guaranteed quota + over-quota borrowing), gang scheduling for training, fractional GPU + MIG for inference; separate training and inference pools or priority classes with preemption. (4) *Workload APIs* — Kubeflow Trainer for training, LWS/Grove for serving. (5) *Topology* — Topograph labels + topology-aware placement. (6) *Resilience* — checkpoint/restart, DCGM health eviction, torchrun elasticity, auto-requeue. (7) *Observability* — DCGM-Exporter → Prometheus/Grafana, per-team utilization dashboards (Run:ai). (8) *Cost* — chargeback via quota + utilization metrics. Name trade-offs: MIG QoS vs flexibility, dedicated vs shared pools, spot vs on-demand.

**Q14. A customer's 1000-GPU training run keeps failing halfway. What do you do?**
Diagnose systematically: is it hardware (DCGM/XID errors, a bad NVLink/NIC), NCCL (timeouts, wrong transport — check `NCCL_DEBUG=INFO`, GPUDirect RDMA enabled?), scheduling (partial placement → enforce gang scheduling), or the job itself (OOM → FSDP/activation checkpointing, or a data issue)? Then resilience: framework checkpointing (NeMo) so a restart resumes; torchrun `--max-restarts` and elasticity to tolerate membership changes; DCGM to health-evict bad nodes and requeue; topology-aware retries. Frame it as pre-sales: "here's how the platform makes a 1000-GPU run *survivable*, not just fast."

**Q15. Design a low-latency, high-throughput inference service for a reasoning model.**
Disaggregated serving with Dynamo: separate prefill (compute-bound, low TP) and decode (bandwidth-bound, high TP) pools scaled independently; Smart Router for KV-cache-aware routing to minimize TTFT; KVBM to tier the KV cache and NIXL to move it fast; Planner to rebalance GPUs against SLOs. On K8s: Grove to model the topology, KAI to gang-schedule it onto a shared NVLink domain (GB200) so KV transfer stays on NVLink. Quantize with TensorRT-LLM (NVFP4 on Blackwell). Package as NIM for a supported OpenAI-compatible endpoint. Discuss the latency (TTFT, inter-token) vs throughput (tokens/sec/GPU) trade-off and how disaggregation lets you tune each independently.

---

## Section 3 — Product, strategy & "why NVIDIA"

**Q16. Why did NVIDIA open-source the Run:ai scheduler?**
Two reasons. Strategic: NVIDIA sells GPUs, and utilization is the ceiling on GPU demand — a better scheduler raises effective utilization, which means more workloads per cluster and validation for bigger cluster purchases; open-sourcing KAI makes NVIDIA's scheduling primitives the default in the Kubernetes AI stack while the commercial NVIDIA Run:ai platform (multi-tenant UI, dashboards, policy) sits on top. And antitrust: the Run:ai acquisition drew EU scrutiny over GPU-orchestration market power, so open-sourcing (and keeping it able to support non-NVIDIA hardware) defused lock-in concerns. It fits NVIDIA's broader "own the full stack from silicon to orchestration, but on open standards" play — KAI, Dynamo, DRA/CDI upstream.

**Q17. What's the "AI factory" and how do you pitch it?**
Reframe the datacenter as a factory that manufactures intelligence — tokens. You measure it in tokens/sec, revenue per rack, and joules per token, not FLOPs on a spec sheet. NVIDIA sells the *whole factory*: GPU → NVLink/NVSwitch → Spectrum-X/Quantum networking → DGX/HGX systems → NVIDIA AI Enterprise software (NIM + NeMo + Run:ai + CUDA-X) → reference designs like DSX. The pitch to an enterprise is that every layer is co-designed for throughput and utilization, so the expensive asset — the GPUs — is amortized as hard as physically possible. It's also the frame for the software business: AI Enterprise is a recurring per-GPU subscription on top of the hardware.

**Q18. What NVIDIA products should this stack make me think about, and where are they going?**
Hardware: Blackwell / GB200-GB300 NVL72 today; Vera Rubin (announced GTC 2026, H2 2026) on an annual cadence; Feynman next. Networking: Spectrum-X Ethernet and Quantum-X InfiniBand with co-packaged optics. Systems: DGX/HGX, DGX Cloud, DGX Spark/Station. Software: AI Enterprise (NIM, NeMo, Run:ai/KAI, CUDA-X), Dynamo for inference, the Nemotron open-model families. The direction from GTC 2025→2026 is agentic AI, physical AI/robotics, and enterprise/structured-data — all riding the same Kubernetes-AI stack you're pitching.

**Q19. Why do you want to be a Developer Advocate rather than a pure engineer?**
(Your authentic answer — but hit these beats:) I'm energized by the *translation* — taking genuinely hard technology and making it land for the developer who has to adopt it, through demos, code, and clear talks. I have the depth to be credible and the communication instinct to make it click, and I'd rather multiply impact across a community than ship one feature. This stack is the perfect canvas because the story — "you write train.py, everything else is automatic" — is both technically deep and genuinely exciting, and I want to be the person who makes ten thousand developers *get* it.

---

## Section 4 — Behavioral / STAR (prepare 6–8 stories)

Map your real stories to these themes (the reqs + culture tell you what they weight). Use **Situation → Task → Action → Result**, lead with the result, and quantify.

1. **Translating deep tech for a non-expert audience** — a talk/blog/demo that changed adoption. *(Directly = "articulate the value prop of an emerging technology.")* Bring numbers: views, stars, workshop attendance, pipeline influenced.
2. **Turning a skeptic/objection into a win** — a developer or customer you converted with an honest answer or a demo.
3. **Built a demo/OSS that drove measurable adoption** — the portfolio story; have the repo linkable.
4. **Handling ambiguity / conflicting requirements** — explicitly screened in the behavioral round.
5. **Intellectual honesty** — a time you told a customer/PM the product *couldn't* do something, or admitted you were wrong. (Maps to "the mission is the boss," not selling vapor.)
6. **Cross-functional influence without authority** — working across product marketing + engineering (mirrors NVIDIA's flat, broadcast culture).
7. **First-principles / "speed of light"** — you benchmarked against the theoretical limit, not the status quo, and found the real headroom.
8. **Learning velocity** — you ramped fast on a new domain. (They ship a new architecture yearly; they hire for slope.)

**Culture-fit answers to have ready:**
- *"Why NVIDIA?"* — the mission (the compute substrate for the AI era), the full-stack ambition, and that it's the rare place where deep tech and storytelling are equally valued.
- *"How do you handle being wrong / disagreement?"* — default to broadcast and intellectual honesty; surface it early and openly; the mission decides, not ego.
- *"Tell me about a time you disagreed with a decision."* — show you can push back with data, then commit.

---

## Section 5 — Curveballs & "gotchas" (correct the misconception cleanly)

- **"KAI replaces the Kubernetes scheduler, right?"** → No — it runs *alongside* kube-scheduler as a secondary scheduler; pods opt in.
- **"Is DRA still experimental?"** → Core APIs went GA in v1.34 (`resource.k8s.io/v1`); several sub-features remain beta/alpha, but the foundation is stable.
- **"Time-slicing isolates GPU memory."** → It does not — only MIG gives hardware memory + fault isolation.
- **"Fractional GPU sharing = MIG."** → KAI's default fractional sharing is a soft, scheduler-level mechanism with no memory isolation by default; MIG is hardware partitioning. Different things.
- **"ComputeDomains are just NVLink."** → NVLink/MNNVL is the hardware fabric; ComputeDomains dynamically provision and *isolate* per-workload IMEX memory domains on top of it.
- **"Dynamo fully replaces Triton."** → For distributed/multi-node LLM serving, yes it's the successor; Triton remains for multi-framework/single-node and isn't a drop-in swap.
- **"GPUDirect RDMA is NVLink."** → Different fabrics: GPUDirect RDMA is NIC↔GPU over IB/RoCE (inter-node); NVLink/MNNVL is GPU↔GPU memory (intra-rack).
- **"You can run the device plugin and the DRA GPU driver on the same GPUs."** → They conflict — disable the device plugin when using DRA for GPUs.
- **"`torch.distributed.launch` is current."** → It's soft-deprecated in favor of `torchrun`, which adds elasticity, fault tolerance, and c10d rendezvous.

---

## Final prep checklist
- [ ] Can you draw the partial-gang deadlock from memory?
- [ ] Can you name the four DRA objects and the four Dynamo components without notes?
- [ ] Can you state the DRA GA version and the KAI open-source date/license?
- [ ] Can you deliver the 3-min, 10-min, and 25-min versions of the abstraction pitch?
- [ ] Do you have a public repo + a recorded demo linkable?
- [ ] Do you have 8 STAR stories, each with a number in the Result?
- [ ] Can you field all 7 pre-sales objections cold?
- [ ] For every technical claim, do you know the *business* translation?

> **Honesty note for the interview:** several version numbers and product details in this pack are current as of mid-2026 but move fast (KAI, Dynamo, DRA sub-features, NGC tags). If you're unsure of an exact version live, say the *mechanism* confidently and flag the number as "let me confirm the exact release" — that intellectual honesty is itself what the role screens for.
