# Week 3 Plan — AI Infrastructure, part 2: Networking, DPUs, datacenter, storage (Domain 2, 40%)

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Dates: Mon 2026-07-27 → Fri 2026-07-31 · 5 days × ~2 h

Second half of the 40% domain: everything outside the server — the network fabric, DPUs, power/cooling, storage, and deployment models. Keep daily flashcards going (now Domains 1 + 2).

## Prerequisites before Monday

- Companion lesson: [Week 03 companion — network fabrics, roofline thinking, and SGEMM support](../../../companion-lessons/week-03.md).
- Math support: matmul FLOPs `2*M*N*K`, arithmetic intensity, and the roofline ridge point.
- Systems support: latency vs bandwidth, InfiniBand/RoCE/RDMA vocabulary, shared-memory tiling, and cuBLAS-as-reference thinking.
- Gate: compute FLOPs for a `1024 x 1024` matmul and explain why GPUDirect RDMA matters.

---

## Day 1 (Mon) — AI networking fundamentals: InfiniBand vs Ethernet/RoCE
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (60 min) Into `notes.md`:
  - Why AI training traffic is different: synchronized bursts (all-reduce), elephant flows, extreme sensitivity to tail latency — one slow packet stalls every GPU
  - **InfiniBand**: lossless by design (credit-based flow control), RDMA native, lowest latency, in-network computing (**SHARP** — switch performs reductions), managed by a subnet manager (UFM); NVIDIA Quantum switches + ConnectX HCAs; the default for DGX SuperPOD training fabrics
  - **RoCE (RDMA over Converged Ethernet)**: RDMA on Ethernet; needs a well-tuned lossless-ish fabric (PFC/ECN); more familiar operationally, broader vendor ecosystem
  - **Spectrum-X**: NVIDIA's Ethernet platform purpose-tuned for AI (Spectrum-4 switch + BlueField-3 SuperNIC, adaptive routing/congestion control)
  - Decision framing: max performance & scale → InfiniBand; Ethernet-standardized shop / cloud alignment → RoCE/Spectrum-X
- (30 min) Read NVIDIA networking pages: Quantum InfiniBand overview + Spectrum-X overview.
- (15 min) Note the two-fabric pattern: compute fabric (IB, east-west) separate from storage/management fabric (Ethernet).

## Day 2 (Tue) — GPUDirect RDMA and friends
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (55 min) The GPUDirect family into `notes.md`:
  - Baseline problem: without GPUDirect, GPU↔NIC data bounces through CPU system memory (extra copies, CPU cycles, latency)
  - **GPUDirect RDMA**: NIC reads/writes GPU memory directly over PCIe — NCCL uses this for multi-node training
  - **GPUDirect Storage (GDS)**: storage/NVMe → GPU memory directly, bypassing CPU bounce buffer
  - **GPUDirect P2P**: GPU↔GPU within a node without going through system memory
  - Requirements: RDMA-capable NIC (ConnectX), proper drivers/toolkit — and in Kubernetes, the network operator (nice tie-in to your GPU Operator work)
- (35 min) Read the GPUDirect RDMA overview (developer.nvidia.com/gpudirect) and an NVIDIA blog post on multi-node training data paths.
- (15 min) Rail-optimized topology: each GPU's NIC connects to its own switch "rail" (8 NICs per DGX → 8 rails) — recognize the term.

## Day 3 (Wed) — DPUs / BlueField
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (50 min) Into `notes.md`:
  - The three processors story: CPU (general compute), GPU (accelerated compute), **DPU (data processing unit: infrastructure offload)**
  - **BlueField DPU** = Arm cores + ConnectX NIC + accelerators on one card. Offloads/isolates: networking (OVS, overlay), storage (NVMe-oF virtualization), security (firewall, encryption, zero-trust microsegmentation) — freeing host CPUs for tenant/application work and isolating infrastructure from the host (key in multi-tenant / bare-metal-cloud)
  - **DOCA**: the SDK/framework for programming BlueField (analogy: DOCA is to DPUs what CUDA is to GPUs)
  - BlueField-3 SuperNIC role in Spectrum-X (congestion control endpoint)
- (40 min) Read NVIDIA BlueField product page + one "What is a DPU?" NVIDIA blog post.
- (15 min) Exam framing drill: "customer wants to isolate infrastructure services from tenant workloads / free CPU cores consumed by networking+security" → DPU.

## Day 4 (Thu) — Datacenter design: power, cooling, storage for AI
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (45 min) Power & cooling into `notes.md`:
  - Scale of the problem: DGX H100 ≈ 10.2 kW per node; GPU racks run 40–120+ kW vs ~10–15 kW traditional racks — power delivery and floor loading become first-order constraints
  - Cooling: air (hot/cold aisle containment, limits around ~40 kW/rack) → rear-door heat exchangers → **direct liquid cooling (DLC)** — required territory for Blackwell NVL72-class racks; liquid is also more energy-efficient (better PUE)
  - **PUE** definition (total facility power ÷ IT power; closer to 1.0 = better)
- (45 min) Storage for AI into `notes.md`:
  - Training I/O pattern: many parallel readers, high sequential throughput, plus checkpoint bursts (huge writes)
  - Tiers: local NVMe (scratch/cache) → parallel/high-performance NAS file systems (Lustre, GPFS/Spectrum Scale, VAST, WEKA — recognize category, not details) → object storage (datasets/archives, S3-style)
  - Why slow storage = idle GPUs (data starvation); GPUDirect Storage tie-in
- (15 min) Skim a DGX SuperPOD RA storage + facilities section.

## Day 5 (Fri) — On-prem vs cloud vs hybrid + review
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (40 min) Deployment models into `notes.md`:
  - **Cloud**: OPEX, elasticity, fast start, GPU instance families (all majors offer A100/H100/B200-class); watch data egress costs and availability/quota
  - **On-prem**: CAPEX, best TCO at sustained high utilization, data sovereignty/security, full control (needed for custom fabrics); long lead times, ops burden
  - **Hybrid**: steady-state on-prem + burst to cloud; data gravity as the deciding force
  - **DGX Cloud**: NVIDIA's rented AI supercomputing (SuperPOD-class infra in partner clouds); colo/GPU-cloud providers (CoreWeave-style) as middle path
  - Decision drivers checklist: utilization %, data sovereignty, dataset location, time-to-start, in-house ops skills
- (20 min) Draw the full stack picture: rack → node (HGX) → NVLink domain → IB fabric → storage → cloud/on-prem wrapper. This is your mental map for the whole domain.
- (45 min) `self-check.md` closed-notes; restudy misses.

---

## Exit criteria — check every box before starting week 4

- [ ] I can explain why AI training traffic breaks normal datacenter network assumptions (synchronized bursts, tail latency)
- [ ] I can compare InfiniBand vs RoCE vs Spectrum-X and recommend one for a given customer scenario with a reason
- [ ] I can explain GPUDirect RDMA and GPUDirect Storage — what copy/bounce they eliminate
- [ ] I can define a DPU, list the three offload categories (network, storage, security), and name DOCA
- [ ] I can state rough power figures (DGX H100 ~10 kW, AI racks 40–120 kW) and explain when liquid cooling becomes necessary; I can define PUE
- [ ] I can describe the AI storage tiers and why storage throughput affects GPU utilization
- [ ] I can argue on-prem vs cloud vs hybrid using utilization, data gravity, sovereignty, and time-to-start
- [ ] I scored ≥ 80% on the week 3 self-check
