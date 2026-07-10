# Week 3 Notes — AI Infrastructure part 2: Networking, DPUs, Datacenter, Storage

## 1. AI networking fundamentals

### Why AI traffic is different
- All-reduce traffic pattern means…
- Tail latency matters because…
- East-west vs north-south traffic:

### InfiniBand
- Key properties (lossless, RDMA, latency):
- SHARP (in-network computing):
- Subnet manager / UFM:
- NVIDIA product names (switches / adapters):
- When to recommend:

### Ethernet / RoCE
- RoCE is…
- What it needs to work well (PFC/ECN, lossless tuning):
- When to recommend:

### Spectrum-X
- What it is (components):
- How it differs from generic Ethernet:

### Fabric design
- Compute fabric vs storage/management fabric:
- Rail-optimized topology:

## 2. GPUDirect family

- The problem without GPUDirect (data path):
- GPUDirect RDMA:
- GPUDirect Storage:
- GPUDirect P2P:
- Hardware/software requirements:
- Who uses it under the hood (NCCL, DALI…):

## 3. DPUs / BlueField

- The CPU / GPU / DPU division of labor:
- BlueField DPU — what's on the card:
- Offload category 1 — networking:
- Offload category 2 — storage:
- Offload category 3 — security:
- Why isolation matters (multi-tenant / zero trust):
- DOCA:
- BlueField-3 SuperNIC (Spectrum-X role):
- Exam trigger phrases that mean "DPU":

## 4. Power and cooling

- DGX H100 node power:
- AI rack density vs traditional rack:
- Air cooling — techniques and rough limit:
- Rear-door heat exchanger:
- Direct liquid cooling — when it's required, benefits:
- PUE — definition and what "good" looks like:
- Other facility concerns (floor loading, power redundancy):

## 5. Storage for AI

- Training I/O pattern (reads):
- Checkpointing (writes):
- Tier: local NVMe —
- Tier: parallel / high-perf file systems (examples) —
- Tier: object storage —
- "Slow storage = idle GPUs" — explain:
- GPUDirect Storage tie-in:

## 6. On-prem vs cloud vs hybrid

| Factor | Cloud | On-prem | Hybrid |
|---|---|---|---|
| Cost model | | | |
| Time to start | | | |
| Best when utilization is… | | | |
| Data sovereignty / gravity | | | |
| Ops burden | | | |

- DGX Cloud is…
- GPU-specialty clouds / colo as a middle path:
- The 4–5 questions I'd ask a customer to pick a model:

## Parking lot
-
