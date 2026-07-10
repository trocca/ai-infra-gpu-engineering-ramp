# Week 3 Self-Check — Networking, DPUs, Datacenter, Storage

Target: ≥ 80% (14/18) before moving to week 4.

**1. Why is tail latency more damaging to distributed training than to typical web traffic?**

<details><summary>Answer</summary>
Training steps are synchronized: every GPU must finish the gradient all-reduce before any can proceed. One delayed flow stalls the entire job — the collective runs at the speed of the slowest path. Web requests are independent, so one slow response affects one user, not the whole system.
</details>

**2. Name three properties that make InfiniBand attractive for AI training fabrics.**

<details><summary>Answer</summary>
Lossless transport by design (credit-based flow control — no drops to retransmit), native RDMA with very low latency, and in-network computing (SHARP offloads all-reduce operations into the switches). Also: adaptive routing and a centrally managed subnet (UFM).
</details>

**3. What is RoCE and what does it require from the Ethernet fabric to perform well?**

<details><summary>Answer</summary>
RDMA over Converged Ethernet — running the RDMA verbs used by InfiniBand over Ethernet networks. To perform well it needs the fabric tuned for (near-)losslessness: Priority Flow Control (PFC), ECN-based congestion control, and careful QoS configuration — otherwise packet loss devastates RDMA performance.
</details>

**4. What is NVIDIA Spectrum-X?**

<details><summary>Answer</summary>
NVIDIA's end-to-end Ethernet networking platform purpose-built for AI: Spectrum-4 (and later) Ethernet switches paired with BlueField-3 SuperNICs, adding adaptive routing and telemetry-based congestion control so Ethernet approaches InfiniBand-class performance for AI workloads while staying standards-based Ethernet.
</details>

**5. A customer runs an Ethernet-standardized enterprise shop and refuses to operate a second fabric technology, but wants strong multi-node training performance. What do you recommend?**

<details><summary>Answer</summary>
Spectrum-X (or well-engineered RoCE): it keeps their Ethernet operational model and tooling while adding AI-specific congestion control and adaptive routing. Pure max-performance greenfield clusters would default to InfiniBand, but their constraint rules it out.
</details>

**6. What is SHARP?**

<details><summary>Answer</summary>
Scalable Hierarchical Aggregation and Reduction Protocol — InfiniBand in-network computing where the switches themselves perform reduction operations (e.g. summing gradients) as data flows through, cutting the data volume and latency of all-reduce instead of doing all the math on the endpoints.
</details>

**7. What data path does GPUDirect RDMA eliminate?**

<details><summary>Answer</summary>
Without it, data moving between GPU memory and the network is staged through CPU system memory (GPU → system RAM → NIC), costing extra copies, latency, and CPU cycles. GPUDirect RDMA lets the RDMA NIC read/write GPU memory directly over PCIe.
</details>

**8. What is GPUDirect Storage and which workloads benefit most?**

<details><summary>Answer</summary>
A direct DMA path between storage (local NVMe or remote) and GPU memory, bypassing the CPU bounce buffer. Benefits data-hungry workloads: training with large datasets, data analytics, anything where I/O throughput to the GPU is the bottleneck.
</details>

**9. Define a DPU and its three classic offload categories.**

<details><summary>Answer</summary>
A Data Processing Unit is a programmable SoC (Arm cores + high-speed NIC + accelerators — NVIDIA's is BlueField) that offloads infrastructure services from the host CPU: (1) networking (virtual switching, overlays), (2) storage (NVMe-oF, storage virtualization), (3) security (firewalling, encryption, microsegmentation). It also isolates the infrastructure domain from the host — key for multi-tenant and zero-trust designs.
</details>

**10. What is DOCA?**

<details><summary>Answer</summary>
NVIDIA's SDK and runtime framework for programming BlueField DPUs — the DPU analog of what CUDA is for GPUs.
</details>

**11. Roughly how much power does a DGX H100 draw, and why does this change datacenter design?**

<details><summary>Answer</summary>
~10.2 kW per node (max). Four nodes exceed 40 kW in a rack — versus ~10–15 kW for traditional racks — so power delivery, floor loading, and cooling become the binding constraints; many facilities can only part-fill racks or must upgrade to liquid cooling.
</details>

**12. When does direct liquid cooling become effectively mandatory, and what are its two main benefits?**

<details><summary>Answer</summary>
When rack density exceeds what air can remove (roughly beyond ~40 kW/rack in practice) — e.g. Blackwell GB200 NVL72 racks (~120 kW) are liquid-cooled by design. Benefits: removes far more heat per rack (enables density) and improves energy efficiency (lower PUE — liquid transfers heat more efficiently than air).
</details>

**13. What is PUE and what does a value close to 1.0 mean?**

<details><summary>Answer</summary>
Power Usage Effectiveness = total facility power ÷ IT equipment power. A value near 1.0 means almost all electricity goes to computing rather than cooling/overhead (e.g. 1.1 is excellent; 2.0 means as much power goes to overhead as to IT).
</details>

**14. Why does storage performance directly affect GPU utilization during training?**

<details><summary>Answer</summary>
GPUs process batches as fast as data arrives; if the storage system can't sustain the read throughput of many parallel data loaders, GPUs sit idle waiting for data ("data starvation") — you pay for expensive accelerators to wait on I/O. Checkpoint writes can similarly stall training if the storage can't absorb the burst.
</details>

**15. Name the three storage tiers in a typical AI cluster and their roles.**

<details><summary>Answer</summary>
Local NVMe on each node (scratch/cache — fastest, not shared), a shared high-performance/parallel file system (e.g. Lustre, Spectrum Scale, VAST, WEKA — hot training data, checkpoints), and object storage (datasets at rest, archives, cheapest per TB).
</details>

**16. A startup needs to fine-tune models for a 6-week project, has no datacenter, and needs to start this month. Cloud or on-prem, and name the two strongest reasons.**

<details><summary>Answer</summary>
Cloud. (1) Time-to-start: GPU instances are available now vs months of procurement/build-out; (2) cost shape: short, bursty usage fits OPEX pay-per-use — on-prem TCO only wins with sustained high utilization over years.
</details>

**17. What is "data gravity" and how does it influence the on-prem vs cloud decision?**

<details><summary>Answer</summary>
Large datasets are expensive and slow to move (and egress fees make pulling data out of a cloud costly), so compute tends to be placed where the data already lives. Petabytes on-prem argue for on-prem GPUs; data born in a cloud argues for training there.
</details>

**18. What is a rail-optimized network topology?**

<details><summary>Answer</summary>
A SuperPOD-style design where each GPU's dedicated NIC connects to its own "rail" (its own leaf switch plane) — 8 GPUs/node → 8 rails — so same-rank GPUs across nodes communicate with minimal hops and congestion, matching NCCL's communication pattern.
</details>
