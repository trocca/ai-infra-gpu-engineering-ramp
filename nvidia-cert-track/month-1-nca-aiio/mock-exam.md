# NCA-AIIO Mock Exam — 50 Questions

**Conditions:** 60 minutes, closed book, no lookups. One correct answer per question. Write your answers elsewhere — the answer key is at the very bottom. Weighting mirrors the real exam: ~19 Essential AI Knowledge, ~20 AI Infrastructure, ~11 AI Operations.

Passing target for readiness: 40/50 (80%).

---

**Q1.** A hospital deploys a system that flags likely tumors in CT scans after being trained on thousands of radiologist-labeled images. Which category best describes this system?

- A. Unsupervised learning, because the model discovers tumors on its own
- B. Supervised deep learning, because a neural network learned from labeled examples
- C. Reinforcement learning, because the model is rewarded for correct diagnoses
- D. Classical rule-based AI, because medical criteria define tumors

**Q2.** A data scientist says her model's loss stopped decreasing during training. Which process computes the values used to update the model's weights each iteration?

- A. The forward pass, which adjusts weights as data flows through
- B. Backpropagation, which computes gradients of the loss with respect to each weight
- C. Quantization, which reduces weight precision until loss decreases
- D. The KV cache, which stores weight updates between batches

**Q3.** A team must choose infrastructure for two workloads: (1) pretraining a large language model, (2) serving that model to end users with strict response-time SLAs. Which pairing of primary optimization goals is correct?

- A. (1) latency per sample; (2) total throughput only
- B. (1) throughput/time-to-train; (2) low latency at acceptable throughput
- C. (1) minimal memory use; (2) maximal batch size regardless of latency
- D. Both optimize identically since they use the same model

**Q4.** A 13B-parameter model trains comfortably on a GPU but the same GPU can serve a much larger model for inference. What best explains this?

- A. Inference uses faster memory than training
- B. Training requires storing activations, gradients, and optimizer states in addition to weights
- C. Inference automatically compresses models to 1/10th size
- D. Training frameworks are less memory-efficient than serving frameworks by design

**Q5.** A customer asks why their GPU cluster trains a vision model 40× faster than their CPU farm. Which explanation is most accurate?

- A. GPUs run at much higher clock frequencies than CPUs
- B. GPUs have thousands of parallel cores, very high memory bandwidth, and Tensor Cores suited to matrix math
- C. GPUs have larger caches and better branch prediction
- D. CPUs cannot execute floating-point math natively

**Q6.** Which task in an AI pipeline is generally best left on the CPU?

- A. The gradient all-reduce across nodes
- B. Convolution layers during training
- C. Orchestration logic and branch-heavy business rules around the model
- D. Attention computation in a transformer

**Q7.** What distinguishes a Tensor Core from a CUDA core?

- A. Tensor Cores are only used for graphics workloads
- B. Tensor Cores execute matrix multiply-accumulate operations on tiles in reduced/mixed precision; CUDA cores execute general scalar/vector ops
- C. CUDA cores are hardware; Tensor Cores are a software library
- D. Tensor Cores only operate in FP64 for scientific accuracy

**Q8.** A PyTorch training job runs on NVIDIA GPUs. Which library is PyTorch most likely calling under the hood for optimized convolution and attention primitives?

- A. cuDNN
- B. DOCA
- C. DCGM
- D. RAPIDS

**Q9.** During multi-node distributed training, which NVIDIA library performs the gradient all-reduce between GPUs, automatically using NVLink and InfiniBand where available?

- A. cuBLAS
- B. DALI
- C. NCCL
- D. TensorRT

**Q10.** A customer has a trained model and wants to cut inference latency on their NVIDIA GPUs with techniques like layer fusion and reduced precision, without retraining. What do you recommend first?

- A. Rewrite the model in CUDA C++
- B. Optimize the model with TensorRT
- C. Move inference to CPUs to reduce cost
- D. Increase the training batch size

**Q11.** A platform team must serve models built in PyTorch, ONNX, and TensorRT behind one endpoint with dynamic batching and metrics. Which NVIDIA tool fits best?

- A. NGC
- B. Triton Inference Server
- C. nvidia-smi
- D. cuDNN

**Q12.** A developer wants a GPU-optimized PyTorch container, a pretrained speech model, and a Helm chart to deploy them. Where does NVIDIA provide all three?

- A. The CUDA Toolkit installer
- B. The NGC catalog
- C. GitHub releases of the driver
- D. The DCGM exporter repository

**Q13.** A regulated enterprise says: "The frameworks are free — why pay for NVIDIA AI Enterprise?" Which answer is most accurate?

- A. It unlocks hidden GPU performance unavailable to free users
- B. It provides enterprise support, security patching, stable branches, and certified deployment on mainstream servers and clouds
- C. It is required for CUDA to run legally in production
- D. It replaces the need for Kubernetes or Slurm

**Q14.** A team wants to stand up an OpenAI-compatible LLM endpoint on their own GPUs in hours, using a prebuilt, supported container with an optimized runtime inside. Which NVIDIA offering matches?

- A. NVIDIA NIM
- B. cuBLAS
- C. Base Command Manager
- D. GPU Operator

**Q15.** Which sequence correctly orders the AI development lifecycle?

- A. Training → data collection → deployment → problem definition
- B. Problem definition → data collection/preparation → training → validation → deployment → monitoring
- C. Deployment → monitoring → training → data preparation
- D. Data collection → deployment → training → validation

**Q16.** Six months after deployment, a fraud-detection model's accuracy quietly degrades as customer behavior changes. Which practice catches this, and what is the phenomenon called?

- A. Load testing; model overflow
- B. Production monitoring of model/data statistics; model drift
- C. Unit testing; gradient explosion
- D. Checkpointing; catastrophic forgetting

**Q17.** A legal firm wants an LLM assistant that answers from their private, frequently-updated document repository without retraining the model. Which approach fits best?

- A. Retrieval-augmented generation (RAG)
- B. Full pretraining from scratch on the documents
- C. Reducing the model to INT8
- D. Increasing the context window with no other changes

**Q18.** An inference service increases its batch size from 1 to 32. What is the most likely effect?

- A. Higher throughput and higher per-request latency
- B. Lower throughput and lower latency
- C. No effect, since batch size only matters in training
- D. Lower GPU utilization

**Q19.** Which pairing of industry and flagship AI use case is most typical?

- A. Retail — seismic wave simulation
- B. Finance — real-time fraud detection
- C. Healthcare — ad-click prediction
- D. Automotive — protein folding

**Q20.** A customer needs FP8 training with the Transformer Engine for a new LLM project. Which is the earliest GPU architecture that supports this?

- A. Pascal
- B. Ampere
- C. Hopper
- D. Volta

**Q21.** An enterprise wants a turnkey 8-GPU AI system with NVIDIA's integrated software stack and single-vendor NVIDIA support. Their integrator instead proposes a customized server built on the same 8-GPU baseboard. Which pair are they choosing between?

- A. DGX (turnkey NVIDIA system) vs HGX-based OEM server
- B. HGX (turnkey) vs MGX (NVIDIA-built)
- C. DGX vs DPU-based server
- D. EGX vs NGC

**Q22.** In an 8-GPU HGX server, what does NVSwitch provide that point-to-point NVLink alone cannot?

- A. Connectivity to the InfiniBand network
- B. All-to-all communication between every GPU at full NVLink bandwidth simultaneously
- C. Conversion of NVLink traffic to PCIe
- D. Power management for the GPU baseboard

**Q23.** A tensor-parallel inference job constantly exchanges activations between 4 GPUs in one server. Between an SXM system (NVLink/NVSwitch) and a PCIe-card server, roughly how much faster is the H100's GPU-to-GPU path over NVLink 4 compared to PCIe Gen5 x16?

- A. About the same
- B. Roughly 2×
- C. Roughly 7× or more
- D. PCIe is faster than NVLink

**Q24.** Which statement about SXM vs PCIe GPU form factors is correct?

- A. PCIe GPUs always have more memory than SXM
- B. SXM modules support higher power limits and full NVLink/NVSwitch connectivity to all peer GPUs
- C. SXM GPUs cannot be used for training
- D. PCIe GPUs connect to NVSwitch through the motherboard

**Q25.** A data-parallel training job scales poorly from 1 node to 4 nodes: GPUs sit idle at the end of every step. What is the most likely bottleneck?

- A. The gradient all-reduce is limited by inter-node network bandwidth/latency
- B. The GPUs are overheating
- C. The model is too small for its dataset
- D. The learning rate is too high

**Q26.** A 70B-parameter model must be trained across GPUs because single-GPU memory is insufficient, and individual layers must be split across GPUs inside each node. Which parallelism does the intra-node splitting describe, and what does it demand?

- A. Data parallelism; demands large local storage
- B. Tensor (model) parallelism; demands very high-bandwidth GPU interconnect like NVLink
- C. Pipeline parallelism; demands no communication
- D. Batch parallelism; demands CPU offload

**Q27.** Why do datacenter GPUs use HBM instead of standard DDR memory?

- A. HBM is cheaper per gigabyte
- B. HBM's stacked on-package design delivers multi-TB/s bandwidth that keeps parallel compute units fed
- C. HBM is removable and user-upgradable
- D. DDR cannot store floating-point data

**Q28.** A customer wants to serve a 70B-parameter LLM at FP16 precision. Each of their GPUs has 80 GB of memory. What is the minimum consideration you must raise?

- A. The weights alone need ~140 GB, so they need multi-GPU serving or quantization
- B. FP16 is not supported on datacenter GPUs
- C. 70B models can only run on CPUs
- D. They need at least 70 GPUs, one per billion parameters

**Q29.** A training run in FP16 keeps producing NaNs from gradient overflow. Which precision change most directly addresses this while keeping 16-bit memory savings?

- A. Switch to BF16, which keeps FP32's exponent range
- B. Switch to INT8 for training
- C. Switch to FP64 everywhere
- D. Disable Tensor Cores

**Q30.** What does TF32 on Ampere-and-later GPUs provide?

- A. A 32-bit integer mode for databases
- B. Tensor Core acceleration for existing FP32 code with FP32-like range and minimal accuracy impact, no code changes
- C. Double the memory capacity via compression
- D. FP64 emulation for HPC

**Q31.** A team wants to cut LLM serving cost by reducing the model's memory footprint and increasing throughput on existing H100s, accepting a small accuracy validation effort. Which technique do you suggest?

- A. Quantization to FP8/INT8 with an optimized runtime like TensorRT-LLM
- B. Increasing to FP64 precision
- C. Adding more CPU RAM
- D. Retraining the model from scratch

**Q32.** What is a DGX SuperPOD?

- A. A single GPU with extra memory
- B. NVIDIA's validated reference architecture for large clusters of DGX nodes with a high-performance fabric, storage, and management software
- C. A container registry for GPU software
- D. A liquid cooling accessory kit

**Q33.** A greenfield 1,000-GPU training cluster must maximize collective-communication performance, and the team accepts a dedicated fabric with its own management. Which network do you recommend for the compute fabric?

- A. 1 GbE with jumbo frames
- B. NVIDIA Quantum InfiniBand
- C. Wi-Fi 7
- D. Standard TCP/IP Ethernet with no tuning

**Q34.** An enterprise mandates "Ethernet-only" operations but wants near-InfiniBand AI performance. Which NVIDIA platform is purpose-built for this?

- A. Spectrum-X (Spectrum switches + BlueField-3 SuperNICs)
- B. NVLink-C2C
- C. Base Command
- D. GeForce NOW

**Q35.** What does SHARP do in an InfiniBand fabric?

- A. Encrypts all GPU traffic end-to-end
- B. Performs reduction operations inside the switches, offloading all-reduce from the endpoints
- C. Converts InfiniBand to Ethernet frames
- D. Manages GPU firmware updates

**Q36.** Without GPUDirect RDMA, what happens to data moving from GPU memory to a remote node's GPU, and what does GPUDirect RDMA change?

- A. Data is dropped; GPUDirect RDMA adds retransmission
- B. Data is staged through CPU system memory; GPUDirect RDMA lets the NIC access GPU memory directly, cutting copies and latency
- C. Data moves over NVLink; GPUDirect RDMA moves it to PCIe
- D. Nothing changes; it is a marketing term

**Q37.** A multi-tenant GPU cloud wants to run network virtualization, storage services, and security agents without consuming host CPU cores, while isolating that infrastructure from tenant workloads. Which component addresses this directly?

- A. A second x86 CPU socket
- B. A BlueField DPU
- C. More GPU memory
- D. A RAID controller

**Q38.** What is DOCA?

- A. NVIDIA's SDK for programming BlueField DPUs, analogous to CUDA for GPUs
- B. A container orchestration system
- C. A precision format for inference
- D. NVIDIA's monitoring dashboard

**Q39.** A colocation facility offers racks limited to 15 kW with air cooling. A customer wants to deploy four DGX H100 systems in one rack. What is the primary problem?

- A. The systems will not fit physically in any rack
- B. Four nodes draw roughly 40+ kW, far exceeding the rack's power and air-cooling envelope
- C. DGX systems only run on 48 V DC telecom power
- D. InfiniBand cables cannot exit a colo rack

**Q40.** A datacenter reports a PUE of 1.1. What does this indicate?

- A. 10% of GPUs are idle
- B. Highly efficient facility: only ~10% overhead power beyond the IT equipment itself
- C. The facility is over capacity by 10%
- D. Cooling consumes 90% of total power

**Q41.** During training, GPU utilization oscillates: bursts of 100% followed by idle gaps aligned with data loading, and checkpoint saves stall all nodes. Which infrastructure change addresses both symptoms most directly?

- A. Higher-throughput shared storage (parallel file system), optionally with GPUDirect Storage
- B. More GPU memory per device
- C. A faster CPU clock speed
- D. Switching from InfiniBand to gigabit Ethernet

**Q42.** A biotech startup needs H100-class GPUs for a 2-month fine-tuning project starting next week; its data is already in a public cloud. A manufacturer will run inference 24/7 for 5+ years on proprietary factory data that cannot leave the site. Which deployment pairing is most sensible?

- A. Startup: build on-prem; manufacturer: public cloud
- B. Startup: public cloud GPU instances; manufacturer: on-prem GPU infrastructure
- C. Both must use on-prem DGX
- D. Both should use consumer GPUs

**Q43.** In Kubernetes, how does a pod request one full GPU?

- A. By mounting /dev/nvidia0 manually in the pod spec
- B. By setting `nvidia.com/gpu: 1` in the container's resource limits, exposed by the NVIDIA device plugin
- C. By labeling the pod "gpu=true"
- D. GPUs cannot be used in Kubernetes

**Q44.** A platform team hand-installs GPU drivers, the container toolkit, the device plugin, and monitoring on every new Kubernetes node, and upgrades break regularly. Which NVIDIA solution directly automates all of this?

- A. The GPU Operator
- B. TensorRT
- C. NGC CLI
- D. nvcc

**Q45.** A research organization runs queued, gang-scheduled, multi-node batch training with fair-share policies between labs. A product group runs autoscaled, containerized inference microservices. Which scheduler fits each best?

- A. Kubernetes for the lab; Slurm for the microservices
- B. Slurm for the lab; Kubernetes for the microservices
- C. Slurm for both, always
- D. Neither supports GPUs

**Q46.** In Slurm, which mechanism schedules GPUs to jobs?

- A. The device plugin
- B. GRES (generic resources), e.g. `--gres=gpu:4`
- C. MIG manager
- D. Helm charts

**Q47.** An operator needs continuous fleet-wide GPU health monitoring with Prometheus integration and pre-job active diagnostics across 200 nodes. Which tool set is designed for this?

- A. Running nvidia-smi in a loop over SSH
- B. DCGM with the DCGM exporter (and dcgmi diag for active health checks)
- C. Task Manager on each node
- D. TensorBoard

**Q48.** A GPU logs repeated double-bit ECC errors and jobs on that node keep crashing. What is the appropriate operational response?

- A. Ignore them; double-bit errors are auto-corrected
- B. Drain the node, run diagnostics, and replace/RMA the GPU if errors persist — double-bit errors are uncorrectable
- C. Reduce the learning rate of the training jobs
- D. Disable ECC to make the errors disappear

**Q49.** nvidia-smi shows a GPU at 100% utilization but performance is poor, temperature is at the limit, and clocks are far below rated boost. What is happening?

- A. The driver is counting utilization incorrectly
- B. Thermal throttling: the GPU is reducing clocks to stay within its thermal envelope
- C. The GPU is defective and must be replaced immediately
- D. ECC is consuming the missing performance

**Q50.** Three teams share A100s: Team 1 needs guaranteed memory and latency isolation for production inference tenants; Team 2 wants cheap besteffort sharing for dev notebooks; Team 3 runs Windows VMs needing GPU acceleration under VMware. Which mapping is correct?

- A. Team 1: time-slicing; Team 2: MIG; Team 3: MIG
- B. Team 1: MIG; Team 2: time-slicing; Team 3: vGPU
- C. Team 1: vGPU; Team 2: MIG; Team 3: time-slicing
- D. All three: time-slicing

---
---

# ANSWER KEY

*(Score /50. Domains: Q1–Q19 Essential AI Knowledge, Q20–Q42 AI Infrastructure, Q43–Q50 AI Operations — with Q20/Q29-style precision items crossing domains 1↔2.)*

1. **B** — Labeled examples + neural network = supervised deep learning.
2. **B** — Backpropagation computes the gradients that the optimizer uses to update weights.
3. **B** — Training optimizes throughput/time-to-train; serving optimizes latency under SLA at acceptable throughput.
4. **B** — Training stores activations, gradients, and optimizer states on top of weights; inference needs only weights + activations/KV cache.
5. **B** — Massive parallelism, HBM bandwidth, and Tensor Cores — not clock speed or caches.
6. **C** — Serial, branch-heavy control/orchestration logic is CPU territory; the rest are GPU-parallel math or GPU-to-GPU communication.
7. **B** — Tensor Cores do tile-level matrix multiply-accumulate in mixed precision; CUDA cores do general arithmetic.
8. **A** — cuDNN provides optimized DL primitives that frameworks call under the hood.
9. **C** — NCCL implements topology-aware collectives (all-reduce etc.) for multi-GPU/multi-node training.
10. **B** — TensorRT is the inference optimizer/runtime: fusion, precision reduction, kernel tuning without retraining.
11. **B** — Triton serves models from multiple frameworks with dynamic batching, one endpoint, metrics.
12. **B** — NGC is the catalog for GPU-optimized containers, pretrained models, and Helm charts.
13. **B** — AI Enterprise = support, security/stability, certification for production; not extra silicon performance or licensing of CUDA.
14. **A** — NIM = prebuilt inference microservice container with an optimized runtime and OpenAI-compatible API.
15. **B** — Problem → data → train → validate → deploy → monitor (a loop).
16. **B** — Monitoring in production detects model drift as data distributions shift.
17. **A** — RAG grounds answers in retrieved, up-to-date private documents without retraining.
18. **A** — Batching raises throughput by amortizing overhead, at the cost of per-request latency.
19. **B** — Fraud detection is the canonical finance use case; the others are mismatched.
20. **C** — FP8 + Transformer Engine arrived with Hopper (H100).
21. **A** — DGX is the turnkey NVIDIA system; OEMs build custom servers on the HGX baseboard.
22. **B** — NVSwitch creates an all-to-all fabric so every GPU talks at full NVLink bandwidth simultaneously.
23. **C** — ~900 GB/s (NVLink 4 total) vs ~128 GB/s bidirectional PCIe Gen5 x16 → roughly 7×+.
24. **B** — SXM = higher power envelope + full NVLink/NVSwitch; PCIe cards at best bridge in pairs.
25. **A** — Idle GPUs at step boundaries during multi-node scaling point to gradient all-reduce limited by the inter-node network.
26. **B** — Splitting layers across GPUs is tensor parallelism, which needs NVLink-class bandwidth.
27. **B** — Stacked on-package HBM delivers the multi-TB/s bandwidth needed to feed GPU compute.
28. **A** — 70B × 2 bytes ≈ 140 GB weights > 80 GB → tensor-parallel multi-GPU or quantize.
29. **A** — BF16 keeps FP32's 8-bit exponent, eliminating the overflow that plagues FP16 (which needs loss scaling).
30. **B** — TF32 transparently runs FP32 code on Tensor Cores with FP32-like range.
31. **A** — FP8/INT8 quantization + TensorRT-LLM cuts memory and boosts throughput with a small accuracy-validation cost.
32. **B** — SuperPOD is the validated reference architecture for DGX clusters (fabric + storage + management).
33. **B** — Dedicated max-performance training fabric at scale → Quantum InfiniBand (lossless, RDMA, SHARP).
34. **A** — Spectrum-X is NVIDIA's AI-tuned Ethernet platform (Spectrum switch + BlueField-3 SuperNIC).
35. **B** — SHARP performs in-network reductions in the switches, accelerating all-reduce.
36. **B** — GPUDirect RDMA removes the CPU-memory bounce: the NIC reads/writes GPU memory directly.
37. **B** — DPUs (BlueField) offload and isolate networking/storage/security from host CPUs — key for multi-tenant.
38. **A** — DOCA is the BlueField DPU SDK; the CUDA analogy is the expected framing.
39. **B** — DGX H100 ≈ 10.2 kW each; four ≈ 41 kW, far beyond a 15 kW air-cooled rack.
40. **B** — PUE 1.1 = total facility power only 10% above IT power — very efficient.
41. **A** — Data-loading gaps and checkpoint stalls are storage-throughput problems; a parallel FS (+ GDS) fixes the feed and the write bursts.
42. **B** — Short bursty project + data already in cloud → cloud; 24/7 multi-year + data sovereignty → on-prem TCO and control win.
43. **B** — The device plugin exposes `nvidia.com/gpu` as a schedulable resource requested via limits.
44. **A** — The GPU Operator manages driver, toolkit, device plugin, DCGM exporter, MIG manager as Kubernetes-native components.
45. **B** — Slurm for queued gang-scheduled batch with fair-share; Kubernetes for autoscaled containerized services.
46. **B** — Slurm schedules GPUs via GRES (generic resources).
47. **B** — DCGM (+ exporter → Prometheus, dcgmi diag for active tests) is the fleet-scale answer; nvidia-smi loops don't scale.
48. **B** — Double-bit ECC errors are uncorrectable: drain, diagnose, RMA if persistent. Never disable ECC to hide them.
49. **B** — At-limit temperature + reduced clocks + "busy" GPU = thermal throttling.
50. **B** — Guaranteed isolation → MIG; best-effort dev sharing → time-slicing; GPUs in VMs/VMware → vGPU.
