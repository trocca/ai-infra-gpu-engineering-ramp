# Week 2 Self-Check — GPUs and GPU Systems

Target: ≥ 80% (14/18) before moving to week 3.

**1. Match the feature to the architecture: (a) MIG first introduced, (b) FP8 + Transformer Engine, (c) FP4 + dual-die design.**

<details><summary>Answer</summary>
(a) Ampere (A100), (b) Hopper (H100), (c) Blackwell (B200).
</details>

**2. What is TF32 and why did NVIDIA introduce it?**

<details><summary>Answer</summary>
TF32 is a Tensor Core math mode introduced with Ampere: it keeps FP32's 8-bit exponent (same dynamic range) but reduces the mantissa to 10 bits. It lets existing FP32 training code get Tensor Core speedups automatically, with no code changes and rarely any accuracy loss.
</details>

**3. A customer asks the difference between buying a DGX system and an HGX-based server from Dell. What do you tell them?**

<details><summary>Answer</summary>
DGX is NVIDIA's fully integrated, turnkey system: NVIDIA hardware design, preinstalled software stack (DGX OS, Base Command), single-vendor NVIDIA support — fastest path to production. HGX is the 8-GPU baseboard (GPUs + NVSwitch) that OEMs like Dell build their own servers around — more vendor choice and configuration flexibility, with support from the OEM. Same GPU silicon either way.
</details>

**4. What is the difference between NVLink and NVSwitch?**

<details><summary>Answer</summary>
NVLink is the high-speed point-to-point GPU interconnect (900 GB/s total per H100). NVSwitch is a switch chip that connects all GPUs in a system through a fabric so every GPU can talk to every other GPU at full NVLink bandwidth simultaneously — without it, 8 GPUs would have to split their links point-to-point.
</details>

**5. Rank by bandwidth, lowest to highest: NVLink 4 (per H100), PCIe Gen5 x16, NVLink 5 (per Blackwell GPU).**

<details><summary>Answer</summary>
PCIe Gen5 x16 (~64 GB/s per direction, ~128 GB/s bidirectional) < NVLink 4 (900 GB/s total per H100) < NVLink 5 (1.8 TB/s total per Blackwell GPU).
</details>

**6. Why does multi-GPU data-parallel training put pressure on the GPU interconnect?**

<details><summary>Answer</summary>
In data parallelism every GPU computes gradients on its own data shard, and after each step gradients must be synchronized across all GPUs (all-reduce, via NCCL). Gradient size ≈ model size, exchanged every iteration — slow interconnect leaves GPUs idle waiting on communication, killing scaling efficiency.
</details>

**7. Distinguish data parallelism, tensor parallelism, and pipeline parallelism in one line each.**

<details><summary>Answer</summary>
Data parallelism: full model copy per GPU, split the data, sync gradients. Tensor (model) parallelism: split individual layers' matrices across GPUs — needs the fastest interconnect (usually within a node/NVLink). Pipeline parallelism: split the model by layers into stages, micro-batches flow through stages on different GPUs.
</details>

**8. What is HBM and why does it matter for AI?**

<details><summary>Answer</summary>
High Bandwidth Memory — DRAM dies stacked and placed on the GPU package, connected by very wide interfaces, delivering multi-TB/s bandwidth (H100 ~3.35 TB/s, H200 ~4.8 TB/s). DL performance is often limited by how fast data/weights can be fed to the compute units, so bandwidth is as important as capacity — especially for memory-bound workloads like LLM inference.
</details>

**9. Roughly how much GPU memory do the weights alone of a 70B-parameter model need at FP16, and what does that imply?**

<details><summary>Answer</summary>
~2 bytes/parameter → ~140 GB at FP16. That exceeds any single GPU (H100 = 80 GB, H200 = 141 GB), implying multi-GPU serving (tensor parallelism) or quantization (e.g. FP8/INT4) to fit on fewer GPUs.
</details>

**10. Why is BF16 generally preferred over FP16 for training?**

<details><summary>Answer</summary>
BF16 keeps FP32's 8-bit exponent, so it has the same dynamic range — gradients don't overflow/underflow, no loss scaling needed. FP16 has a 5-bit exponent (limited range) and typically requires loss scaling to train stably. Both are 16 bits, so memory/speed benefits are similar.
</details>

**11. What is the Transformer Engine?**

<details><summary>Answer</summary>
Hopper (and later) feature combining FP8 Tensor Cores with software that dynamically chooses precision and scaling per layer during training/inference of transformer models — getting FP8 speed while maintaining accuracy.
</details>

**12. A customer wants maximum GPU-to-GPU bandwidth inside a server. SXM or PCIe GPUs, and why?**

<details><summary>Answer</summary>
SXM. SXM modules mount on an HGX baseboard with NVLink/NVSwitch connectivity between all GPUs (and higher power limits, e.g. 700 W H100 SXM vs 350–400 W PCIe). PCIe cards communicate over the PCIe bus (or NVLink bridges pairing only two cards at best).
</details>

**13. What is a DGX SuperPOD?**

<details><summary>Answer</summary>
NVIDIA's reference architecture for AI infrastructure at scale: tens to hundreds+ of DGX nodes with an InfiniBand (or Spectrum-X Ethernet) fabric, high-performance storage, and management software (Base Command), engineered as a validated, repeatable cluster design. BasePOD is the smaller-scale version.
</details>

**14. What are Tensor Cores, and which architecture introduced them?**

<details><summary>Answer</summary>
Specialized GPU units that execute matrix multiply-accumulate operations on small tiles in one operation, in mixed/reduced precision — the workhorse of DL math. Introduced with Volta (V100); each generation since added formats (Ampere: TF32 + sparsity, Hopper: FP8, Blackwell: FP4).
</details>

**15. What is GH200 / what is the Grace Hopper superchip?**

<details><summary>Answer</summary>
A module combining NVIDIA's Grace Arm CPU with a Hopper GPU, connected by NVLink-C2C — a coherent 900 GB/s chip-to-chip link that lets the GPU access CPU memory at far higher bandwidth than PCIe. Aimed at workloads with huge memory footprints. GB200 is the Blackwell-generation equivalent (Grace + 2 Blackwell GPUs).
</details>

**16. Why doesn't training speed scale perfectly from 1 to 8 GPUs?**

<details><summary>Answer</summary>
Communication overhead (gradient synchronization), serial portions of the pipeline (data loading, logging), and possible load imbalance eat into scaling (Amdahl's law). Good interconnects (NVLink/NVSwitch), NCCL's overlap of compute and communication, and larger batch sizes improve but never fully eliminate the gap.
</details>

**17. Which precision formats would you expect in (a) LLM pretraining on H100s, (b) production LLM inference on H100s, (c) legacy scientific computing?**

<details><summary>Answer</summary>
(a) BF16 mixed precision, increasingly with FP8 via Transformer Engine. (b) FP8 or INT8/INT4 quantized (TensorRT-LLM), FP16 as the conservative default. (c) FP64 — HPC needs full double precision, a key differentiator of datacenter GPUs.
</details>

**18. What claim does 2:4 structured sparsity make?**

<details><summary>Answer</summary>
If a model is pruned so that in every group of 4 weights at least 2 are zero, Ampere+ Tensor Cores can skip the zeros and roughly double math throughput for those layers, with (ideally) minimal accuracy loss after fine-tuning.
</details>
