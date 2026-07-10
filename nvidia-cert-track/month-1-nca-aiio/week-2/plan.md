# Week 2 Plan — AI Infrastructure, part 1: GPUs and GPU systems (Domain 2, 40%)

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Dates: Mon 2026-07-20 → Fri 2026-07-24 · 5 days × ~2 h

Domain 2 is the biggest domain (40%, ~20 of 50 questions), split across weeks 2 and 3. This week: the compute — GPU architectures, DGX/HGX systems, NVLink/NVSwitch, memory, precision formats. Also start daily flashcards (10–15 min, Domain 1 + this week's cards).

## Prerequisites before Monday

- Companion lesson: [Week 02 companion — GPU hardware math, memory bandwidth, and CUDA/Rust basics](../../../companion-lessons/week-02.md).
- Math support: bytes-moved estimates, bandwidth lower bounds, and precision-format trade-offs.
- Programming support: Rust `Result`, CUDA grid/block/thread vocabulary, global vs shared memory, and timed GPU benchmarking.
- Gate: estimate bytes moved by SAXPY and identify the first week-2 toolchain proof command before Day 1.

---

## Day 1 (Mon) — Datacenter GPU generations
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards (Domain 1).
- (60 min) High-level tour of datacenter GPU architectures into `notes.md`. For each, one line on what it introduced:
  - **Ampere (A100, 2020)**: TF32, MIG (up to 7 instances), 3rd-gen NVLink, 40/80 GB HBM2e
  - **Hopper (H100/H200, 2022)**: 4th-gen Tensor Cores with FP8, Transformer Engine, NVLink 4, H200 = 141 GB HBM3e
  - **Blackwell (B200/GB200, 2024)**: dual-die design, FP4/FP6 support, NVLink 5, GB200 = Grace CPU + 2× Blackwell superchip
  - Also know: **Grace CPU** (NVIDIA Arm datacenter CPU), **Grace Hopper / GH200** (CPU+GPU superchip, NVLink-C2C coherent link)
- (30 min) Read the NVIDIA H100 or Blackwell architecture overview page/whitepaper intro (developer.nvidia.com). Depth needed: recognize which generation a feature belongs to — not SM counts.
- (15 min) Compare specs: look up A100 vs H100 vs B200 datasheet tables (memory, bandwidth, NVLink speed).

## Day 2 (Tue) — DGX vs HGX, SuperPOD
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (55 min) Systems into `notes.md`:
  - **DGX**: NVIDIA's fully-integrated, NVIDIA-branded AI server/appliance (e.g. DGX H100: 8× H100 SXM, NVSwitch, ConnectX NICs, preinstalled DGX software stack — "AI supercomputer in a box")
  - **HGX**: the GPU baseboard/platform (8× SXM GPUs + NVSwitch) that OEMs (Dell, Supermicro, Lenovo…) build their own servers around
  - **DGX ≠ HGX exam angle**: DGX = turnkey NVIDIA system + software + support; HGX = OEM building block for custom/vendor-choice designs
  - **MGX**: modular reference architecture for OEMs (broader configs, incl. Grace)
  - **DGX BasePOD / SuperPOD**: reference architectures for scaling DGX nodes into clusters (compute + InfiniBand fabric + storage + management, e.g. Base Command)
- (35 min) Read the DGX platform page + a DGX SuperPOD reference architecture overview (skim the RA PDF's architecture diagrams).
- (15 min) SXM vs PCIe form factors: SXM = higher power + NVLink to all peers via NVSwitch; PCIe = standard slots, lower power, NVLink bridge only pairs (if at all).

## Day 3 (Wed) — NVLink, NVSwitch, multi-GPU scaling
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (50 min) Interconnect ladder into `notes.md`, with rough bandwidth ordering:
  - PCIe (Gen5 x16 ≈ 64 GB/s eff. per direction) ≪ NVLink (H100: 900 GB/s total per GPU; Blackwell NVLink5: 1.8 TB/s)
  - **NVLink**: direct GPU-to-GPU link. **NVSwitch**: switch chip letting all 8 GPUs in a node talk at full NVLink speed simultaneously (all-to-all)
  - **NVLink Switch System / NV domains**: extending NVLink beyond one node (GB200 NVL72 = 72 GPUs in one NVLink domain)
- (40 min) Multi-GPU scaling concepts: why interconnect bandwidth matters (gradient all-reduce in data parallelism), data vs model/tensor/pipeline parallelism (one line each — depth comes in your DDP/FSDP work anyway), strong vs weak scaling, and why 8 GPUs ≠ 8× speedup (communication overhead).
- (15 min) Skim an NVIDIA blog post on NCCL/all-reduce to connect NVLink hardware to the software you already know.

## Day 4 (Thu) — GPU memory and precision formats
**Domain served: AI Infrastructure (40%) — plus Domain 1 crossover (training vs inference precision)**

- (15 min) Flashcards.
- (45 min) Memory into `notes.md`:
  - **HBM** (High Bandwidth Memory): stacked DRAM on-package → multi-TB/s bandwidth (H100: ~3.35 TB/s; H200 HBM3e: ~4.8 TB/s); vs GDDR (consumer) and system DDR
  - Why memory bandwidth is often the bottleneck (LLM inference is memory-bound); why model size vs GPU memory drives GPU count (rule of thumb: FP16 weights = 2 bytes/param)
- (45 min) Precision formats table in `notes.md`: FP64, FP32, **TF32** (Ampere+: FP32 range, ~10-bit mantissa, transparent speedup), FP16 (needs loss scaling), **BF16** (FP32 range, less precision, no loss scaling — preferred for training), **FP8** (Hopper+, Transformer Engine manages scaling), INT8 (inference quantization), FP4 (Blackwell inference). Know: fewer bits → more throughput + less memory, and which generation introduced which.
- (15 min) Read NVIDIA blog "TF32" or "FP8/Transformer Engine" post.

## Day 5 (Fri) — Tensor Cores deep-ish dive + review
**Domain served: AI Infrastructure (40%)**

- (15 min) Flashcards.
- (30 min) Tensor Cores: generation recap (Volta introduced them; Ampere added TF32 + sparsity; Hopper added FP8 + Transformer Engine; Blackwell added FP4). What structured sparsity (2:4) claims to do.
- (20 min) Tie-together exercise: sketch (on paper) a DGX H100 node — 8 GPUs, NVSwitch, NICs, CPUs — and trace what happens in an all-reduce.
- (45 min) `self-check.md` closed-notes; restudy misses.
- (10 min) Skim week 3 plan.

---

## Exit criteria — check every box before starting week 3

- [ ] I can name Ampere, Hopper, and Blackwell with one signature feature each (MIG/TF32; FP8/Transformer Engine; FP4/dual-die + NVLink 5)
- [ ] I can explain DGX vs HGX vs MGX in one sentence each, and what a BasePOD/SuperPOD is
- [ ] I can explain NVLink vs NVSwitch vs PCIe and rank their bandwidth
- [ ] I can explain why interconnect matters for multi-GPU training (all-reduce) and name data/tensor/pipeline parallelism
- [ ] I can explain what HBM is and why memory bandwidth (not just capacity) matters
- [ ] I can order FP32/TF32/BF16/FP16/FP8/INT8/FP4 by size and say what each is used for and which GPU generation introduced TF32, FP8, FP4
- [ ] I can explain SXM vs PCIe GPU form factors
- [ ] I scored ≥ 80% on the week 2 self-check
