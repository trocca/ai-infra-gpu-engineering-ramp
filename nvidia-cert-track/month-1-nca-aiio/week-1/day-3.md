# Week 1 · Day 3 — GPU vs CPU architecture

[← Master Plan](../../../MASTER-PLAN.md) · [Week 1 overview](plan.md) · [← previous day](day-2.md) · [next day →](day-4.md)

## Study block (2 h)

DLI course GPU-architecture module first (~40 min), then internalize the mental model below into `notes.md`. This is the "why NVIDIA exists" day — everything in week 2 is a refinement of today.

### The core mental model: latency machine vs throughput machine

A **CPU** is a *latency* machine: a handful of very powerful cores (dozens in servers), each with deep pipelines, sophisticated **branch prediction**, out-of-order execution, and a large multi-level **cache** hierarchy — all silicon spent making *one thread finish as fast as possible*. It excels at serial logic, branchy code, and unpredictable memory access.

A **GPU** is a *throughput* machine: thousands of small, simple cores (**CUDA cores**) organized into **Streaming Multiprocessors (SMs)** — an H100 has 132 SMs and over 16,000 CUDA cores. Each core is individually slow, but the design assumption is that you have *tens of thousands of independent, identical operations* to run at once (data parallelism). Instead of avoiding memory latency with big caches, the GPU **hides** it with massive multithreading: when one group of threads stalls waiting for memory, the SM instantly switches to another group that's ready. Threads execute in lockstep groups of 32 called **warps** (SIMT — single instruction, multiple threads); you'll feel this personally in week 2 of the lab.

**Where the silicon goes — few big cores + cache vs thousands of small cores + bandwidth:**

```
        CPU — latency machine                 GPU — throughput machine
 ┌────────────────────────────────┐   ┌────────────────────────────────┐
 │  ┌──────────┐   ┌──────────┐   │   │ SM0  SM1  SM2  SM3  ...  SM131 │
 │  │ Big core │   │ Big core │   │   │ ████ ████ ████ ████      ████  │
 │  └──────────┘   └──────────┘   │   │ ████ ████ ████ ████      ████  │
 │  ┌──────────┐   ┌──────────┐   │   │                                │
 │  │ Big core │   │ Big core │   │   │  each SM = many CUDA cores +   │
 │  └──────────┘   └──────────┘   │   │  Tensor Cores; stalled warps   │
 │  branch prediction,            │   │  swapped out to hide latency   │
 │  out-of-order execution        │   ├────────────────────────────────┤
 ├────────────────────────────────┤   │  L2 cache (small vs CPU's)     │
 │  large L1/L2/L3 caches         │   ├────────────────────────────────┤
 ├────────────────────────────────┤   │  HBM  ~3.35 TB/s (H100)        │
 │  DDR5  ~0.5 TB/s (dual-socket) │   └────────────────────────────────┘
 └────────────────────────────────┘
```

### Memory bandwidth is the headline number

Feeding thousands of cores requires enormous memory bandwidth, so datacenter GPUs use **HBM** (high-bandwidth memory, stacked on-package): an H100 delivers ~3.35 TB/s, versus a strong dual-socket CPU with DDR5 at ~0.5 TB/s and typical single-socket setups far less — call it an **order of magnitude** difference and you're exam-safe. Many DL workloads (LLM inference decode especially) are limited by how fast bytes move, not how fast math runs, so bandwidth is often the spec that matters more than TFLOPS. Full treatment on week 2 Day 4.

### Why matrix multiply is the whole game

Every dense neural-network layer is `output = activation(input × weights + bias)` — a **matrix multiplication**. Convolutions reduce to matmuls; the transformer's attention is matmuls. Matmul is the ideal GPU workload: massively parallel (every output element is independent), regular memory access, and high **arithmetic intensity** (each loaded value is reused many times, so compute can outrun bandwidth). Deep learning is, computationally, the art of keeping matmul units fed.

### CUDA cores vs Tensor Cores — a distinction the exam loves

- **CUDA core**: general-purpose scalar unit; one FP32 multiply-add per clock. Runs *any* parallel code — physics, graphics, data processing.
- **Tensor Core**: specialized matrix-multiply-accumulate unit that computes a small matrix operation (e.g., 4×4 tiles) *per clock*, in **mixed precision** (multiply in FP16/BF16/TF32/FP8, accumulate in FP32). Introduced with Volta (2017); an order of magnitude more matmul throughput than CUDA cores at the same power.
- Trap: Tensor Cores don't replace CUDA cores — a GPU has both; frameworks route matmul/conv to Tensor Cores (via cuDNN/cuBLAS) and everything else (elementwise ops, normalizations, data movement) to CUDA cores.

### When the CPU is still the right tool

Don't be a GPU maximalist in front of a customer — or on the exam. CPUs win for: **serial, branchy logic** (business rules, orchestration, web serving); **data preprocessing/ETL glue** (though RAPIDS moves much of it to GPU when data is large); **small models with tiny request rates** where a GPU would idle; **latency-critical single-sample inference of small classical models** (a decision tree doesn't need an H100); and everything around the AI system — schedulers, databases, APIs. The honest framing: an AI system is CPU-orchestrated with GPU-accelerated hot spots, which is exactly why Grace (NVIDIA's CPU) exists.

### Pre-sales angle

- "Can't we just use our existing CPU farm for training?" → a single H100 delivers matmul throughput that would take hundreds of CPU sockets, at a fraction of the power per unit of work; DL training on CPUs is economically irrational at any scale beyond toy.
- "GPU utilization is 100% but training is slow" → utilization ≠ efficiency; the job may be bandwidth-bound or input-pipeline-bound (CPU-side data loading starving the GPU — the case for DALI).
- Sizing tell: ask whether the workload is *data-parallel and dense math* (GPU) or *branchy and serial* (CPU) before quoting anything.

### Read next

- DLI course — GPU architecture module (assigned above).
- NVIDIA Technical Blog: "An Even Easier Introduction to CUDA" — the grid/block/thread mental model (also pre-reading for lab week 2).
- NVIDIA glossary: "CUDA cores vs Tensor Cores" style pages / the Tensor Core product page — one line per generation is enough today.
- Optional: skim an H100 datasheet — find CUDA core count, SM count, memory bandwidth; you'll compare generations in week 2.

### Quick check

1. How does a GPU deal with memory latency, and how does that differ from a CPU's approach?
<details><summary>Answer</summary>The GPU hides latency by oversubscribing each SM with many warps and switching to a ready warp whenever one stalls; the CPU avoids latency using large caches, prefetching, branch prediction, and out-of-order execution to keep one thread fast.</details>

2. What operation dominates deep-learning compute, and name two properties that make it GPU-friendly.
<details><summary>Answer</summary>Matrix multiplication. It's massively data-parallel (independent output elements) and has high arithmetic intensity with regular memory access, so thousands of cores can be kept busy per byte loaded.</details>

3. CUDA core vs Tensor Core in one sentence each.
<details><summary>Answer</summary>CUDA core: general-purpose unit doing one scalar FP multiply-add per clock for any parallel code. Tensor Core: specialized unit doing a whole small matrix multiply-accumulate per clock in mixed precision, giving order-of-magnitude higher matmul throughput.</details>

4. A customer wants to accelerate a rules-based fraud engine (thousands of nested if/else per transaction). GPU or CPU, and why?
<details><summary>Answer</summary>CPU — branchy, serial, divergent logic is what CPUs are built for; warps executing in lockstep make heavily divergent code inefficient on GPUs. (If they *replace* rules with a learned model on large data, the GPU conversation reopens.)</details>

## Build block (4 h)

**Today: `nn`, `optim`, and MNIST — the dense day.** [Project brief](../../../gpu-engineering-lab/01-foundations/week-01-autograd-from-scratch/README.md)

- `src/nn.py`: `Module` base, `Linear` (Kaiming init), `ReLU`, `Sequential`, numerically-stable `softmax_cross_entropy` (max-subtraction + log-sum-exp).
- `src/optim.py`: `SGD` (with momentum) and `Adam`, matching PyTorch's documented update rules exactly — tests lockstep-compare five steps against `torch.optim`.
- `src/train_mnist.py`: data loading (raw IDX or `fetch_openml`, no torchvision), minibatching, train/eval loops; MLP 784→256→128→10, Adam, ~10–15 epochs.
- Definition of done: **≥97.0% MNIST test accuracy**; record max relative gradient error vs PyTorch (≤1e-6 in float64) and seconds/epoch (median of ≥5 epochs after warmup).
- Kick off the training run and start Day 4's Rust reading while it trains.
- Hint: if Adam diverges from PyTorch on step 1, check the bias-correction terms and whether you update `t` before or after using it — the docs' pseudocode ordering matters.

## Close the day (15 min)

- Anki: cards for SM/warp/CUDA core/Tensor Core definitions, the bandwidth order-of-magnitude, and two "CPU wins when…" cases; review prior days.
- One line in [notes.md](notes.md): the hardest thing today.
- Log blockers (accuracy plateau? optimizer mismatch? note it — Friday's benchmark depends on today).
