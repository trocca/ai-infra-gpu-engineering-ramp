# Week 1 Plan — Essential AI Knowledge (Domain 1, 38%)

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Dates: Mon 2026-07-13 → Fri 2026-07-17 · 5 days × ~2 h

This week covers the single heaviest exam domain (38%, ~19 of 50 questions). It is also the vocabulary layer for everything in weeks 2–4 — get these concepts crisp and the infrastructure material becomes obvious.

## Prerequisites before Monday

- Companion lesson: [Week 01 companion — AI math notation, backprop, and first Rust contact](../../../companion-lessons/week-01.md).
- Math support: scalar/vector/matrix notation, the chain rule, and the gradient-descent update `theta <- theta - eta * grad L`.
- Programming support: Python environment, `pytest`, basic tensor shapes, and Rust vocabulary: ownership, borrow, `Result`, and `Cargo`.
- Gate: finish the companion lesson's 25-minute drill. If it takes more than 45 minutes, reserve a repair block before Day 1.

---

## Day 1 (Mon) — AI / ML / DL fundamentals + common use cases
**Domain served: Essential AI Knowledge (38%)**

- (30 min) Enroll in and start the free NVIDIA DLI course **"AI Infrastructure and Operations Fundamentals"** (via nvidia.com/en-us/training/). Complete the intro + "Introduction to AI" modules.
- (45 min) Nail the hierarchy: AI ⊃ ML ⊃ DL. Supervised vs unsupervised vs reinforcement learning. What a neural network is (layers, weights, activation), what "training a model" actually means (forward pass, loss, backprop, gradient descent, epochs, batches).
- (30 min) Generative AI and LLMs at exam depth: transformer (attention), foundation models, fine-tuning vs prompting vs RAG (one sentence each).
- (15 min) Common use cases by industry — read NVIDIA's industry solution pages (healthcare imaging, fraud detection in finance, recommenders in retail, autonomous vehicles, conversational AI). The exam asks "which is an example of X" style questions.
- Fill in `notes.md` section 1 as you go.

## Day 2 (Tue) — Training vs inference
**Domain served: Essential AI Knowledge (38%) — with direct payoff in Domain 2 sizing questions**

- (40 min) DLI course: training vs inference module.
- (50 min) Build the comparison table in `notes.md`: compute pattern (throughput-heavy long-running batch vs latency-sensitive request/response), precision typically used (FP32/TF32/BF16 mixed for training; FP16/FP8/INT8 for inference), memory pressure (weights + gradients + optimizer states + activations vs weights + KV cache), hardware implications (big multi-GPU clusters + fast interconnect vs scale-out, sometimes MIG-partitioned or edge).
- (30 min) Read NVIDIA Technical Blog posts on inference optimization (TensorRT overview post, "what is batching for inference"). Understand throughput vs latency trade-off and why batch size matters.
- Payoff to your DE role: this is exactly the framing behind vLLM/NIM sizing conversations.

## Day 3 (Wed) — GPU vs CPU architecture
**Domain served: Essential AI Knowledge (38%)**

- (40 min) DLI course: GPU architecture module.
- (50 min) Core mental model into `notes.md`: CPU = few powerful cores, big caches, branch prediction, optimized for serial/latency-sensitive work. GPU = thousands of small cores (CUDA cores) organized in SMs, optimized for throughput on data-parallel work; hides memory latency with massive threading. Memory bandwidth as the headline number (HBM vs DDR — order of magnitude difference).
- (30 min) Why matrix multiply dominates DL, and why that maps to GPUs. What a Tensor Core is (matrix-multiply unit, mixed precision) vs a CUDA core. When CPUs still win (preprocessing, serial logic, small models, orchestration).

## Day 4 (Thu) — CUDA ecosystem + NVIDIA software stack
**Domain served: Essential AI Knowledge (38%)**

- (30 min) DLI course: NVIDIA software stack module.
- (60 min) Build the stack map in `notes.md`, bottom-up, one sentence per layer:
  - **Driver** → **CUDA Toolkit** (compiler, runtime, libraries) → **CUDA-X libraries**: cuDNN (DL primitives), cuBLAS (linear algebra), NCCL (multi-GPU/multi-node collectives), DALI (data loading), RAPIDS (GPU data science: cuDF/cuML)
  - **Frameworks**: PyTorch, TensorFlow (sit on cuDNN/cuBLAS/NCCL)
  - **Inference**: TensorRT (optimizer/runtime: layer fusion, precision calibration), TensorRT-LLM, Triton Inference Server (multi-framework model serving), NIM (containerized inference microservices)
  - **NGC**: catalog of GPU-optimized containers, pretrained models, Helm charts
  - **NVIDIA AI Enterprise**: the supported, licensed distribution of this stack for production (support, security patches, certified on mainstream servers/clouds, includes NIM)
- (30 min) Browse catalog.ngc.nvidia.com for 20 minutes — find the PyTorch container, a pretrained model, a Helm chart. Exam questions about NGC become trivial once you've clicked around it.

## Day 5 (Fri) — AI development lifecycle + week review
**Domain served: Essential AI Knowledge (38%)**

- (40 min) AI development lifecycle end-to-end in `notes.md`: business problem → data collection/preparation (labeling, cleaning, augmentation — often 80% of the work) → model selection/training → evaluation/validation → optimization for deployment (quantization, TensorRT) → deployment (Triton/NIM) → monitoring & retraining (drift). Map NVIDIA tools to each stage.
- (20 min) Where MLOps fits; why "deploy" is not the end (model drift, retraining loops).
- (45 min) Do `self-check.md` for this week, closed-notes. Re-study anything you miss.
- (15 min) Skim week 2 plan; start flashcards (Domain 1 cards) if time allows.

---

## Exit criteria — check every box before starting week 2

- [ ] I can explain AI vs ML vs DL and supervised vs unsupervised vs reinforcement learning in under a minute
- [ ] I can describe what happens during training (forward pass, loss, backprop, gradient descent) without notes
- [ ] I can contrast training vs inference on four axes: compute pattern, precision, memory needs, and infrastructure implications
- [ ] I can explain why GPUs beat CPUs for deep learning (parallelism, memory bandwidth, Tensor Cores) and name a case where the CPU is the right tool
- [ ] I can place each of these in the stack and give its one-line job: driver, CUDA, cuDNN, cuBLAS, NCCL, TensorRT, Triton, NIM, NGC, NVIDIA AI Enterprise
- [ ] I can walk the AI development lifecycle stage by stage and name an NVIDIA tool for each stage
- [ ] I scored ≥ 80% on the week 1 self-check
