# Week 7 Plan (Aug 24–28, 2026) — Model Optimization + GPU Acceleration

[← Master Plan](../../../MASTER-PLAN.md)

**Days:** [1](day-1.md) · [2](day-2.md) · [3](day-3.md) · [4](day-4.md) · [5](day-5.md)

Exam coverage this week: **Model Optimization 17% + GPU Acceleration & Optimization 14% = 31%** — the two heaviest domains. Highest-leverage week; protect these five days.

## Prerequisites before Monday

- Companion lesson: [Week 07 companion — quantization, Triton tiling, and FlashAttention support](../../../companion-lessons/week-07.md).
- Math support: affine quantization, memory estimates for FP16/INT8/INT4, softmax stability, and FlashAttention as an IO reduction.
- Programming support: Triton program IDs, block pointers, masks, PyTorch reference tests, and error-vs-speed reporting.
- Gate: estimate memory for an 8B model in FP16, INT8, and INT4, then name the tests that protect custom kernels.

---

## Day 1 (Mon) — Quantization theory
**Domain: Model Optimization (17%)**

- (35 min) Fundamentals: quantization maps FP weights/activations to low-bit representations with scale (and zero-point) factors. Distinguish **weight-only** (W8A16/W4A16 — helps memory + bandwidth, decode gets faster) vs **weight+activation** (W8A8 INT8/FP8 — also uses low-precision Tensor Core math, helps compute-bound prefill). **PTQ** (calibrate on a few hundred samples, no training) vs **QAT** (train with fake-quant; better at very low bits, costs a training run).
- (35 min) Formats: **INT8** — LLM.int8() outlier problem in activations; **SmoothQuant** migrates activation outliers into weights to make W8A8 viable. **FP8** (E4M3 for weights/activations, E5M2 for gradients) — Hopper/Ada Tensor Core native; near-lossless for inference, also usable in training. **FP4/NVFP4** — Blackwell-native 4-bit floating point with per-block scaling. Know the hardware mapping: FP8 → Hopper/Ada+, NVFP4 → Blackwell.
- (30 min) Memory math drill: 8B model at FP16 = 16 GB, INT8 = 8 GB, INT4 ≈ 4–4.5 GB (plus scales). KV cache quantization (FP8 KV) as a separate lever. Rule of thumb: decode throughput scales roughly with bytes moved → 4-bit ≈ 2–3× decode speedup on bandwidth-bound workloads.
- (20 min) NVIDIA tooling: **TensorRT Model Optimizer** (PTQ/QAT for FP8/INT8/INT4/NVFP4, exports to TensorRT-LLM), quantization in vLLM (loads AWQ/GPTQ/FP8 checkpoints). Flashcards.

## Day 2 (Tue) — AWQ/GPTQ, pruning, distillation, speculative decoding
**Domain: Model Optimization (17%)**

- (30 min) **GPTQ**: layer-by-layer weight quantization minimizing output error using second-order (Hessian-based) information; 3–4 bit weight-only; needs calibration data. **AWQ**: observes that ~1% of weight channels are "salient" based on *activation* magnitudes; protects them by per-channel scaling before 4-bit quantization; no backprop, robust, fast inference kernels. Both are PTQ weight-only — know the one-line difference (GPTQ = error-compensating rounding; AWQ = activation-aware scaling).
- (30 min) **Pruning**: unstructured (individual weights → sparse, needs special kernels) vs structured (drop heads/layers/channels → real speedups on GPUs); **2:4 semi-structured sparsity** — Ampere+ Sparse Tensor Cores give up to 2× math throughput; requires fine-tuning to recover accuracy. **Distillation**: student trained on teacher's soft logits (temperature-scaled KL) ± intermediate states; NVIDIA's **Minitron** approach: prune a large model (depth/width) then distill to recover — how Llama-3.1-Minitron / Nemotron small models were made.
- (40 min) **Speculative decoding**: small draft model proposes k tokens, target model verifies them in ONE parallel forward pass; rejection sampling keeps the output distribution *exactly* the target model's. Speedup depends on acceptance rate (draft/target alignment) and only helps memory-bound decode. Variants: draft-model, **Medusa** (extra decoding heads), **EAGLE** (feature-level drafting). Also today: know distinct axes — quantization (precision), pruning (parameters), distillation (capacity), speculative decoding (algorithm) — the exam loves "which technique for which constraint."
- (20 min) Flashcards + scenario drill: latency-bound chat vs throughput-bound batch summarization vs 24 GB memory ceiling — pick techniques for each.

## Day 3 (Wed) — GPU acceleration fundamentals
**Domain: GPU Acceleration & Optimization (14%)**

- (35 min) **Tensor Cores**: matrix-multiply-accumulate units; peak throughput doubles as precision halves (FP16 → FP8 → FP4 on Blackwell). Mixed-precision training: BF16 compute + FP32 accumulate; why BF16 beats FP16 for training (FP32-range exponent, no loss scaling). **Arithmetic intensity & roofline**: FLOPs/byte determines compute-bound vs memory-bound; GEMMs with large batch = compute-bound; decode GEMV (batch 1) = memory-bound → the whole inference-optimization story follows from this.
- (35 min) **Prefill vs decode** (now in depth): prefill = parallel over prompt tokens, compute-bound, Tensor-Core-heavy; decode = sequential, one token/step, re-reads all weights + KV cache → bandwidth-bound. **Kernel fusion**: merging ops (e.g., bias+GeLU, attention ops) to avoid HBM round-trips — the point is reducing memory traffic, not FLOPs. **FlashAttention**: exact attention computed in tiles inside SRAM with online softmax; avoids materializing the N×N score matrix → memory O(N) not O(N²), big speedups at long context. **CUDA graphs**: capture kernel-launch sequences to eliminate per-step launch overhead in decode loops.
- (30 min) Hands-on: on any GPU (or Colab), run a quick roofline intuition check — time `torch.matmul` at batch 1 vs batch 64 for a 4096×4096 layer; observe near-identical latency (memory-bound at low batch). Note result in notes.md.
- (20 min) Flashcards.

## Day 4 (Thu) — Serving-engine mechanics + TensorRT-LLM
**Domain: GPU Acceleration & Optimization (14%), feeds Deployment (9%)**

- (35 min) **Continuous (in-flight) batching**: scheduler admits/evicts sequences every iteration instead of waiting for the whole static batch to finish → GPU stays full despite variable output lengths; this is THE serving-throughput idea (vLLM, TensorRT-LLM in-flight batching, Triton). **Paged attention**: KV cache in fixed-size blocks with an indirection table (virtual memory analogy) → kills fragmentation, enables high concurrency and **prefix caching** (shared system-prompt KV blocks reused across requests).
- (30 min) **Chunked prefill** (split long prompts into chunks interleaved with decode steps → protects ITL/TTFT of concurrent requests), speculative decoding in serving, KV-cache offload. Know knobs conceptually: max batch/num-seqs, KV block size, gpu-memory-utilization.
- (35 min) **TensorRT-LLM**: ahead-of-time compiled engines (fused kernels, precision baked in: FP8/INT4/NVFP4), in-flight batching, TP/PP support; engines are GPU-arch-specific — build per target. Trade-off vs vLLM: TRT-LLM = peak performance on NVIDIA GPUs + build step + less flexibility; vLLM = flexibility, model-day-one support, easy ops. NIM packages TRT-LLM or vLLM behind one API (week 8). Cross-ref demo repo: this is the layer under your vLLM/Dynamo/NIM demo.
- (20 min) Flashcards + skim `labs/lab-quantize-serve.md` (runs week 8 day 1).

## Day 5 (Fri) — Distributed training refresher + DDP lab
**Domain: GPU Acceleration & Optimization (14%) — exam explicitly assumes DDP/FSDP/Megatron competency**

You've built demos on all of this — today is consolidation into exam-answer form.

- (30 min) Parallelism taxonomy, one paragraph each: **DDP** (replicate model, all-reduce gradients; model must fit on one GPU), **ZeRO/FSDP** (shard optimizer states/grads/params; all-gather on demand, reduce-scatter grads; fits big models, more communication), **Tensor parallel** (split individual matmuls; all-reduce per layer → needs NVLink-class bandwidth, stay intra-node), **Pipeline parallel** (split layers into stages; microbatches to shrink the bubble; inter-node friendly), **Megatron-style 3D** = TP intra-node × PP inter-node × DP across replicas. Sequence/context parallel for long-context. Map each to its dominant collective: DDP→all-reduce, FSDP→all-gather+reduce-scatter, TP→all-reduce (per layer), PP→point-to-point send/recv.
- (20 min) NCCL refresher from your demo repo: transport selection (NVLink/P2P > SHM > NET Socket/IB), ring vs tree algorithms, NCCL_DEBUG=INFO reading. Rehearse: "how does NCCL pick a transport and how do I verify it?" — that's both an exam answer and a demo-booth answer.
- (~60 min) Run `labs/lab-distributed-ddp.md` on a 2-GPU instance. Deliverables: env-var table filled in, NCCL transport line pasted into notes.
- (10 min) `week-7/self-check.md` + exit criteria.

## Exit criteria (Friday)

- [ ] I can explain weight-only vs weight+activation quantization and which serving phase each accelerates
- [ ] I can map INT8/FP8/NVFP4 to GPU generations and name the E4M3/E5M2 split
- [ ] I can give the one-line difference between GPTQ and AWQ, and PTQ vs QAT
- [ ] I can explain 2:4 sparsity, Minitron prune-then-distill, and speculative decoding's correctness guarantee
- [ ] I can pick optimization techniques for latency / throughput / memory-ceiling scenarios
- [ ] I can explain why decode is memory-bound and derive why quantization ≈ decode speedup from that
- [ ] I can explain FlashAttention (tiling + online softmax, exact, O(N) memory) in 3 sentences
- [ ] I can explain continuous batching and paged attention and what problem each kills
- [ ] I can compare vLLM vs TensorRT-LLM as a deployment recommendation
- [ ] I can contrast DDP / FSDP / TP / PP with their collectives and placement (intra vs inter-node)
- [ ] Lab complete: env vars documented, NCCL transport identified from logs
- [ ] Self-check score ≥ 80%
