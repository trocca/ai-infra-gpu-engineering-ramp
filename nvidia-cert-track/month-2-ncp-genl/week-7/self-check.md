# Week 7 Self-Check — Model Optimization + GPU Acceleration

**1. Weight-only INT4 quantization speeds up decode but barely helps prefill. Why?**

<details><summary>Answer</summary>
Decode is memory-bandwidth-bound: each step streams all weights, so shrinking weights 4× cuts bytes moved ≈ proportional speedup. Prefill is compute-bound and W4A16 still computes in FP16 (weights are dequantized before the matmul), so FLOPs don't drop. To speed up compute-bound phases you need low-precision *math* — W8A8 INT8 or FP8 on Tensor Cores.
</details>

**2. What problem does SmoothQuant solve and how?**

<details><summary>Answer</summary>
Activation outliers: a few channels in LLM activations have huge magnitudes, wrecking per-tensor INT8 activation quantization. SmoothQuant migrates the difficulty from activations to weights via a per-channel scaling factor (divide activations, multiply weights), making both smooth enough for W8A8 INT8.
</details>

**3. FP8: what are E4M3 and E5M2 and when is each used?**

<details><summary>Answer</summary>
Two FP8 encodings: E4M3 (4 exponent bits, 3 mantissa) — more precision, less range → weights and activations, the default for inference. E5M2 (5 exponent, 2 mantissa) — more range → gradients in FP8 training. Hardware: FP8 Tensor Cores from Hopper/Ada onward; NVFP4 adds 4-bit FP on Blackwell.
</details>

**4. GPTQ vs AWQ in one line each.**

<details><summary>Answer</summary>
GPTQ: quantizes weights layer-by-layer, using approximate second-order (Hessian) information to compensate rounding error on calibration data. AWQ: identifies the ~1% salient weight channels by looking at *activation* magnitudes and rescales them before uniform 4-bit quantization — no gradient-based optimization. Both: post-training, weight-only, need a small calibration set.
</details>

**5. When is QAT worth it over PTQ?**

<details><summary>Answer</summary>
When PTQ accuracy loss is unacceptable — typically at very low bit-widths (4-bit and below, especially weights+activations), for smaller models (less redundancy), or safety-critical accuracy targets. QAT simulates quantization during (re)training so the model learns around it, at the price of a training run and data.
</details>

**6. What is 2:4 structured sparsity and what hardware exploits it?**

<details><summary>Answer</summary>
In every contiguous group of 4 weights, at least 2 are zero. Sparse Tensor Cores (Ampere and later) skip the zeros, giving up to 2× matmul throughput with a compact storage format. Requires pruning to the 2:4 pattern and usually fine-tuning/distillation to recover accuracy.
</details>

**7. Explain why speculative decoding cannot change the model's output distribution.**

<details><summary>Answer</summary>
The draft model proposes k tokens; the target model scores them in one parallel forward pass; each token is accepted with a rejection-sampling probability based on the ratio of target to draft probabilities, and the first rejection is replaced with a sample from an adjusted target distribution. This acceptance rule mathematically guarantees samples are distributed exactly as if drawn from the target model alone — you trade extra compute for fewer sequential steps.
</details>

**8. Your customer has a 24 GB L4 and wants to serve an 8B model with long system prompts to many concurrent users. Name three optimizations and why each fits.**

<details><summary>Answer</summary>
(1) 4-bit weight quantization (AWQ/GPTQ): 16 GB weights → ~4.5 GB, freeing memory for KV cache. (2) Paged attention + prefix caching: fragmentation-free KV and shared system-prompt KV blocks across users. (3) FP8/INT8 KV-cache quantization and/or chunked prefill: more concurrent sequences, protected inter-token latency. (Continuous batching assumed as the baseline.)
</details>

**9. What is the Minitron approach?**

<details><summary>Answer</summary>
NVIDIA's compression recipe: take a trained large model, apply structured pruning (depth — remove layers — and/or width — remove heads/channels), then distill from the original model to recover quality using a small fraction of original training tokens. Produces the small Llama-3.1-Minitron / Nemotron variants far cheaper than training from scratch.
</details>

**10. Why does batch size ~1 matmul latency barely differ from batch 64 on a big linear layer?**

<details><summary>Answer</summary>
At small batch the operation is memory-bound: time is dominated by streaming the weight matrix from HBM, which is identical regardless of batch. Until arithmetic intensity crosses the roofline ridge point, extra batch rows are "free" — the foundation of why batching is the #1 throughput lever in LLM serving.
</details>

**11. What does FlashAttention actually optimize, and is it an approximation?**

<details><summary>Answer</summary>
It is *exact* attention. It optimizes memory traffic: computes attention in tiles held in on-chip SRAM using online (streaming) softmax, never materializing the N×N attention matrix in HBM. Result: O(N) memory instead of O(N²) and large wall-clock speedups at long sequence lengths, because attention is memory-bound.
</details>

**12. Why is BF16 generally preferred over FP16 for LLM training?**

<details><summary>Answer</summary>
BF16 has the same 8-bit exponent as FP32 → same dynamic range, so gradients don't underflow/overflow and loss scaling is unnecessary; it gives up mantissa precision, which training tolerates. FP16's 5-bit exponent forces loss-scaling machinery and still risks instability at scale.
</details>

**13. Static batching vs continuous batching — what exactly does continuous batching fix?**

<details><summary>Answer</summary>
Static batching waits for every sequence in the batch to finish; short outputs sit idle while the longest generates, wasting GPU and delaying queued requests. Continuous (in-flight) batching re-schedules at every iteration: finished sequences leave, waiting requests join immediately. Fixes utilization collapse from variable output lengths — typically several-fold throughput gains.
</details>

**14. Explain paged attention via the OS analogy, and the two features it unlocks.**

<details><summary>Answer</summary>
KV cache is allocated in fixed-size blocks with a per-sequence block table mapping logical to physical blocks — virtual memory for KV. Kills internal/external fragmentation from contiguous pre-allocation (which wasted 60–80% of KV memory). Unlocks: (1) much higher concurrency for the same VRAM; (2) sharing — prefix caching / copy-on-write of common prompt blocks across requests.
</details>

**15. A customer asks "vLLM or TensorRT-LLM?" Give the two-sentence pre-sales answer.**

<details><summary>Answer</summary>
TensorRT-LLM compiles model+precision into arch-specific engines with fused kernels and FP8/FP4 support — peak performance and efficiency on NVIDIA GPUs, at the cost of a build step per model/GPU and less day-one model flexibility. vLLM is Python-flexible with immediate new-model support and easy ops; NIM wraps both behind one standard API so you can start with vLLM-style agility and move hot paths to TRT-LLM engines.
</details>

**16. Contrast DDP and FSDP: what is replicated vs sharded, and which collectives does each rely on?**

<details><summary>Answer</summary>
DDP: full model, gradients, and optimizer state replicated on every GPU; gradients synchronized with all-reduce (overlapped with backward). Works only if the model fits on one GPU. FSDP/ZeRO-3: parameters, gradients, and optimizer states *sharded*; all-gather parameters just-in-time per layer for compute, reduce-scatter gradients after backward. Fits far larger models at the cost of more communication volume.
</details>

**17. Why does tensor parallelism want NVLink-class links while pipeline parallelism tolerates Ethernet/IB across nodes?**

<details><summary>Answer</summary>
TP splits individual matmuls and must all-reduce activations *multiple times per layer, every microbatch* — enormous frequency and volume, latency-sensitive → intra-node NVLink. PP only sends activations once per stage boundary per microbatch (point-to-point) — far less traffic → fine over inter-node fabric. Hence Megatron's rule: TP inside the node, PP across nodes, DP across replicas.
</details>

**18. In NCCL_DEBUG=INFO logs, what would tell you two GPUs are communicating over NVLink P2P vs falling back to sockets?**

<details><summary>Answer</summary>
Channel lines like "Channel 00 : 0 -> 1 via P2P/CUMEM" (or P2P/direct, P2P/IPC) indicate NVLink/PCIe peer-to-peer; "via SHM" means shared host memory (no P2P); "via NET/Socket" or "NET/IB" means network transport — Socket implying no RDMA. Your demo-repo NCCL walkthrough shows exactly these lines; the DDP lab reproduces it on 2 GPUs.
</details>

**19. What is the pipeline "bubble" and how do you shrink it?**

<details><summary>Answer</summary>
Idle time while pipeline stages wait for the first microbatches to flow through (and drain at the end). Bubble fraction ≈ (stages − 1) / (microbatches + stages − 1), so you shrink it by increasing the number of microbatches per global batch, and with interleaved/1F1B schedules that overlap forward and backward.
</details>

**20. Which single optimization would you reach for first: a chat product whose complaint is "responses feel slow to start," vs a batch pipeline whose complaint is "GPU bill too high"?**

<details><summary>Answer</summary>
Slow to start = TTFT problem → attack prefill: chunked-prefill tuning/prioritization, prefix caching of the system prompt, shorter prompts, possibly FP8 prefill compute. GPU bill for batch = throughput/$ problem → maximize batching (continuous batching, high concurrency), quantize weights to 4-bit, consider cheaper GPUs at high utilization. Matching the metric (TTFT vs throughput) to the phase (prefill vs decode) is the exam's favorite trick.
</details>
