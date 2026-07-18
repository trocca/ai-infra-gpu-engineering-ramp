# Hugging Face Ultra-Scale Playbook Integration Map

[<- Reference shelf](README.md) · [Master Plan](../MASTER-PLAN.md) · [Companion lessons](../companion-lessons/README.md)

Source: [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook), by the Hugging Face Nanotron team. This guide maps the public playbook, and your local PDF copy, into the 12-week path.

Use this as a weekly source-reading lane. Do not read the book front-to-back during
exam weeks. Read the assigned section, extract one formula or systems constraint into
the week's `notes.md`, then return to the build.

## Why It Belongs Here

The playbook fills a gap between certification vocabulary and real training-infra
judgment. It is strongest for:

- memory accounting for transformer training;
- choosing between recomputation, gradient accumulation, data parallelism, ZeRO, tensor parallelism, context parallelism, pipeline parallelism, and expert parallelism;
- explaining why topology and interconnect speed decide whether a parallelism strategy wins;
- connecting low-level GPU kernels, FlashAttention, mixed precision, and profiling to end-to-end throughput.

<a id="weekly-integration"></a>
## Weekly Integration

| Week | Read | Pull into the week | Math or systems insight |
|------|------|--------------------|-------------------------|
| 5 | Ch. 2, pp. 19-39: one-GPU training, transformer memory, activation recomputation, gradient accumulation | GPT from scratch should include a memory ledger: parameters, gradients, optimizer states, and activations. | Non-activation training memory is roughly `16-20 bytes * parameter_count` with Adam depending on precision/FP32 gradient storage; activation memory grows with batch and sequence length. |
| 6 | Ch. 2.3 and Ch. 3.1-3.3, pp. 35-69: gradient accumulation, global batch size, data parallelism, ZeRO | LoRA/QLoRA should be explained as memory pressure relief: fewer trainable weights, fewer optimizer states, and smaller adapter checkpoints. | `global_batch_tokens = micro_batch * sequence_length * grad_accum_steps * data_parallel_size`. ZeRO shards optimizer state, gradients, then parameters across the DP axis. |
| 7 | Ch. 10, pp. 163-199: GPU primer, kernels, fusion, FlashAttention, mixed precision | Triton/quantization week gets its source backbone here: HBM vs SRAM, fused kernels, online softmax, BF16/FP16/FP8 trade-offs. | FlashAttention is IO optimization: same attention result, fewer HBM reads/writes, and no materialized `T x T` score matrix. |
| 8 | Ch. 2.1-2.3 plus Ch. 10.3-10.5, pp. 23-39 and 185-199 | Serving week should reuse the memory vocabulary for KV cache, batching, and precision choices, even though the playbook focuses on training. | Distinguish training activation memory from serving KV-cache memory; both are sequence-length problems, but they appear in different parts of the lifecycle. |
| 9 | Ch. 3 and Appendix A1-A2, pp. 41-69 and 217-239 | Distributed training internals should reference DP, collective primitives, ZeRO stages, DDP overlap, and profiling. | Ring all-reduce moves about `2 * (N - 1) / N * S` bytes per rank for tensor size `S`; latency adds `2 * (N - 1)` communication steps. |
| 10 | Ch. 4-8 and Ch. 9.1-9.3, pp. 71-152 and 153-157 | Tensor/pipeline parallelism week should become a strategy comparison, not isolated tricks. Include TP, sequence parallelism, context parallelism, PP, expert parallelism, and 5D composition. | Fit memory first, hit target global batch second, optimize throughput third. TP prefers fast intra-node links; PP can tolerate slower links but pays bubble/scheduling cost. |
| 11 | Ch. 9.4-9.5 and Appendix A2, pp. 158-162 and 233-239 | Kubernetes GPU serving and observability should borrow the benchmark discipline: measure, log, explain hangs, and separate theory from cluster reality. | A theoretically valid configuration is not operationally valid until profiler traces, NCCL logs, and restart behavior are understood. |
| 12 | Ch. 8-9 and Appendix A3-A4, pp. 141-162 and 240-244 | Capstone defense should use the playbook to justify chosen parallelism, memory budget, profiling method, and what would change on an 8xH100 or larger cluster. | Compute/communication overlap is a budget test: communication can hide only when its time fits under useful compute time. |

## Week Details

<a id="week-5---transformer-memory-and-single-gpu-training"></a>
### Week 5 - Transformer Memory and Single-GPU Training

- Before implementing GPT, write the memory ledger: weights, gradients, optimizer
  states, activations.
- Treat activation recomputation as a deliberate memory-for-compute trade.
- Gate: estimate whether a toy transformer fits before you run it.

<a id="week-6---fine-tuning-memory-pressure-and-global-batch-math"></a>
### Week 6 - Fine-Tuning Memory Pressure and Global Batch Math

- Connect LoRA and QLoRA to optimizer-state and checkpoint-size pressure.
- Use global batch math to separate micro-batch, gradient accumulation, sequence length,
  and data-parallel size.
- Gate: explain which tensors still need optimizer states when the base model is frozen.

<a id="week-7---kernels-flashattention-and-mixed-precision"></a>
### Week 7 - Kernels, FlashAttention, and Mixed Precision

- Treat `torch.compile` as the first kernel optimization step, then Triton as the
  explicit implementation layer.
- Explain FlashAttention as tiling plus online softmax that avoids materializing the
  full attention matrix in HBM.
- Gate: compare FP32, FP16, BF16, and FP8 by exponent/mantissa trade-off, not only byte size.

<a id="week-8---serving-memory-and-precision-carryover"></a>
### Week 8 - Serving Memory and Precision Carryover

- Reuse the memory vocabulary from training, but distinguish activations from KV cache.
- Link batching and precision choices to observed latency, throughput, and memory.
- Gate: compute KV-cache bytes, then explain why that is not the same as training activations.

<a id="week-9---data-parallelism-zero-collectives-and-profiling"></a>
### Week 9 - Data Parallelism, ZeRO, Collectives, and Profiling

- Start from naive replicated data parallelism, then explain what DDP overlaps and what
  ZeRO/FSDP shards.
- Use Appendix A1 as the collective-operation primer for broadcast, reduce, all-reduce,
  scatter, all-gather, and reduce-scatter.
- Gate: explain why all-reduce cost is both bandwidth-bound and latency-sensitive.

<a id="week-10---model-parallelism-and-5d-strategy"></a>
### Week 10 - Model Parallelism and 5D Strategy

- Tensor parallelism shards inside layers; pipeline parallelism shards layers; context
  parallelism shards long sequence attention; expert parallelism routes tokens to MoE experts.
- Use the playbook's decision order: fit memory, hit global batch, then optimize throughput.
- Gate: justify TP vs PP on a no-NVLink 2-GPU node.

<a id="week-11---benchmarking-observability-and-cluster-reality"></a>
### Week 11 - Benchmarking, Observability, and Cluster Reality

- Promote logs and traces to deliverables: PyTorch profiler, NCCL debug logs, restart
  behavior, and resource utilization.
- Separate "algorithm works" from "cluster run is operationally healthy."
- Gate: name the one signal that would disprove your current bottleneck theory.

<a id="week-12---capstone-defense-and-scale-up-story"></a>
### Week 12 - Capstone Defense and Scale-Up Story

- Use the 5D vocabulary to explain what changes on an 8xH100 node, a multi-node cluster,
  and a long-context workload.
- Use scale estimates and overlap math to defend why the chosen capstone design is modest
  but structurally correct.
- Gate: write a paragraph that says what you would change with more GPUs and why.

## Math Cards To Add

These are the formulas and distinctions worth turning into flashcards or gates:

```text
global_batch_tokens = micro_batch * sequence_length * grad_accum_steps * data_parallel_size
```

```text
Adam training memory, rough non-activation budget:
FP32 params + FP32 grads + FP32 Adam states ~= 16 bytes/parameter
BF16 mixed precision with FP32 gradient storage can reach ~= 20 bytes/parameter
```

```text
attention_scores shape = batch * heads * sequence_length * sequence_length
```

```text
ring_all_reduce_bytes_per_rank ~= 2 * (num_ranks - 1) / num_ranks * tensor_bytes
```

```text
pipeline_efficiency ~= microbatches / (microbatches + pipeline_stages - 1)
```

```text
MFU excludes recomputation work; HFU includes the actual hardware FLOPs performed.
Use MFU to compare model throughput and HFU to understand accelerator utilization.
```

## Corrections To The Study Path

- Week 5 needs a memory ledger before GPT implementation feels real. Add parameter,
  activation, gradient, and optimizer-state accounting to notes before coding.
- Week 6 should not treat LoRA as only an application trick. It is also an optimizer-state
  and checkpoint-size strategy.
- Week 7 should frame FlashAttention as IO avoidance, not just "faster attention."
- Week 9 should explicitly connect DDP to ZeRO/FSDP: replication is the baseline;
  sharding is the correction.
- Week 10 should use the playbook's decision order: fit in memory, hit global batch,
  then tune throughput.
- Week 11 should treat profiler/NCCL logs as first-class artifacts, not optional debugging.
- Week 12 should include a written "what changes at larger scale" answer using the
  5D parallelism vocabulary.
