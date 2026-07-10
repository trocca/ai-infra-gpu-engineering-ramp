# Week 7 Notes — Model Optimization + GPU Acceleration

## Quantization fundamentals
### Scale/zero-point; per-tensor vs per-channel vs per-block
-
-
### Weight-only (W4A16/W8A16) vs weight+activation (W8A8) — which phase each helps
-
-
### PTQ vs QAT
-

## Formats
### INT8: activation outliers, LLM.int8(), SmoothQuant
-
-
### FP8: E4M3 vs E5M2, hardware support
-
-
### FP4 / NVFP4 (Blackwell), per-block scales
-
### KV-cache quantization
-
### Memory math: 8B model @ FP16 / INT8 / INT4
-

## PTQ methods
### GPTQ — one-line mechanism
-
### AWQ — one-line mechanism
-
### When to prefer which; calibration data role
-

## Pruning
### Unstructured vs structured
-
### 2:4 semi-structured sparsity + Sparse Tensor Cores
-
-

## Distillation
### Soft-logit KD (temperature, KL)
-
### Minitron prune-then-distill; example models
-

## Speculative decoding
### Draft/verify loop; why output distribution is unchanged
-
-
### Acceptance rate; when it helps (memory-bound decode)
-
### Medusa / EAGLE one-liners
-

## GPU fundamentals
### Tensor Cores; precision vs throughput ladder
-
### BF16 vs FP16 for training
-
### Arithmetic intensity / roofline; compute- vs memory-bound
-
-
### Prefill vs decode characteristics
-
-

## Kernel-level optimizations
### Kernel fusion — what it actually saves
-
### FlashAttention — tiling, online softmax, exact, O(N) memory
-
-
### CUDA graphs
-

## Serving-engine mechanics
### Continuous / in-flight batching
-
-
### Paged attention; prefix caching
-
-
### Chunked prefill
-
### Key vLLM knobs (gpu-memory-utilization, max-num-seqs, block size)
-

## TensorRT-LLM
### Engine build model; arch-specific engines
-
### vs vLLM trade-off (the pre-sales answer)
-
-
### TensorRT Model Optimizer role
-

## Distributed training (exam-answer form; cross-ref demo repo)
### DDP — mechanism, collective, limit
-
### ZeRO/FSDP — stages, collectives, trade-off
-
-
### Tensor parallel — why intra-node/NVLink
-
### Pipeline parallel — bubble, microbatches
-
### Megatron 3D layout rule of thumb
-
### NCCL: transports (P2P/NVLink, SHM, NET Socket/IB), ring vs tree
-
-

## Matmul timing experiment result (batch 1 vs 64)
-

## Lab observations (env vars, NCCL_DEBUG transport lines)
-
-

## Misses from self-check
-
