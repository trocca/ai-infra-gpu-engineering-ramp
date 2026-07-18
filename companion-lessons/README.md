# Companion Lessons

[<- Master Plan](../MASTER-PLAN.md) · [Readiness review](../READINESS-REVIEW.md)

These pages are the support layer for the 12-week curriculum. Use each one before the week starts. They are deliberately smaller than the weekly plans: their job is to make the math, programming, and systems prerequisites concrete enough that Monday can start with execution.

Rust support shelf: [rust-book-companion](../rust-book-companion/README.md).
C++/CUDA mirror shelf: [cpp-cuda-track](../cpp-cuda-track/README.md).
Source shelf: [references](../references/README.md), especially the
[HF Ultra-Scale Playbook map](../references/hf-ultrascale-playbook.md) for weeks 5-12.

## How to use them

1. Open the companion lesson on the Sunday before the week.
2. Do the prerequisite checklist and the gate exercise.
3. Copy weak spots into that week's `notes.md`.
4. If the gate takes more than 45 minutes, schedule a repair block before Day 1.

## Weekly Map

| Week | Companion lesson | Cert plan | Build project | C++/CUDA mirror | Source reading |
|------|------------------|-----------|---------------|-----------------|----------------|
| 1 | [AI math notation, backprop, and first Rust contact](week-01.md) | [Week 1 plan](../nvidia-cert-track/month-1-nca-aiio/week-1/plan.md) | [Autograd from scratch](../gpu-engineering-lab/01-foundations/week-01-autograd-from-scratch/README.md) | [execution model](../cpp-cuda-track/01-execution-model/README.md) | - |
| 2 | [GPU hardware math, memory bandwidth, and CUDA/Rust basics](week-02.md) | [Week 2 plan](../nvidia-cert-track/month-1-nca-aiio/week-2/plan.md) | [CUDA-in-Rust basics](../gpu-engineering-lab/01-foundations/week-02-cuda-basics/README.md) | [memory](../cpp-cuda-track/02-memory-hierarchy/README.md) + [SAXPY](../cpp-cuda-track/03-data-parallel-saxpy/README.md) | - |
| 3 | [Network fabrics, roofline thinking, and SGEMM support](week-03.md) | [Week 3 plan](../nvidia-cert-track/month-1-nca-aiio/week-3/plan.md) | [SGEMM ladder](../gpu-engineering-lab/01-foundations/week-03-matmul-optimization/README.md) | [matmul](../cpp-cuda-track/06-matmul-tiling/README.md) + [roofline](../cpp-cuda-track/09-profiling-roofline/README.md) | - |
| 4 | [Operations primitives, normalization math, and PyTorch extension support](week-04.md) | [Week 4 plan](../nvidia-cert-track/month-1-nca-aiio/week-4/plan.md) | [PyTorch custom ops in Rust](../gpu-engineering-lab/01-foundations/week-04-pytorch-custom-ops/README.md) | [PyTorch extension](../cpp-cuda-track/12-capstone-pytorch-extension/README.md) | - |
| 5 | [Transformer math, PyTorch architecture, and tokenizer support](week-05.md) | [Week 5 plan](../nvidia-cert-track/month-2-ncp-genl/week-5/plan.md) | [GPT from scratch](../gpu-engineering-lab/02-llm-engineering/week-05-gpt-from-scratch/README.md) | [reduction](../cpp-cuda-track/04-reduction/README.md) | [memory](../references/hf-ultrascale-playbook.md#week-5---transformer-memory-and-single-gpu-training) |
| 6 | [LoRA rank math, QLoRA, and evaluation support](week-06.md) | [Week 6 plan](../nvidia-cert-track/month-2-ncp-genl/week-6/plan.md) | [LoRA from scratch](../gpu-engineering-lab/02-llm-engineering/week-06-lora-from-scratch/README.md) | [scan/histogram](../cpp-cuda-track/05-scan-histogram/README.md) | [batch and ZeRO](../references/hf-ultrascale-playbook.md#week-6---fine-tuning-memory-pressure-and-global-batch-math) |
| 7 | [Quantization, Triton tiling, and FlashAttention support](week-07.md) | [Week 7 plan](../nvidia-cert-track/month-2-ncp-genl/week-7/plan.md) | [Triton kernels and quantization](../gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md) | [advanced GPU](../cpp-cuda-track/10-advanced-gpu/README.md) | [kernels](../references/hf-ultrascale-playbook.md#week-7---kernels-flashattention-and-mixed-precision) |
| 8 | [Serving math, queueing, Rust async, and safety support](week-08.md) | [Week 8 plan](../nvidia-cert-track/month-2-ncp-genl/week-8/plan.md) | [ferrum-serve](../gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md) | [async overlap](../cpp-cuda-track/07-async-overlap/README.md) | [serving carryover](../references/hf-ultrascale-playbook.md#week-8---serving-memory-and-precision-carryover) |
| 9 | [Distributed training, all-reduce cost, and NCCL support](week-09.md) | [Week 9 plan](../nvidia-cert-track/month-3-ncp-aio/week-9/plan.md) | [Distributed training internals](../gpu-engineering-lab/03-scale-and-serve/week-09-distributed-training/README.md) | [multi-device](../cpp-cuda-track/11-multi-device/README.md) | [DP/ZeRO](../references/hf-ultrascale-playbook.md#week-9---data-parallelism-zero-collectives-and-profiling) |
| 10 | [Tensor/pipeline parallelism math and admin command support](week-10.md) | [Week 10 plan](../nvidia-cert-track/month-3-ncp-aio/week-10/plan.md) | [Tensor and pipeline parallelism](../gpu-engineering-lab/03-scale-and-serve/week-10-parallelism-internals/README.md) | [atomics/memory model](../cpp-cuda-track/08-sync-atomics-memory-model/README.md) | [5D strategy](../references/hf-ultrascale-playbook.md#week-10---model-parallelism-and-5d-strategy) |
| 11 | [Kubernetes GPU serving, scheduling invariants, and observability support](week-11.md) | [Week 11 plan](../nvidia-cert-track/month-3-ncp-aio/week-11/plan.md) | [K8s GPU serving](../gpu-engineering-lab/03-scale-and-serve/week-11-k8s-gpu-serving/README.md) | [profiling roofline](../cpp-cuda-track/09-profiling-roofline/README.md) | [benchmarking](../references/hf-ultrascale-playbook.md#week-11---benchmarking-observability-and-cluster-reality) |
| 12 | [Troubleshooting, capstone integration, and portfolio defense](week-12.md) | [Week 12 plan](../nvidia-cert-track/month-3-ncp-aio/week-12/plan.md) | [Capstone](../gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md) | [PyTorch extension](../cpp-cuda-track/12-capstone-pytorch-extension/README.md) | [scale-up story](../references/hf-ultrascale-playbook.md#week-12---capstone-defense-and-scale-up-story) |
