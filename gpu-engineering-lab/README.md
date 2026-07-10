# GPU Engineering Lab — Python, Rust & CUDA, from Kernels to Cluster

A 12-week, build-in-public portfolio: one shipped GPU-engineering project per week,
progressing from backprop-from-scratch to GPU kernels written in **Rust**, to a
**Rust inference engine** (Candle + continuous batching + paged KV), to multi-GPU
distributed training in PyTorch, to Kubernetes-served LLM inference.

**The language thesis**: *Python where the ecosystem is, Rust where performance and
reliability matter.* Model internals, fine-tuning, and distributed training happen in
PyTorch — that's where the industry lives. Kernels and the serving layer happen in
Rust — the same bet NVIDIA made when it wrote Dynamo's core in Rust. Every Rust kernel
is benchmarked against its cuBLAS/PyTorch counterpart, and the Rust serving stack ships
as a tiny container with millisecond cold-starts.

**How to follow it**: [ROADMAP.md](ROADMAP.md) is the curriculum; the day-by-day
schedule that pairs each week here with the certification study track lives in the
companion repo: [`../nvidia-cert-track/STUDY-PATH.md`](../nvidia-cert-track/STUDY-PATH.md).
Before starting each build week, clear the matching prerequisite support page in
[`../companion-lessons`](../companion-lessons/README.md).

**Why this repo exists**: certifications prove I studied; this repo proves I can build.
Every project is implemented from first principles, benchmarked against the
industry-standard implementation (cuBLAS, FlashAttention, vLLM, DDP…), and documented
with reproducible numbers.

**Hardware**: local RTX 5090 laptop (24 GB, Blackwell, sm_120) under WSL2 for weeks 1–8;
rented 2× L4/A10G cloud instances for the multi-GPU weeks (9–11).

## The progression

| Phase | Weeks | Theme | Capstone |
|-------|-------|-------|----------|
| 1 — GPU Foundations | 1–4 | backprop internals (Python) + Rust ramp → CUDA-in-Rust kernels → SGEMM optimization ladder → Rust kernels exposed to PyTorch | `rusty-kernels`: fused softmax/LayerNorm written in Rust, callable from PyTorch via PyO3, beating eager mode |
| 2 — LLM Engineering | 5–8 | GPT from scratch → LoRA from scratch → Triton kernels & quantization → Rust inference engine | `ferrum-serve`: a Rust (Candle + axum) continuous-batching, paged-KV inference server benchmarked against vLLM |
| 3 — Scale & Serve | 9–12 | distributed training → tensor/pipeline parallelism internals → K8s GPU serving | end-to-end: PyTorch fine-tune → quantize → serve the Rust engine on Kubernetes with monitoring |

## Project index & headline results

*(results tables filled in as each week ships — every project README carries its own
benchmark section with reproduction instructions)*

| Week | Project | Headline result |
|------|---------|-----------------|
| 01 | [autograd from scratch + Rust ramp](01-foundations/week-01-autograd-from-scratch/) | trains MNIST MLP, gradients match PyTorch to 1e-6 |
| 02 | [CUDA-in-Rust basics & benchmark suite](01-foundations/week-02-cuda-basics/) | |
| 03 | [SGEMM optimization ladder (Rust)](01-foundations/week-03-matmul-optimization/) | naive → __% of cuBLAS |
| 04 | [Rust kernels → PyTorch via PyO3](01-foundations/week-04-pytorch-custom-ops/) | fused LayerNorm __× vs eager |
| 05 | [GPT from scratch](02-llm-engineering/week-05-gpt-from-scratch/) | |
| 06 | [LoRA from scratch](02-llm-engineering/week-06-lora-from-scratch/) | |
| 07 | [Triton kernels & quantization](02-llm-engineering/week-07-triton-quantization/) | |
| 08 | [ferrum-serve: Rust inference engine](02-llm-engineering/week-08-mini-inference-server/) | __ tok/s @ batch __, __% of vLLM |
| 09 | [distributed training internals](03-scale-and-serve/week-09-distributed-training/) | |
| 10 | [tensor & pipeline parallelism](03-scale-and-serve/week-10-parallelism-internals/) | |
| 11 | [K8s GPU serving](03-scale-and-serve/week-11-k8s-gpu-serving/) | |
| 12 | [capstone: train→optimize→serve](03-scale-and-serve/week-12-capstone/) | |

## Repo standards

Every project follows the same contract (see [ROADMAP.md](ROADMAP.md)):

- `README.md` — what/why, results table with plots, exact reproduce commands
- `src/` — implementation; `tests/` — correctness checks (vs reference implementation)
- `bench/` — benchmark harness emitting JSON + plots; numbers in READMEs come from here
- CI runs lint + CPU-safe tests on every push

## Running anything here

See [setup/](setup/) for the WSL2 + CUDA toolchain guide and the cloud multi-GPU guide.

## License

MIT — take anything, but benchmarks were measured on my hardware; rerun before trusting.
