# AI Infra & GPU Engineering Ramp

Personal ramp-up repo for AI infrastructure and GPU engineering: parallelism with
PyTorch, orchestration with Ray / Slurm / Kubernetes, and the systems layer beneath
them.

## Tracks

| Track | Path | Status |
|---|---|---|
| **C++ ↔ CUDA dual track** — same operation on both stacks, confronted module by module, from "hello, thread" to tensor cores and multi-GPU | [`cpp-cuda-track/`](cpp-cuda-track/) | active |
| PyTorch parallelism (DDP → FSDP → 3D) | _planned_ | — |
| Orchestration (Slurm, Kubernetes, Ray) | _planned_ | — |

## Reference shelf

- [The Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/) — Hugging Face, the parallelism textbook.
- *AI Systems Performance Engineering* (Fregly, O'Reilly 2025) — GPU/CUDA/PyTorch performance reference.
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — Edward Yang.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — the primary source for the CUDA half of the dual track.
