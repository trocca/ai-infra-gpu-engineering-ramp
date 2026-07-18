# AI Infrastructure & GPU Engineering Ramp: 12-Week NVIDIA Kubernetes-AI Curriculum

This site documents a complete, self-built ramp-up path onto the NVIDIA AI
infrastructure stack — built in 12 weeks at ~6 focused hours a day, with optional
C++/CUDA mirror drills for low-level parallelism, source-reading maps for long-form
references, and published so anyone can follow (or fork) the same path.

**If you're reviewing my candidacy**: this is both the artifact and the evidence.
The curriculum below is what I built to get ramped; designing a path others can
follow is the developer-evangelist job in miniature.

**If you're here to ramp up yourself**: fork it, start at the master plan, and
replace my dates with yours. Everything assumes one modern consumer GPU and
~$50 of cloud rentals — no DGX required.

## Start here

→ **[The Master Plan](MASTER-PLAN.md)** — the navigation hub: every week links to
the exact study material, project brief, and checkpoint for that week.

→ **[Readiness Review](READINESS-REVIEW.md)** — weaknesses in the plan for my profile,
plus the corrections that keep the 12-week run honest.

→ **[Companion Lessons](companion-lessons/README.md)** — week-by-week prerequisite
support for the math, Rust, PyTorch, CUDA, Triton, distributed training, and K8s topics.

→ **[C++/CUDA Dual Track](cpp-cuda-track/README.md)** — side-by-side CPU/GPU
parallelism drills from execution model to PyTorch extensions.

→ **[Reference Shelf](references/README.md)** — source-reading maps, including the
[Hugging Face Ultra-Scale Playbook](references/hf-ultrascale-playbook.md) for
distributed training and scaling.

## The four tracks

| Track | What | Where |
|-------|------|-------|
| **PROVE** — certifications | NCA-AIIO → NCP-GENL → NCP-AIO: weekly plans, self-checks, 300+ flashcards, full mock exams | [nvidia-cert-track](nvidia-cert-track/README.md) |
| **BUILD** — 12 shipped projects | autograd from scratch → CUDA kernels in Rust → SGEMM ladder → GPT + LoRA from scratch → Triton + quantization → a Rust inference engine → distributed training internals → Kubernetes GPU serving | [gpu-engineering-lab](gpu-engineering-lab/README.md) |
| **DRILL** — C++/CUDA mirror | execution model → memory hierarchy → reductions/scans → matmul tiling → streams/atomics → roofline → tensor cores → multi-device → PyTorch extension | [cpp-cuda-track](cpp-cuda-track/README.md) |
| **SHOW** — the demo | a narrated 7-scene demo of the Kubernetes-AI stack: TrainJob v2, KAI gang scheduling, fractional GPUs, LWS multi-node vLLM, DRA | [k8s-ai-stack-demo](k8s-ai-stack-demo/README.md) |

The design principle: **certifications give the vocabulary, source readings give the
frontier map, projects give the scars, the demo gives the story.** Study and build tracks run in parallel and are
week-aligned — the LoRA math you study Monday morning is the LoRA you implement
Monday afternoon.

## What makes this path different

- **Learn-by-doing contract**: every project ships with acceptance tests that grade
  your implementation against the industry reference (PyTorch, cuBLAS, PEFT, vLLM) —
  you can't fool yourself.
- **Honest benchmarking**: every claim in a results table states hardware, method,
  and what *didn't* work.
- **Hybrid language thesis**: Python where the ecosystem is, Rust where performance
  matters — the same bet NVIDIA made writing Dynamo's core in Rust.
- **Prerequisites made explicit**: every week has a companion lesson that turns the
  necessary math and programming support into a Sunday gate before Monday starts.
- **Gates, not vibes**: exams are sat at ≥80% mock scores; project weeks ship only
  when acceptance criteria pass; every Friday is logged.

## Reference material

- [Readiness review](READINESS-REVIEW.md) · [Companion lessons](companion-lessons/README.md) · [Reference shelf](references/README.md) · [C++/CUDA dual track](cpp-cuda-track/README.md) · [The pitch](01_pitch_and_demo.md) · [Mock interview Q&A](03_mock_interview_qa.md) · [Verified stack reference](04_stack_reference_verified.md)
